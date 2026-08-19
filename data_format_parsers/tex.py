# =====================================================
#   GLACIER 1 TEX (TEXTURE ARCHIVE) EXTRACTOR
#       Parses IO Interactive's Glacier 1 texture
#       archives used by:
#           - Hitman 2: Silent Assassin
#           - Hitman: Contracts
#           - Hitman: Blood Money (and Mini Ninjas, same tech)
#           - Freedom Fighters
#
#       Mirrors TEX.bt. If you change the binary template,
#       mirror the change here (and vice versa).
#
#       A .TEX is a flat archive of many textures. Each
#       texture stores its raw pixel payload (block-
#       compressed DXT, uncompressed RGBA / I8 / U8V8, or
#       an 8-bit palette-indexed PALN blob) plus enough
#       metadata to reconstruct a standalone .DDS around
#       it. This module walks the archive, rebuilds a DDS
#       header per texture, writes every DDS out to a
#       folder beside the .TEX, and (optionally) loads the
#       results into Blender.
#
#       DDS header construction follows GlacierTEXEditor's
#       reference (glacier-modding), which round-trips TEX
#       <-> DDS in-engine; the field order and the block-
#       size maths here match it byte-for-byte.
# =====================================================

import struct, os
from ..io import Reader
from ..utilities import *

# ==========
# CONSTANTS
# ==========

# The header is four uints. The first is the byte offset of the primary offset TABLE, which sits
# in the last 16 KiB of the file (fileSize - 0x4000); the second points at a secondary table
# 8 KiB further in. Everything between the 16-byte header and the primary table offset is texture
# payload. Enumerating textures through the offset table (rather than walking payloads and
# guessing gap sizes) is what GlacierTEXEditor does and is far more robust than a sequential scan.
TEX_HEADER_SIZE            = 16
TEX_PRIMARY_TABLE_ENTRIES  = 2048   # 8 KiB / 4 bytes: one u32 slot per possible texture id
TEX_FALLBACK_TABLE_OFFSET  = 0x210  # older layout: table sits at a fixed offset instead
TEX_FALLBACK_TABLE_END     = 0x2010

# Fixed portion of a texture entry, BEFORE the null-terminated name. See parse_entry for the
# field-by-field decode. 36 bytes: imageSize + format(4) + formatDup(4) + id + H + W + mipCount
# + flagA + flagB + reserved(4).
TEX_ENTRY_FIXED_SIZE = 36

# On-disk format codes are the readable four-character code stored REVERSED (little-endian tag).
# Reversing "1TXD" gives "DXT1", "ABGR" gives "RGBA", and so on. We reverse on read and work with
# the readable form everywhere below.
TEX_FORMAT_DXT1 = "DXT1"
TEX_FORMAT_DXT3 = "DXT3"
TEX_FORMAT_RGBA = "RGBA"
TEX_FORMAT_PALN = "PALN"   # 8-bit palette-indexed; palette appended after the indexed pixels
TEX_FORMAT_I8   = "I8  "   # 8-bit luminance (note the two trailing spaces in the on-disk tag)
TEX_FORMAT_U8V8 = "U8V8"   # two-channel signed bump map

# The complete set of tags we recognise, in their readable (reversed) form.
TEX_KNOWN_FORMATS = (TEX_FORMAT_DXT1, TEX_FORMAT_DXT3, TEX_FORMAT_RGBA, TEX_FORMAT_PALN, TEX_FORMAT_I8, TEX_FORMAT_U8V8)

# Block-compressed byte counts per 4x4 block. Uncompressed formats carry bytes-per-pixel instead.
TEX_DXT1_BLOCK_BYTES = 8
TEX_DXT3_BLOCK_BYTES = 16
TEX_RGBA_BYTES_PER_PIXEL = 4
TEX_I8_BYTES_PER_PIXEL   = 1
TEX_U8V8_BYTES_PER_PIXEL = 2
TEX_PALETTE_ENTRY_BYTES  = 4  # VEC4UB per palette colour

# --- DDS header field values (see build_dds_header) ---
DDS_MAGIC = b"DDS "
DDS_HEADER_SIZE = 124
DDS_PIXELFORMAT_SIZE = 32

# DDSD_* surface-description flags.
DDSD_CAPS        = 0x1
DDSD_HEIGHT      = 0x2
DDSD_WIDTH       = 0x4
DDSD_PITCH       = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE  = 0x80000

# DDPF_* pixel-format flags.
DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC      = 0x4
DDPF_RGB         = 0x40
DDPF_LUMINANCE   = 0x20000

# DDSCAPS_* capability flags.
DDSCAPS_COMPLEX = 0x8
DDSCAPS_TEXTURE = 0x1000
DDSCAPS_MIPMAP  = 0x400000

# The games this extractor accepts. Mini Ninjas rides on GLACIER1_HBM (same texture tech).
GLACIER1_TEX_SUPPORTED = (GLACIER1_H2SA, GLACIER1_HMC, GLACIER1_HBM, GLACIER1_FIGHTERS)

# =====================================================
# SINGLE-TEXTURE RECORD
# =====================================================

class GlacierTexEntry():
    """One decoded texture out of a .TEX archive: its format, dimensions, name, mip payloads and
    (for PALN) its palette. Knows how to turn itself into a standalone .DDS byte string."""
    def __init__(self, texture_id: int, image_format: str, width: int, height: int, mip_count: int, name: str, mips: list[bytes], palette: bytes, flag_a: int, flag_b: int):
        self.texture_id: int = texture_id
        """The archive-global texture id. Differentiates textures that share a name."""

        self.image_format: str = image_format
        """Readable format code (DXT1 / DXT3 / RGBA / PALN / I8 / U8V8)."""

        self.width: int = width
        """Texture width in pixels."""

        self.height: int = height
        """Texture height in pixels."""

        self.mip_count: int = mip_count
        """Number of mip levels stored."""

        self.name: str = name
        """The texture's stored name. May be empty and may contain subdirectories (slashes)."""

        self.mips: list[bytes] = mips
        """Raw payload of each mip, largest first. For PALN these are palette INDICES."""

        self.palette: bytes = palette
        """PALN only: the appended [count][VEC4UB...] palette bytes (colours in RGBA order)."""

        self.flag_a: int = flag_a
        self.flag_b: int = flag_b

    # -------------------------------------------------
    # DDS RECONSTRUCTION
    # -------------------------------------------------

    def to_dds(self) -> Optional[bytes]:
        """Rebuild a standalone .DDS around this texture's payload.

        Block-compressed and directly-DDS-expressible formats (DXT1/DXT3/RGBA/I8/U8V8) keep their
        raw mip bytes and just gain a header. PALN has no native DDS representation, so its single
        indexed image is expanded through the palette into a 32-bit A8B8G8R8 surface. Returns None
        for a format we cannot express."""
        if self.image_format in (TEX_FORMAT_DXT1, TEX_FORMAT_DXT3, TEX_FORMAT_RGBA, TEX_FORMAT_I8, TEX_FORMAT_U8V8):
            pixel_body = self.rgba_to_bgra_all_mips() if self.image_format == TEX_FORMAT_RGBA else b"".join(self.mips)
            return build_dds_header(self.image_format, self.width, self.height, self.mip_count, self.top_mip_size()) + pixel_body

        if self.image_format == TEX_FORMAT_PALN:
            expanded = self.expand_palette()
            if expanded is None: return None
            # A palette expansion produces one full-res A8B8G8R8 surface; treat as a single mip.
            return build_dds_header(TEX_FORMAT_RGBA, self.width, self.height, 1, len(expanded)) + expanded

        return None

    def top_mip_size(self) -> int:
        """Size in bytes of mip 0, computed from the format + dimensions (NOT trusted from disk).

        This is the value the DDS header's pitchOrLinearSize field carries, and getting it right
        is what keeps NON-POWER-OF-TWO textures from shearing: block-compressed formats round the
        dimensions UP to whole 4x4 blocks via ceiling division, exactly as the hardware samples
        them. A 258x130 DXT1 image is (65 x 33) blocks, not (64.5 x 32.5)."""
        return calculate_image_size(self.image_format, self.width, self.height)

    def rgba_to_bgra_all_mips(self) -> bytes:
        """Reorder every RGBA mip to the B,G,R,A byte order an A8B8G8R8 DDS expects on disk.

        The DDS A8B8G8R8 masks (R=0x00FF0000, G=0x0000FF00, B=0x000000FF, A=0xFF000000) put blue
        in the low byte, so a texture stored R,G,B,A on disk must be written B,G,R,A."""
        out = bytearray()
        for mip in self.mips:
            reordered = bytearray(len(mip))
            for pixel in range(0, len(mip) - 3, 4):
                r, g, b, a = mip[pixel], mip[pixel + 1], mip[pixel + 2], mip[pixel + 3]
                reordered[pixel]     = b
                reordered[pixel + 1] = g
                reordered[pixel + 2] = r
                reordered[pixel + 3] = a
            out += reordered
        return bytes(out)

    def expand_palette(self) -> Optional[bytes]:
        """Expand a PALN index image (mip 0) through its palette into a B,G,R,A surface.

        The palette is [u32 count][count * VEC4UB], each entry stored R,G,B,A. We look each index
        up and write B,G,R,A so the result drops straight into an A8B8G8R8 DDS. Returns None if
        the palette or index data is missing/short."""
        if not self.palette or not self.mips: return None
        if len(self.palette) < 4: return None

        palette_count = struct.unpack_from("<I", self.palette, 0)[0]
        colours: list[tuple[int, int, int, int]] = []
        for index in range(palette_count):
            base = 4 + index * TEX_PALETTE_ENTRY_BYTES
            if base + TEX_PALETTE_ENTRY_BYTES > len(self.palette): break
            r, g, b, a = self.palette[base], self.palette[base + 1], self.palette[base + 2], self.palette[base + 3]
            colours.append((r, g, b, a))
        if not colours: return None

        indices = self.mips[0]
        out = bytearray(len(indices) * 4)
        for position, index in enumerate(indices):
            r, g, b, a = colours[index] if index < len(colours) else (0, 0, 0, 0)
            out[position * 4]     = b
            out[position * 4 + 1] = g
            out[position * 4 + 2] = r
            out[position * 4 + 3] = a
        return bytes(out)

    # -------------------------------------------------
    # NAMING
    # -------------------------------------------------

    def output_relative_path(self) -> str:
        """Relative path (with subfolders preserved) for this texture's .dds inside the extraction
        folder. Duplicate names are disambiguated by the texture id, which is always unique:

            'weapons/silverballer'  ->  weapons/(1234)_silverballer.dds
            ''                      ->  (1234)_Unnamed_Texture.dds

        Slashes in the stored name become real subdirectories; a nameless texture falls back to
        an id-tagged placeholder so nothing is ever silently overwritten or dropped."""
        # Stored names use either separator; normalise to the OS one and split off the leaf.
        normalised = self.name.replace("\\", "/").strip("/")
        if normalised:
            directory, _, leaf = normalised.rpartition("/")
            if not leaf: leaf = "Unnamed_Texture"
        else:
            directory, leaf = "", "Unnamed_Texture"

        filename = f"({self.texture_id})_{leaf}.dds"
        return os.path.join(*directory.split("/"), filename) if directory else filename

# =====================================================
# DDS + SIZE HELPERS
# =====================================================

def calculate_image_size(image_format: str, width: int, height: int) -> int:
    """Bytes occupied by a single mip of the given format at the given dimensions.

    Block-compressed formats use CEILING division into 4x4 blocks - the `max(1, (n + 3) // 4)`
    idiom - so a non-multiple-of-four dimension rounds up to a whole block row/column rather than
    truncating. This is the single most important calculation for non-power-of-two correctness."""
    if image_format == TEX_FORMAT_DXT1:
        return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * TEX_DXT1_BLOCK_BYTES
    if image_format == TEX_FORMAT_DXT3:
        return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * TEX_DXT3_BLOCK_BYTES
    if image_format in (TEX_FORMAT_PALN, TEX_FORMAT_I8):
        return width * height * TEX_I8_BYTES_PER_PIXEL
    if image_format == TEX_FORMAT_U8V8:
        return width * height * TEX_U8V8_BYTES_PER_PIXEL
    return width * height * TEX_RGBA_BYTES_PER_PIXEL  # RGBA / A8B8G8R8

def build_dds_header(image_format: str, width: int, height: int, mip_count: int, top_mip_size: int) -> bytes:
    """Assemble a 128-byte legacy DDS header (magic + 124-byte DDS_HEADER) for the given format.

    Mirrors GlacierTEXEditor's ExportDDSFile field-for-field:
      - block-compressed + RGBA/U8V8 use DDSD_LINEARSIZE with pitchOrLinearSize = mip0 size;
      - I8 (written as an L8 luminance surface) uses DDSD_PITCH with pitch = width * 1;
      - the pixel-format block carries the FourCC for DXT, or explicit channel masks otherwise."""
    is_block = image_format in (TEX_FORMAT_DXT1, TEX_FORMAT_DXT3)
    is_i8    = image_format == TEX_FORMAT_I8

    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    if is_i8:
        flags |= DDSD_PITCH
        pitch_or_linear = width * TEX_I8_BYTES_PER_PIXEL
    else:
        flags |= DDSD_LINEARSIZE
        pitch_or_linear = top_mip_size
    if mip_count > 1:
        flags |= DDSD_MIPMAPCOUNT

    # --- DDS_PIXELFORMAT (32 bytes) ---
    if is_block:
        four_cc = image_format.encode("ascii")  # 'DXT1' / 'DXT3'
        pf = struct.pack("<II4sIIIII", DDS_PIXELFORMAT_SIZE, DDPF_FOURCC, four_cc, 0, 0, 0, 0, 0)
    elif image_format == TEX_FORMAT_I8:
        # Luminance-8 (L8): single 8-bit channel in the red mask.
        pf = struct.pack("<II4sIIIII", DDS_PIXELFORMAT_SIZE, DDPF_LUMINANCE, b"\x00\x00\x00\x00", 8, 0xFF, 0, 0, 0)
    elif image_format == TEX_FORMAT_U8V8:
        # Two 8-bit channels. Written as an RGB surface with R/G masks so viewers show both.
        pf = struct.pack("<II4sIIIII", DDS_PIXELFORMAT_SIZE, DDPF_RGB, b"\x00\x00\x00\x00", 16, 0x00FF, 0xFF00, 0, 0)
    else:
        # RGBA / palette-expanded -> A8B8G8R8. Masks put blue low (see rgba_to_bgra).
        pf = struct.pack("<II4sIIIII", DDS_PIXELFORMAT_SIZE, DDPF_RGB | DDPF_ALPHAPIXELS, b"\x00\x00\x00\x00", 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)

    # --- DDSCAPS ---
    caps1 = DDSCAPS_TEXTURE
    if mip_count > 1: caps1 |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP

    # --- DDS_HEADER (124 bytes): size, flags, height, width, pitch, depth, mips, 11 reserved,
    #     then the 32-byte pixel format, then caps(4) + reserved. ---
    header = struct.pack(
        "<7I44x", DDS_HEADER_SIZE, flags, height, width, pitch_or_linear, 0, mip_count,
    )  # 7 uints + 44 bytes of reserved (11 uints)
    header += pf
    header += struct.pack("<5I", caps1, 0, 0, 0, 0)  # caps1, caps2, caps3, caps4, reserved2

    return DDS_MAGIC + header

# =====================================================
# MAIN PARSER CLASS
# =====================================================

class TEX():
    """Glacier 1 texture-archive parser. Decodes every texture into a `GlacierTexEntry`."""
    def __init__(self, file_path: str, game: str):
        """Construct the parser and run the full parse pass."""

        super().__init__()

        # ===============================
        # == CLASS MEMBERS ==============
        # ===============================

        self.archive_file: str = file_path
        """Path to the source `.TEX` file."""

        self.game: str = game
        """Which Glacier 1 title produced this file. Mini Ninjas rides on GLACIER1_HBM."""

        self.header: dict = {}
        """Decoded 16-byte header."""

        self.entries: list[GlacierTexEntry] = []
        """Every decoded texture, in offset-table order."""

        self.buffer: bytes = b""
        """Raw file bytes, retained for random-access entry decode."""

        if game not in GLACIER1_TEX_SUPPORTED: raise ValueError(f"Unsupported game type for TEX parsing: '{game}'. Use GLACIER1_H2SA, GLACIER1_HMC, GLACIER1_HBM or GLACIER1_FIGHTERS.")

        self.parse_archive()

    # =====================================================
    # TOP-LEVEL DRIVER
    # =====================================================

    def parse_archive(self) -> None:
        """Parse the archive: header, then every texture the offset table points at."""
        print(f"\nParsing TEX archive ({self.game}): {self.archive_file}\n")
        self.buffer = open(self.archive_file, "rb").read()
        reader = Reader(self.buffer)

        # ---------------- HEADER ----------------
        primary_table_offset   = reader.uint32()
        secondary_table_offset = reader.uint32()
        header_field_2         = reader.uint32()
        version               = reader.uint32()
        self.header = {
            "primary_table_offset":   primary_table_offset,
            "secondary_table_offset": secondary_table_offset,
            "version":                version,
        }

        table_offset, table_entries = self.resolve_offset_table(primary_table_offset)
        print(f"Offset table: {table_entries} slots at 0x{table_offset:08X}")

        # ---------------- WALK THE OFFSET TABLE ----------------
        seen_offsets: set[int] = set()
        for slot in range(table_entries):
            reader.seek(table_offset + 4 * slot)
            entry_offset = reader.uint32()
            if entry_offset == 0 or entry_offset in seen_offsets: continue
            if not (TEX_HEADER_SIZE <= entry_offset < len(self.buffer)): continue
            seen_offsets.add(entry_offset)

            entry = self.parse_entry(reader, entry_offset)
            if entry is not None: self.entries.append(entry)

        print(f"\nTEX PARSING COMPLETE!  {len(self.entries)} textures decoded.\n")

    def resolve_offset_table(self, primary_table_offset: int) -> tuple[int, int]:
        """Pick the offset table to walk. The modern layout stores its location in the header and
        holds 2048 slots; a fallback older layout keeps the table at a fixed offset. We take the
        header's pointer when it lands inside the file, else the fixed fallback."""
        if 0 < primary_table_offset < len(self.buffer):
            return primary_table_offset, TEX_PRIMARY_TABLE_ENTRIES
        return TEX_FALLBACK_TABLE_OFFSET, (TEX_FALLBACK_TABLE_END - TEX_FALLBACK_TABLE_OFFSET) // 4

    # =====================================================
    # ENTRY DECODE
    # =====================================================

    def parse_entry(self, reader: Reader, entry_offset: int) -> Optional[GlacierTexEntry]:
        """Decode one texture entry at `entry_offset`.

        Fixed header (36 bytes):
            +0x00 uint   imageSize        (payload + metadata; not needed for the DDS rebuild)
            +0x04 char4  imageFormat      (stored REVERSED - '1TXD' on disk is 'DXT1')
            +0x08 char4  imageFormatDup   (a duplicate, ignored)
            +0x0C uint   imageID
            +0x10 ushort imageHeight      <-- HEIGHT is stored first
            +0x12 ushort imageWidth       <-- then WIDTH
            +0x14 uint   imageMipCount
            +0x18 uint   imageFlagA
            +0x1C uint   imageFlagB       (a float on Blood Money / Mini Ninjas)
            +0x20 uint   reserved
        then a null-terminated name, then each mip as [uint mipSize][mipData], then - for PALN
        only - an appended [uint paletteCount][paletteCount * VEC4UB] palette.

        Height-before-width is the ordering GlacierTEXEditor uses; reading them the other way
        round leaves non-square textures sheared (the sawtooth artefact). Returns None if the
        entry does not decode to a known format."""
        if entry_offset + TEX_ENTRY_FIXED_SIZE > len(self.buffer): return None

        reader.seek(entry_offset)
        image_size    = reader.uint32()
        raw_format    = reader.read_bytes(4)
        raw_format    = raw_format.tobytes() if hasattr(raw_format, "tobytes") else bytes(raw_format)
        reader.skip(4)                       # duplicate format tag
        texture_id    = reader.uint32()
        image_height  = reader.ushort()      # HEIGHT first
        image_width   = reader.ushort()      # WIDTH second
        mip_count     = reader.uint32()
        flag_a        = reader.uint32()
        flag_b        = reader.uint32()
        reader.skip(4)                       # reserved

        image_format = raw_format[::-1].decode("ascii", errors="replace")  # reverse to readable
        if image_format not in TEX_KNOWN_FORMATS: return None
        if image_width == 0 or image_height == 0 or mip_count == 0: return None

        # --- NAME (null-terminated; may carry subdirectories, may be empty) ---
        name = reader.read_null_terminated_string(encoding="ascii", errors="replace")

        # --- MIP CHAIN ---
        mips: list[bytes] = []
        for _ in range(mip_count):
            if reader.tell() + 4 > len(self.buffer): break
            mip_size = reader.uint32()
            if mip_size == 0:
                mips.append(b"")
                continue
            if reader.tell() + mip_size > len(self.buffer): break
            chunk = reader.read_bytes(mip_size)
            mips.append(chunk.tobytes() if hasattr(chunk, "tobytes") else bytes(chunk))

        # --- PALETTE (PALN only) ---
        palette = b""
        if image_format == TEX_FORMAT_PALN and reader.tell() + 4 <= len(self.buffer):
            palette_count = struct.unpack_from("<I", self.buffer, reader.tell())[0]
            palette_length = 4 + palette_count * TEX_PALETTE_ENTRY_BYTES
            if reader.tell() + palette_length <= len(self.buffer):
                palette_bytes = reader.read_bytes(palette_length)
                palette = palette_bytes.tobytes() if hasattr(palette_bytes, "tobytes") else bytes(palette_bytes)

        return GlacierTexEntry(texture_id, image_format, image_width, image_height, mip_count, name, mips, palette, flag_a, flag_b)

# =====================================================================================================================================================
# EXTRACTION
# =====================================================================================================================================================

def extraction_root_for(tex_path: str) -> str:
    """The folder textures extract into: a directory beside the .TEX with the SAME stem and NO
    '_extracted' suffix. `/game/M00.TEX` -> `/game/M00/`."""
    tex_file = Path(tex_path)
    return str(tex_file.parent / tex_file.stem)

def extract_archive(tex: TEX, output_root: str) -> list[tuple[GlacierTexEntry, str]]:
    """Rebuild a .DDS for every decoded texture and write it under `output_root`, preserving the
    stored subfolder hierarchy and disambiguating duplicate names by id. Returns the list of
    (entry, absolutePath) pairs actually written, so the caller can optionally load them."""
    written: list[tuple[GlacierTexEntry, str]] = []
    skipped = 0

    for entry in tex.entries:
        dds_bytes = entry.to_dds()
        if dds_bytes is None:
            print(f"  Skipping texture {entry.texture_id} ('{entry.name or 'unnamed'}'): unsupported format {entry.image_format!r}.")
            skipped += 1
            continue

        relative_path = entry.output_relative_path()
        absolute_path = os.path.join(output_root, relative_path)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        with open(absolute_path, "wb") as handle: handle.write(dds_bytes)
        written.append((entry, absolute_path))

    print(f"Extraction: wrote {len(written)} DDS files to '{output_root}'" + (f" ({skipped} unsupported skipped)." if skipped else "."))
    return written

# =====================================================================================================================================================
# BLENDER IMPORT
# =====================================================================================================================================================

def load_dds_into_blender(dds_path: str) -> "Optional[bpy.types.Image]":
    """Load one extracted .DDS into Blender as an image datablock, named after the file on disk.

    Blender decodes DXT1/DXT3 and uncompressed DDS natively. The datablock is named after the
    .dds file (which already carries the id tag and, for duplicates, is unique) rather than the
    texture's stored name, so name collisions in the archive never collide in Blender."""
    try:
        image = bpy.data.images.load(dds_path, check_existing=False)
    except Exception as error:
        print(f"  Blender could not load {os.path.basename(dds_path)}: {error}")
        return None

    if image is None or image.size[0] == 0:
        if image is not None: bpy.data.images.remove(image)
        print(f"  Blender loaded {os.path.basename(dds_path)} but it decoded empty; leaving it on disk.")
        return None

    image.name = os.path.basename(dds_path)
    image.alpha_mode = "CHANNEL_PACKED"
    return image

def import_tex_archive(self, context, file_path: str, game: str, import_to_blender: bool = False) -> set[str]:
    """Import a Glacier 1 TEX texture archive.

    Always: parse the archive, rebuild a .DDS per texture, and write them into a folder beside the
    .TEX with the same stem (no '_extracted' suffix), preserving subfolders and disambiguating
    duplicate names by id.

    When `import_to_blender` is True, additionally load every extracted .DDS into Blender as an
    image datablock. When False, the operator is a pure extract-and-rebuild pass and touches no
    Blender data.

    Args:
        self:               Calling operator (for self.report()).
        context:            Blender context.
        file_path:          Absolute path to the .TEX file.
        game:               GLACIER1_H2SA / HMC / HBM / FIGHTERS.
        import_to_blender:  Load the rebuilt DDS files into Blender as well as writing them.

    Returns a Blender operator result set.
    """
    print(f"\nIMPORTING GLACIER 1 TEX ARCHIVE: {file_path}...\n")

    if not os.path.exists(file_path):
        self.report({'ERROR'}, f"TEX file not found: {file_path}")
        return {'CANCELLED'}

    if game not in GLACIER1_TEX_SUPPORTED:
        self.report({'ERROR'}, f"Unsupported game type for Glacier 1 TEX import: {game}")
        return {'CANCELLED'}

    # --- PARSE ---
    try:
        tex = TEX(file_path, game)
    except Exception as exc:
        self.report({'ERROR'}, f"Failed to parse TEX: {exc}")
        print(f"TEX parse failure: {exc}")
        return {'CANCELLED'}

    if not tex.entries:
        self.report({'WARNING'}, "TEX contains no decodable textures.")
        return {'CANCELLED'}

    # --- EXTRACT + REBUILD ---
    output_root = extraction_root_for(file_path)
    try:
        os.makedirs(output_root, exist_ok=True)
        written = extract_archive(tex, output_root)
    except Exception as exc:
        self.report({'ERROR'}, f"Failed to extract TEX: {exc}")
        print(f"TEX extraction failure: {exc}")
        return {'CANCELLED'}

    # --- OPTIONAL BLENDER LOAD ---
    loaded = 0
    if import_to_blender:
        for _, dds_path in written:
            if load_dds_into_blender(dds_path) is not None: loaded += 1
        print(f"Loaded {loaded} of {len(written)} textures into Blender.")

    summary = f'Extracted {len(written)} textures from "{Path(file_path).name}" to "{Path(output_root).name}/"'
    if import_to_blender: summary += f" and loaded {loaded} into Blender"
    summary += "."
    self.report({'INFO'}, summary)
    print(summary)

    return {'FINISHED'}
