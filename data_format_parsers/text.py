# ===================================================
#   GLACIER2 RENDERTEXTURE (.TEXT / .TEXD) PARSER
#       Reads IO Interactive's Glacier2 texture maps
#       across three games. The per-mip LZ4 codec now
#       lives in utilities.py (general-purpose) so we
#       just call lz4_decompress_block() from there.
#
#   Layouts (all little-endian), confirmed byte-exact
#   against real samples + IOI's ZTextureMap.h:
#
#     Hitman: WoA / 007: First Light  (Glacier 2)
#       .TEXT = 152-byte header + a low-res PROXY mip
#               chain (per-mip raw-block LZ4). The proxy
#               is the tail of the full chain: mips
#               [scaling_width .. mip_count-1]. When
#               scaling_width == 0 the .TEXT is self-
#               contained and holds the whole chain.
#       .TEXD = HEADERLESS full-res chain, mips [0..N-1],
#               same per-mip LZ4. Its boundaries live in
#               the paired .TEXT's compressed size table.
#
#     Hitman: Absolution  (older Glacier)
#       .TEXT = 32-byte ZTextureMap::STextureMapHeader +
#               a RAW (uncompressed) mip pyramid. Self-
#               contained; Absolution has no .TEXD files.
# ===================================================

from ..io import Reader
from ..utilities import *

# The proxy inside every Glacier 2 .TEXT is the sub-128px tail of the mip chain so its top mip is always 128x128 (2048>>4, 4096>>5, ...).
# Handy when reasoning about the export proxy/TEXD split.
GLACIER2_PROXY_RESOLUTION = 128

# Hitman: World of Assassination / 007 First Light headers are a fixed 152 bytes; The texture body follows (atlas_offset + atlas_size).
GLACIER2_HEADER_SIZE = 152
ABSOLUTION_HEADER_SIZE = 32

# Human-readable enum labels, used only for logging.
TEXTURE_TYPE_NAMES = {0: "Color", 1: "Normal", 2: "Height", 3: "CompoundNormal", 4: "Billboard"}
INTERPRET_AS_NAMES = {0: "Color", 1: "Normal", 2: "Height"}
DIMENSIONS_NAMES   = {0: "2D", 1: "Cube", 2: "Volume"}

# ETextureFlags bits. Glacier 2 reuses the low bits and adds at least one higher bit (0x40 observed alongside GAMMA), surfaced as hex by the decoder.
TEXTURE_FLAG_BITS = [
    (0x01, "SWIZZLED"), (0x02, "DEFERRED"), (0x04, "XBOX360_MEM"),
    (0x08, "GAMMA"), (0x10, "EMISSIVE"), (0x20, "DDSC_ENCODED"),
]

def decode_texture_flags(flags: int) -> str:
    """Render a flags bitfield as 'GAMMA | 0x40' style text for logging (unknown bits shown as hex)."""
    known_mask = 0
    parts = []
    for bit, name in TEXTURE_FLAG_BITS:
        known_mask |= bit
        if flags & bit: parts.append(name)
    leftover = flags & ~known_mask
    if leftover: parts.append(f"0x{leftover:X}")
    return " | ".join(parts) if parts else "none"

# ===================================================
# GLACIER TEXTURE
#   One parsed .TEXT plus mip access. Game-aware:
#   Absolution uses the older flat header, WoA/Bond
#   the 152-byte streaming header.
# ===================================================

class GlacierTexture:
    """Parsed Glacier RenderTexture header plus mip-extraction helpers."""
    def __init__(self, data: bytes, game: str, source_name: str = "", verbose: bool = True):
        """Parse texture data from the games, source_name is used only for logging; verbose prints a metadata summary in the same style as the other parsers (Set False on re-parses to avoid duplicate console spam)."""
        self.data: bytes = data
        self.game: str = game
        self.source_name: str = source_name
        self.is_absolution: bool = (game == GLACIER2_ABSOLUTION)

        reader = Reader(data)
        if self.is_absolution: self.parse_absolution_header(reader)
        else: self.parse_glacier2_header(reader)

        if verbose: self.print_metadata()

    # ---------------
    # HEADER PARSERS
    # ---------------

    def parse_glacier2_header(self, reader: Reader) -> None:
        """Hitman: World of Assassination | 007: First Light - 152 bytes header."""
        self.magic = reader.ushort()             # Always 0x0001
        self.tex_type = reader.ushort()          # ERenderTextureType (0 Color, 1 Normal, ...)
        self.file_size = reader.uint32()         # Companion TEXD size + 152 (Our match key)
        self.flags = reader.uint32()
        self.width = reader.ushort()
        self.height = reader.ushort()
        self.format_code = reader.ushort()
        self.mip_count = reader.ubyte()
        self.default_mip = reader.ubyte()
        self.interpret_as = reader.ubyte()       # Glacier 2: Observed constant, Semantics unconfirmed
        self.dimensions = reader.ubyte()
        self.mip_interpolation = reader.ushort()
        self.mip_sizes = list(reader.read("14I"))            # Cumulative DECODED end offsets
        self.mip_sizes_compressed = list(reader.read("14I")) # Cumulative ON-DISK end offsets
        self.atlas_size = reader.uint32()
        self.atlas_offset = reader.uint32()
        self.scaling_data_1 = reader.ubyte()
        self.scaling_width = reader.ubyte()      # Proxy start mip index (0 => self-contained)
        self.scaling_height = reader.ubyte()
        self.scaling_data_2 = reader.ubyte()
        self.reserved = reader.uint32()

        self.header_size = GLACIER2_HEADER_SIZE
        self.body_offset = self.atlas_offset + self.atlas_size  # Proxy / self-contained body start

        # Hitman: Absolution Only Fields (Defined for a uniform interface).
        self.num_slices = 1
        self.ia_data_size = 0

    def parse_absolution_header(self, reader: Reader) -> None:
        """Hitman: Absolution - 32 bytes header."""
        self.num_slices = reader.uint32()        # 1 for a plain 2D texture
        self.total_size = reader.uint32()        # = fileSize - 4
        self.flags = reader.uint32()
        self.width = reader.ushort()
        self.height = reader.ushort()
        self.format_code = reader.ushort()
        self.mip_count = reader.ubyte()
        self.default_mip = reader.ubyte()
        self.interpret_as = reader.ubyte()       # EInterpretAs (0: Color | 1: Normal | 2: Height)
        self.dimensions = reader.ubyte()         # EDimensions  (0: 2D    | 1: Cube   | 2: Volume)
        self.mip_interpolation = reader.ubyte()
        self.pad = reader.ubyte()
        self.ia_data_size = reader.uint32()
        self.ia_data_offset = reader.uint32()

        self.header_size = ABSOLUTION_HEADER_SIZE
        self.body_offset = ABSOLUTION_HEADER_SIZE

        # Hitman: World of Assassination / 007 First Light Only Fields (Again defined for a uniform interface).
        self.tex_type = None
        self.scaling_width = 0
        self.scaling_height = 0
        self.mip_sizes = []
        self.mip_sizes_compressed = []

    # --------
    # LOGGING
    # --------

    def print_metadata(self) -> None:
        """Print a parse summary in the same shape as the PRIM/BORG parsers."""
        print(f"\nParsing RenderTexture ({self.game}): {self.source_name}\n")
        print(f"  Dimensions:    {self.width} x {self.height}  |  Format: {self.format_name} (0x{self.format_code:02X})")
        print(f"  Mip Levels:    {self.mip_count}  |  Default Mip: {self.default_mip}")

        if self.is_absolution:
            print(f"  Interpret As:  {INTERPRET_AS_NAMES.get(self.interpret_as, '?')} ({self.interpret_as})  |  "f"Dimensions: {DIMENSIONS_NAMES.get(self.dimensions, '?')}  |  Non-Color: {self.is_non_color()}")
            print(f"  Flags:         0x{self.flags:08X}  [{decode_texture_flags(self.flags)}]")
            print(f"  Slices:        {self.num_slices}  |  IA Data: {self.ia_data_size} bytes")
        else:
            print(f"  Texture Type:  {TEXTURE_TYPE_NAMES.get(self.tex_type, '?')} ({self.tex_type})  |  Non-Color: {self.is_non_color()}")
            print(f"  Flags:         0x{self.flags:08X}  [{decode_texture_flags(self.flags)}]")
            print(f"  Streaming:     proxy start mip {self.scaling_width}  |  companion TEXD: {self.has_companion_texd}")
            if self.has_companion_texd: print(f"  Match Key:     file_size {self.file_size}  ->  expected TEXD size {self.expected_texd_size}")

        print(f"\nRENDERTEXTURE PARSING COMPLETE!\n")

    # -------------------
    # DERIVED PROPERTIES
    # -------------------

    @property
    def format_meta(self) -> dict:
        """The resolved render-format descriptor for this texture's game + code."""
        return get_render_format(self.game, self.format_code)

    @property
    def format_name(self) -> str:
        """Canonical format family name, e.g. 'BC1', 'BC5', 'R8G8B8A8'."""
        return self.format_meta["name"]

    @property
    def has_companion_texd(self) -> bool:
        """True when this Glacier 2 .TEXT streams from a separate .TEXD (i.e. not self-contained)."""
        return (not self.is_absolution) and self.scaling_width > 0

    @property
    def expected_texd_size(self) -> int | None:
        """Byte size the paired .TEXD must have (== file_size - 152 == compressed chain total)."""
        if not self.has_companion_texd: return None
        return self.mip_sizes_compressed[self.mip_count - 1]

    def mip_dimensions(self, level: int) -> tuple[int, int]:
        """Width/height of mip `level`. Floor-shift + clamp - correct for NPOT as well as POT."""
        return max(1, self.width >> level), max(1, self.height >> level)

    def is_non_color(self) -> bool:
        """Decide sRGB vs linear from the HEADER (names are hashed so suffix detection is useless). Data-only formats (single/two channel) are always linear; Otherwise we trust the texture-type / interpret-as field the engine itself set."""
        if self.format_name in ("BC4", "BC5", "R8G8", "A8"): return True
        if self.is_absolution: return self.interpret_as in (1, 2)   # Normal, Height
        return self.tex_type in (1, 2, 3)                           # Normal, Height, CompoundNormal

    # ---------------------------------------------------
    # MIP EXTRACTION
    # ---------------------------------------------------

    def iter_source_mips(self, source: bytes, source_body_offset: int, start_level: int):
        """Yield (level, w, h, compressed_bytes, decoded_size) for a mip source, highest first. source is either a paired .TEXD (start_level 0, body offset 0) or this .TEXT itself (start_level = scaling_width, body offset = self.body_offset). The compressed table is cumulative over the FULL chain, so proxy offsets are rebased by the first included mip."""
        base = self.mip_sizes_compressed[start_level - 1] if start_level > 0 else 0
        for level in range(start_level, self.mip_count):
            comp_start = self.mip_sizes_compressed[level - 1] if level > 0 else 0
            comp_end = self.mip_sizes_compressed[level]
            dec_start = self.mip_sizes[level - 1] if level > 0 else 0
            dec_end = self.mip_sizes[level]

            offset = source_body_offset + (comp_start - base)
            comp = source[offset:offset + (comp_end - comp_start)]
            width, height = self.mip_dimensions(level)
            yield level, width, height, bytes(comp), (dec_end - dec_start)

    def get_highest_mip(self, texd_bytes: bytes | None = None) -> tuple[int, int, bytes]:
        """Return (width, height, decoded_texture_bytes) for the best available mip.

        Preference order, matching the requested behavior:
          1. If a paired .TEXD is supplied  -> its mip 0 (full resolution).
          2. Otherwise                      -> the .TEXT's own top internal mip (proxy mip 0, or the true mip 0 for a self-contained .TEXT).
        We walk from the highest mip downward and return the first that decodes cleanly so a single corrupt/truncated top mip degrades gracefully to the next resolution instead of failing the whole import. The returned bytes are the DECODED texture-format payload (BCn blocks or raw pixels) - LZ4 already stripped, ready to wrap in a DDS."""
        # Absolution: Raw pyramid, mip 0 sits right after the 32-byte header.
        if self.is_absolution:
            width, height = self.mip_dimensions(0)
            size = mip_size_for(self.format_meta, width, height)
            return width, height, bytes(self.data[self.body_offset:self.body_offset + size])

        # Glacier 2: Choose the source and its starting mip.
        if texd_bytes is not None: source, source_body_offset, start_level = texd_bytes, 0, 0
        else: source, source_body_offset, start_level = self.data, self.body_offset, self.scaling_width

        last_error: Exception | None = None
        for level, width, height, comp, decoded_size in self.iter_source_mips(source, source_body_offset, start_level):
            try: # Equal compressed/decoded size means the mip is stored raw (no LZ4).
                pixels = comp if len(comp) == decoded_size else bytes(lz4_decompress_block(comp, decoded_size))
                return width, height, pixels
            except Exception as error:  # fall through to the next-lower mip
                last_error = error
                continue

        raise RuntimeError(f"No decodable mip found for texture (last error: {last_error})")
