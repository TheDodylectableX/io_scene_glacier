# ===================================================
#   GLACIER TEXTURE HANDLER
#       Import : bulk .TEXT -> (pair with .TEXD) ->
#                decode highest mip -> DDS -> Blender.
#       Export : Blender image -> texconv BCn DDS ->
#                LZ4 -> rebuild .TEXT (+ .TEXD).
#
#   Names are hashed, so nothing lines up by filename.
#   We pair a .TEXT with its .TEXD by SIZE (the header
#   file_size field == TEXD size + 152) and then CONFIRM
#   byte-exact, since the .TEXT proxy is literally the
#   tail of its .TEXD. That makes a size collision
#   impossible to mis-resolve.
#
#   When a WoA/007FL .TEXT's companion .TEXD isn't beside
#   it, we still import immediately at the internal proxy
#   resolution, then offer a folder prompt to UPGRADE
#   those textures to full res. Cancel = keep the proxy.
# ===================================================

from ..io import Writer
from ..utilities import *  # bpy, os, time, Path, subprocess, tempfile, defaultdict, LZ4 + format helpers
from ..data_format_parsers.text import GlacierTexture, GLACIER2_HEADER_SIZE, GLACIER2_PROXY_RESOLUTION

# DDS header bit flags we actually set.
DDSD_CAPS = 0x1; DDSD_HEIGHT = 0x2; DDSD_WIDTH = 0x4; DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000; DDSD_LINEARSIZE = 0x80000
DDPF_FOURCC = 0x4
DDSCAPS_TEXTURE = 0x1000
DDS_DX10_RESOURCE_DIMENSION_TEXTURE2D = 3

# Cross-operator handoff for the deferred "locate .TEXD folder" step. The import operator fills
# this in, then invokes the folder picker, which reads it back. Overwritten on every import, so a
# cancelled prompt just leaves harmless stale data that the next import replaces.
pending_texd_upgrades = {"orphans": [], "game": None}

# ===================================================
# DDS CONSTRUCTION
# ===================================================

def create_dds_file(pixel_data: bytes, width: int, height: int, dxgi_id: int, linear_size: int) -> bytes:
    """Wrap a single decoded mip in a 148-byte DX10 DDS header. Returns the full DDS bytes.

    We always emit the DX10 (extended) header so the DXGI format is explicit - no guessing from
    legacy FourCCs. `linear_size` is the FULL top-mip surface size (not a row pitch): under
    DDSD_LINEARSIZE a row-pitch value shears non-power-of-two textures in compliant decoders, so
    this must be the whole-surface byte count (mip_size_for()).
    """
    dds = bytearray()
    writer = Writer(dds)

    writer.ascii_string("DDS ")                          # magic
    writer.uint32(124)                                   # header size (excludes magic)
    writer.uint32(DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT | DDSD_MIPMAPCOUNT | DDSD_LINEARSIZE)
    writer.uint32(height)
    writer.uint32(width)
    writer.uint32(linear_size)                           # pitchOrLinearSize = full mip0 surface bytes
    writer.uint32(0)                                     # depth
    writer.uint32(1)                                     # mip count (single highest mip)
    writer.write("11I", *([0] * 11))                     # reserved1

    # DDS_PIXELFORMAT (32 bytes) - DX10 marker
    writer.uint32(32)                                    # pixelformat size
    writer.uint32(DDPF_FOURCC)                           # flags
    writer.ascii_string("DX10")                          # fourCC
    writer.write("5I", *([0] * 5))                       # RGBBitCount + 4 masks (unused for DX10)

    # caps
    writer.uint32(DDSCAPS_TEXTURE)                       # caps1
    writer.uint32(0)                                     # caps2
    writer.write("3I", *([0] * 3))                       # caps3, caps4, reserved2

    # DDS_HEADER_DXT10 (20 bytes)
    writer.uint32(dxgi_id)                               # dxgiFormat
    writer.uint32(DDS_DX10_RESOURCE_DIMENSION_TEXTURE2D) # resourceDimension
    writer.uint32(0)                                     # miscFlag
    writer.uint32(1)                                     # arraySize
    writer.uint32(0)                                     # miscFlags2

    dds += pixel_data                                    # header is 148 bytes; pixels follow
    return bytes(dds)

# ===================================================
# TEXD PAIRING (hashed, nameless files)
# ===================================================

def build_texd_size_index(texd_paths) -> dict:
    """Group candidate .TEXD paths by byte size, so a TEXT's expected size is an O(1) lookup."""
    index = defaultdict(list)
    for path in texd_paths:
        try: index[os.path.getsize(path)].append(path)
        except OSError: continue
    return index

def find_companion_texd(texture: GlacierTexture, texd_index: dict) -> str | None:
    """Find the .TEXD that belongs to `texture`: size lookup, then byte-exact proxy-tail confirm.

    The .TEXT's proxy body is the same LZ4 blocks as the tail of its .TEXD, so a real pair always
    satisfies `texd[-len(proxy):] == proxy`. That upgrades a size guess into a certainty and makes
    an accidental size collision harmless. Self-contained / Absolution textures return None.
    """
    if not texture.has_companion_texd: return None
    candidates = texd_index.get(texture.expected_texd_size, [])
    if not candidates: return None

    proxy_body = texture.data[texture.body_offset:]
    proxy_len = len(proxy_body)
    for candidate in candidates:
        try:
            with open(candidate, "rb") as f:
                f.seek(max(0, os.path.getsize(candidate) - proxy_len))
                tail = f.read()
        except OSError:
            continue
        if tail == proxy_body:
            return candidate

    # Size matched but content did not (should never happen): only trust a lone candidate.
    return candidates[0] if len(candidates) == 1 else None

def gather_texd_pool(text_paths: list[str], selected_texd_paths: set[str]) -> dict:
    """Build the .TEXD size index from explicitly-selected TEXDs plus a scan of each TEXT's folder.

    This lets the user simply select many .TEXT files (companions auto-discovered next to them),
    while still honoring any .TEXD they picked directly.
    """
    pool = set(selected_texd_paths)
    scanned_dirs = set()
    for text_path in text_paths:
        directory = os.path.dirname(text_path)
        if directory in scanned_dirs: continue
        scanned_dirs.add(directory)
        try:
            for name in os.listdir(directory):
                if name.lower().endswith(".texd"): pool.add(os.path.join(directory, name))
        except OSError:
            continue
    return build_texd_size_index(pool)

def build_texd_index_from_folder(folder: str) -> dict:
    """Build a .TEXD size index from a single folder (used by the deferred folder prompt)."""
    pool = set()
    if folder and os.path.isdir(folder):
        try:
            for name in os.listdir(folder):
                if name.lower().endswith(".texd"): pool.add(os.path.join(folder, name))
        except OSError:
            pass
    return build_texd_size_index(pool)

# ===================================================
# BLENDER IMAGE LOADING
# ===================================================

def transcode_dds_to_rgba8(dds_path: str) -> str | None:
    """Fallback for older Blender builds that cannot decode a given BCn DDS: texconv -> RGBA8.

    texconv writes the result back into the same folder under the same base name, overwriting our
    BCn DDS with the RGBA8 version, so the returned path is unchanged. Returns None on failure.
    """
    try:
        texconv = get_texconv_path()
    except FileNotFoundError as error:
        print(f"[Glacier TEXT] {error}")
        return None

    out_dir = str(Path(dds_path).parent)
    command = f'"{texconv}" -y -nologo -dx10 -f R8G8B8A8_UNORM -m 1 -o "{out_dir}" "{dds_path}"'
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"[Glacier TEXT] texconv transcode failed: {result.stderr.strip()}")
            return None
    except Exception as error:
        print(f"[Glacier TEXT] texconv error: {error}")
        return None
    return dds_path

def load_dds_image(dds_path: str, texture: GlacierTexture) -> "bpy.types.Image | None":
    """Load a DDS into Blender, transcoding to RGBA8 only if the native BCn load comes back empty.

    The datablock is named after the .dds ON DISK (e.g. '01673C4916558956.dds'), since that is the
    file Blender is actually backed by - not the internal .TEXT it came from. Colorspace is decided
    from the HEADER (tex_type / interpret_as / format), never the filename (Glacier names are hashed).
    """
    def try_load(path: str):
        try: return bpy.data.images.load(path, check_existing=False)
        except Exception as error:
            print(f"[Glacier TEXT] Blender could not load {os.path.basename(path)}: {error}")
            return None

    image = try_load(dds_path)
    if image is None or image.size[0] == 0:  # native decode failed - transcode and retry once
        if image is not None: bpy.data.images.remove(image)
        converted = transcode_dds_to_rgba8(dds_path)
        image = try_load(converted) if converted else None
        if image is None or image.size[0] == 0:
            return None

    image.name = os.path.basename(dds_path)  # keep the .dds extension in the datablock name
    image.alpha_mode = "CHANNEL_PACKED"
    if texture.is_non_color():
        image.colorspace_settings.name = "Non-Color"
    return image

def find_image_by_dds_path(dds_path: str) -> "bpy.types.Image | None":
    """Locate a loaded image datablock by the absolute path of its backing .dds file."""
    target = os.path.normcase(os.path.normpath(os.path.abspath(dds_path)))
    for image in bpy.data.images:
        if not image.filepath: continue
        try:
            candidate = os.path.normcase(os.path.normpath(bpy.path.abspath(image.filepath)))
        except Exception:
            continue
        if candidate == target:
            return image
    return None

# ===================================================
# IMPORT
# ===================================================

def write_texture_dds(texture: GlacierTexture, text_path: str, texd_bytes: bytes | None) -> tuple[str, int, int]:
    """Decode the best available mip and write it as a DX10 DDS next to the .TEXT. Returns the path."""
    width, height, pixels = texture.get_highest_mip(texd_bytes)  # TEXD mip0, else proxy top mip
    meta = texture.format_meta
    dds_path = str(Path(text_path).with_suffix(".dds"))
    with open(dds_path, "wb") as f:
        f.write(create_dds_file(pixels, width, height, meta["dxgi_id"], mip_size_for(meta, width, height)))
    return dds_path, width, height

def import_single_texture(context, text_path: str, game: str, texd_index: dict) -> tuple["bpy.types.Image | None", bool]:
    """Import one .TEXT. Returns (image, is_orphan).

    `is_orphan` is True only for a WoA/007FL texture that expects a .TEXD but whose companion was
    not found locally - it is imported now at the internal proxy resolution and flagged so the
    caller can later offer to upgrade it from a user-chosen folder.
    """
    with open(text_path, "rb") as f:
        data = f.read()
    texture = GlacierTexture(data, game, source_name=os.path.basename(text_path))

    companion = find_companion_texd(texture, texd_index)
    is_orphan = texture.has_companion_texd and companion is None

    texd_bytes = None
    if companion is not None:
        with open(companion, "rb") as f:
            texd_bytes = f.read()

    dds_path, _, _ = write_texture_dds(texture, text_path, texd_bytes)
    image = load_dds_image(dds_path, texture)
    if image is None:
        return None, is_orphan

    # Stash provenance for a future exporter and for the user's own reference.
    image["glacier_game"] = game
    image["glacier_format"] = texture.format_name
    image["glacier_full_width"] = texture.width
    image["glacier_full_height"] = texture.height
    image["glacier_mip_count"] = texture.mip_count
    image["glacier_sourced_texd"] = bool(companion)

    if context and context.area and context.area.type == 'IMAGE_EDITOR':
        context.area.spaces.active.image = image
    return image, is_orphan

def import_texture(self, context, file_paths, game: str, **options) -> list[str]:
    """Bulk import entry point. `file_paths` may be a single path or a list (TEXT and/or TEXD).

    TEXT files are imported; TEXD files (selected or found beside the TEXTs) form the companion
    pool. Each TEXT is paired to its full-res TEXD by size+content; if none is found we fall back
    to the TEXT's own highest internal (proxy) mip. Returns the list of orphan .TEXT paths (WoA/
    007FL textures whose .TEXD wasn't beside them) so the caller can offer a folder-locate upgrade.
    """
    start_time = time.time()
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    text_paths = [p for p in file_paths if p.lower().endswith(".text")]
    selected_texd = {p for p in file_paths if p.lower().endswith(".texd")}
    if not text_paths:
        self.report({'WARNING'}, "No .TEXT files selected to import.")
        return []

    texd_index = gather_texd_pool(text_paths, selected_texd)

    imported = 0
    orphans: list[str] = []
    for text_path in text_paths:
        try:
            image, is_orphan = import_single_texture(context, text_path, game, texd_index)
            if image is not None:
                imported += 1
                if is_orphan: orphans.append(text_path)
            else:
                print(f"[Glacier TEXT] No decodable image produced for {os.path.basename(text_path)}")
        except Exception as error:
            print(f"[Glacier TEXT] Failed to import {os.path.basename(text_path)}: {error}")

    elapsed = time.time() - start_time
    minutes, seconds = int(elapsed // 60), elapsed % 60
    time_str = f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"

    if imported == 0:
        self.report({'ERROR'}, f"No textures were imported ({time_str}).")
    else:
        self.report({'INFO'}, f"Imported {imported} texture(s) ({time_str}). Check the Image Editor or UV Editor")

    # Hand the orphans (if any) to the deferred folder-locate step.
    pending_texd_upgrades["orphans"] = orphans
    pending_texd_upgrades["game"] = game
    return orphans

# ---------------------------------------------------
# Deferred upgrade: locate a folder holding the missing .TEXD streams
# ---------------------------------------------------

def resolve_orphan_textures(self, context, folder: str | None) -> None:
    """Try to upgrade proxy-imported orphans to full resolution using .TEXD files in `folder`.

    Called by the folder-picker operator once the user confirms (folder set) - a cancel simply
    never calls this, leaving the already-imported proxy images in place. For each orphan whose
    companion is found in `folder`, we rewrite its .dds with the full-res mip and reload() the
    existing image datablock in place, so nothing is duplicated.
    """
    orphans = list(pending_texd_upgrades.get("orphans", []))
    game = pending_texd_upgrades.get("game")
    pending_texd_upgrades["orphans"] = []
    pending_texd_upgrades["game"] = None
    if not orphans or game is None:
        return

    texd_index = build_texd_index_from_folder(folder) if folder else {}

    upgraded = 0
    for text_path in orphans:
        try:
            with open(text_path, "rb") as f:
                data = f.read()
            texture = GlacierTexture(data, game, source_name=os.path.basename(text_path), verbose=False)
            companion = find_companion_texd(texture, texd_index)
            if companion is None:
                continue  # not in this folder either - it stays at proxy resolution

            with open(companion, "rb") as f:
                texd_bytes = f.read()
            dds_path, width, height = write_texture_dds(texture, text_path, texd_bytes)

            image = find_image_by_dds_path(dds_path)
            if image is not None:
                image.reload()  # same datablock, now backed by the full-res .dds we just rewrote
            else:  # image was closed since import - load it fresh
                image = load_dds_image(dds_path, texture)
            if image is not None:
                image["glacier_sourced_texd"] = True
                upgraded += 1
                print(f"[Glacier TEXT] Upgraded {os.path.basename(text_path)} to {width}x{height} from {os.path.basename(companion)}")
        except Exception as error:
            print(f"[Glacier TEXT] Failed to upgrade {os.path.basename(text_path)}: {error}")

    if upgraded:
        self.report({'INFO'}, f"Upgraded {upgraded} texture(s) to full resolution. Check the Image Editor or UV Editor")
    else:
        self.report({'INFO'}, "No matching .TEXD found in that folder; kept internal (proxy) resolution")

# ===================================================
# EXPORT  (Hitman: WoA / 007: First Light)
#
#   Encode a Blender image to BCn via texconv, LZ4 each
#   mip, then rebuild the .TEXT (proxy) and .TEXD (full
#   chain). Absolution export is intentionally deferred:
#   its sample data is SWIZZLED and the swizzle pattern
#   is not yet reversed, so we would emit data the game
#   cannot read correctly.
#
#   The on-disk layout below is byte-for-byte confirmed,
#   but a full round-trip through the shipping games has
#   not been done yet - treat generated files as needing
#   an in-engine test.
# ===================================================

def compute_scaling_width(width: int, height: int, mip_count: int) -> int:
    """Proxy start mip = the first level whose largest dimension is <= 128 (0 => self-contained)."""
    for level in range(mip_count):
        w = max(1, width >> level)
        h = max(1, height >> level)
        if max(w, h) <= GLACIER2_PROXY_RESOLUTION:
            return level
    return 0

def encode_image_to_bcn_dds(image, dxgi_str: str, generate_mips: bool) -> str:
    """Run texconv to encode a Blender image to a DX10 BCn DDS (full mip chain by default)."""
    texconv = get_texconv_path()
    temp_dir = Path(tempfile.gettempdir())

    source = bpy.path.abspath(image.filepath) if image.filepath else ""
    if not source or not os.path.exists(source):  # packed / generated image - stage a temp PNG
        source = str(temp_dir / f"glacier_export_src_{int(time.time())}.png")
        image.save_render(source, scene=bpy.context.scene)

    mip_arg = "" if generate_mips else "-m 1"
    color_arg = "-srgb" if "R8G8B8A8_UNORM" in dxgi_str else "-l"
    command = f'"{texconv}" -y -nologo -dx10 -f {dxgi_str} {mip_arg} {color_arg} -o "{temp_dir}" "{source}"'
    result = subprocess.run(command, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"texconv encode failed: {result.stderr.strip()}")

    encoded = temp_dir / (Path(source).stem + ".DDS")
    if not encoded.exists():
        encoded = temp_dir / (Path(source).stem + ".dds")
    if not encoded.exists():
        raise RuntimeError("texconv ran but no output DDS was produced.")
    return str(encoded)

def read_dds_mip_chain(dds_path: str, game: str, format_code: int) -> tuple[int, int, list[bytes]]:
    """Read a DX10 DDS and slice its raw mip chain using our own size math (no reliance on pitch)."""
    with open(dds_path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"DDS ":
        raise ValueError("Not a DDS file.")
    height = int.from_bytes(blob[12:16], "little")
    width = int.from_bytes(blob[16:20], "little")
    mip_count = max(1, int.from_bytes(blob[28:32], "little"))
    is_dx10 = blob[84:88] == b"DX10"
    offset = 148 if is_dx10 else 128

    mips: list[bytes] = []
    meta = get_render_format(game, format_code)
    for level in range(mip_count):
        w = max(1, width >> level)
        h = max(1, height >> level)
        size = mip_size_for(meta, w, h)
        mips.append(blob[offset:offset + size])
        offset += size
    return width, height, mips

def write_glacier2_header(writer: Writer, texture_type: int, flags: int, width: int, height: int,
                          format_code: int, mip_count: int, mip_sizes: list[int],
                          mip_sizes_compressed: list[int], scaling_width: int) -> None:
    """Emit the 152-byte Glacier 2 TEXT header. Constant fields mirror every observed sample."""
    texd_total = mip_sizes_compressed[mip_count - 1]
    writer.ushort(1)                                   # magic
    writer.ushort(texture_type)                        # tex_type
    writer.uint32(texd_total + GLACIER2_HEADER_SIZE)   # file_size (== TEXD size + 152)
    writer.uint32(flags)                               # flags (0x48 on every Glacier 2 sample)
    writer.ushort(width)
    writer.ushort(height)
    writer.ushort(format_code)
    writer.ubyte(mip_count)
    writer.ubyte(0)                                    # default_mip
    writer.ubyte(1)                                    # interpret_as (observed constant)
    writer.ubyte(0)                                    # dimensions
    writer.ushort(0)                                   # mip_interpolation

    padded_decoded = (mip_sizes + [0] * 14)[:14]
    padded_compressed = (mip_sizes_compressed + [0] * 14)[:14]
    writer.write("14I", *padded_decoded)
    writer.write("14I", *padded_compressed)

    writer.uint32(0)                                   # atlas_size
    writer.uint32(GLACIER2_HEADER_SIZE)                # atlas_offset (body begins at 152)
    writer.ubyte(0xFF)                                 # scaling_data_1 (observed constant)
    writer.ubyte(scaling_width)                        # proxy start mip
    writer.ubyte(scaling_width)                        # scaling_height (== scaling_width for a true mip proxy)
    writer.ubyte(0x08)                                 # scaling_data_2 (observed constant)
    writer.uint32(0)                                   # reserved

def export_texture(self, context, output_path: str, game: str, image, format_name: str,
                   generate_mips: bool = True, texture_type: int | None = None) -> set[str]:
    """Export a Blender image to a Glacier 2 .TEXT (+ .TEXD). WoA / 007: First Light only.

    Absolution is deferred (swizzle pattern unconfirmed). The write path reproduces the confirmed
    layout exactly but still wants an in-engine round-trip test before you trust it in a mod.
    """
    if game == GLACIER2_ABSOLUTION:
        self.report({'ERROR'}, "Absolution texture export is not supported yet (its data is swizzled and the pattern is unconfirmed).")
        return {'CANCELLED'}
    if image is None:
        self.report({'ERROR'}, "No image supplied to export.")
        return {'CANCELLED'}

    start_time = time.time()
    try:
        format_code = get_render_format_code(game, format_name)
        meta = get_render_format(game, format_code)

        # 1) texconv-encode the image to a BCn DDS with a full mip chain.
        dds_path = encode_image_to_bcn_dds(image, meta["dxgi_str"], generate_mips)
        width, height, raw_mips = read_dds_mip_chain(dds_path, game, format_code)
        mip_count = len(raw_mips)

        # 2) LZ4 each mip and build the two cumulative tables (decoded + on-disk).
        compressed_mips: list[bytes] = []
        mip_sizes: list[int] = []
        mip_sizes_compressed: list[int] = []
        running_decoded = 0
        running_compressed = 0
        for raw in raw_mips:
            comp = lz4_compress_block(raw)
            compressed_mips.append(comp)
            running_decoded += len(raw)
            running_compressed += len(comp)
            mip_sizes.append(running_decoded)
            mip_sizes_compressed.append(running_compressed)

        # 3) TEXD = full compressed chain [0..N-1]; proxy (TEXT body) = the sub-128 tail.
        scaling_width = compute_scaling_width(width, height, mip_count)
        texd_payload = b"".join(compressed_mips)
        proxy_payload = b"".join(compressed_mips[scaling_width:])

        derived_type = texture_type if texture_type is not None else (1 if meta["name"] in ("BC5", "BC4") else 0)

        # 4) Write the .TEXT (152-byte header + proxy).
        text_bytes = bytearray()
        writer = Writer(text_bytes)
        write_glacier2_header(writer, derived_type, 0x48, width, height, format_code,
                              mip_count, mip_sizes, mip_sizes_compressed, scaling_width)
        text_bytes += proxy_payload
        text_out = output_path if output_path.lower().endswith(".text") else output_path + ".TEXT"
        with open(text_out, "wb") as f:
            f.write(text_bytes)

        # 5) Write the companion .TEXD only when the texture actually streams (scaling_width > 0).
        wrote_texd = False
        if scaling_width > 0:
            texd_out = str(Path(text_out).with_suffix(".TEXD"))
            with open(texd_out, "wb") as f:
                f.write(texd_payload)
            wrote_texd = True

    except Exception as error:
        self.report({'ERROR'}, f"Texture export failed: {error}")
        return {'CANCELLED'}

    elapsed = time.time() - start_time
    minutes, seconds = int(elapsed // 60), elapsed % 60
    time_str = f"{minutes}m {seconds:.2f}s" if minutes > 0 else f"{seconds:.2f}s"
    self.report({'INFO'}, f"Exported {mip_count}-mip {format_name} .TEXT{' + .TEXD' if wrote_texd else ''} ({time_str})")
    return {'FINISHED'}
