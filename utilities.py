# ----------------------------------------
#   BLENDER PYTHON UTILITY FUNCTIONS
#       Various utility scripts for the
#       Blender Python (bpy) module to
#       allow for ease of implementing
#       other various features!
# ----------------------------------------

import os, bpy, random, math, time, struct, numpy as np, tempfile, subprocess
from pathlib import Path
from typing import cast, Optional
from mathutils import Matrix, Euler, Vector, Quaternion
from collections import defaultdict
from itertools import chain

# --------------------
# GLOBALS: GAME TYPES
# --------------------

GLACIER1_H2SA       = 'Hitman 2: Silent Assassin'
GLACIER1_HMC        = 'Hitman: Contracts'
GLACIER1_HBM        = 'Hitman: Blood Money & Mini Ninjas'
GLACIER1_FIGHTERS   = 'Freedom Fighters'
GLACIER1_KNL1       = 'Kane & Lynch: Dead Men'
GLACIER1_KNL2       = 'Kane & Lynch 2: Dog Days'

# Default extent for a Glacier 1 armature bone. Glacier 1 skeletons store only a bind transform
# per bone (no per-bone length), so the tail is placed a short fixed distance along the bone's +Y
# purely for visibility - it has no effect on skinning, which is driven by the bind matrix.
GLACIER1_ARMATURE_BONE_LENGTH = 0.1

GLACIER2_ABSOLUTION = 'Hitman: Absolution'
GLACIER2_TRILOGY    = 'Hitman: World of Assassination'
GLACIER2_BOND       = '007: First Light'

# -------------------------------------------------------
# DETERMINE BLENDER VERSION TO HANDLE THINGS DIFFERENTLY
# -------------------------------------------------------

APP_VERSION    = bpy.app.version
IS_BLENDER_4_0 = APP_VERSION >= (4, 0, 0)
IS_BLENDER_4_1 = APP_VERSION >= (4, 1, 0)

# --------------------------------------------

# ------------------------------
# DATA CONVERSIONS / INVERSIONS
# ------------------------------

def invert_uv_map(uv_set: tuple[float, float]) -> tuple[float, float]:
    """UV Map's V component inverter for import and export purposes."""
    return (uv_set[0], 1.0 - uv_set[1])

def reverse_vector(vector: list | tuple) -> tuple:
    """Reverse an n-point vector's values. Used for flipping faces' indices ordering."""
    return tuple(reversed(vector))

def convert_vertex_normal(nx: int, ny: int, nz: int) -> tuple[float, float, float]:
    """Glacier 2 (both Trilogy and Bond) normal/tangent/bitangent decode.

    NTB components are stored as unsigned bytes (0-255) centered at 128. To unpack into the
    canonical [-1, 1] range we shift and scale: `(b - 128) / 127.5`. Both branches of the
    Glacier 2 family (Hitman: WoA and 007: First Light) use this same encoding - earlier code
    treated WoA's bytes as signed which produced subtly-but-visibly wrong normals (mostly fine
    in the [-1, 1] middle range, but with hemisphere flips near the extremes).

    Used by both `_parse_vertex_buffer_trilogy` and the Bond attribute parsers in prim.py.
    """
    return ((nx - 128) / 127.5, (ny - 128) / 127.5, (nz - 128) / 127.5)

def decode_handedness_byte(value: int) -> float:
    """Decode the 4th byte of an NTB triple as a handedness sign for tangent-space reconstruction.

    Both games pack handedness as an unsigned byte centered at 128: `>= 128` = same hand,
    `< 128` = flipped. Returns +1.0 or -1.0 accordingly.

    Currently unused by the import path (Blender derives tangents from UVs + normals via
    `mesh.calc_tangents`), but kept here for the future exporter which will need to repack
    handedness back into the on-disk byte stream.
    """
    return 1.0 if value >= 128 else -1.0

def convert_vertex_color(r: int, g: int, b: int, a: int) -> tuple[float, float, float, float]:
    """Takes the RGBA of the vertex colors and divides them by 255 to convert them from unsigned bytes to floats in the [0, 1] range."""
    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

def linear_to_srgb(value: float) -> float:
    """Converter for single color channels from Linear to sRGB Color Space."""
    return value * 12.92 if value <= 0.0031308 else 1.055 * (value ** (1.0 / 2.4)) - 0.055

# ----------------------
# QUANTIZATION DECODERS
# ----------------------

# 32767 = Nax value of a signed short. Positions and UVs are stored as int16 fractions of this max and rescaled into world / texture space by per-mesh scale + bias vectors.
INT16_MAX_FLOAT = 32767.0

def dequantize_position(packed: tuple[int, int, int, int], scale: tuple[float, float, float, float], bias: tuple[float, float, float, float]) -> tuple[float, float, float]:
    """Decode a quantized int16x4 vertex position into world-space (X, Y, Z). The 4th component is the last bone index for the weights, We ignore it because that doesn't need dequantization lol | Formula: position[i] = (packed[i] / 32767.0) * scale[i] + bias[i]   for i in {0, 1, 2}."""
    return ((packed[0] / INT16_MAX_FLOAT) * scale[0] + bias[0], (packed[1] / INT16_MAX_FLOAT) * scale[1] + bias[1], (packed[2] / INT16_MAX_FLOAT) * scale[2] + bias[2])

def dequantize_uv(packed: tuple[int, int], scale: tuple[float, float], bias: tuple[float, float]) -> tuple[float, float]:
    """Decode a quantized int16x2 UV coordinate into texture-space (U, V). Formula: uv[i] = (packed[i] / 32767.0) * scale[i] + bias[i] for i in {0, 1}."""
    return ((packed[0] / INT16_MAX_FLOAT) * scale[0] + bias[0], (packed[1] / INT16_MAX_FLOAT) * scale[1] + bias[1])

# -----------------------------------
# TRIANGLE STRIP -> TRIANGLE LIST (G1)
# -----------------------------------
#
# The Glacier 1 games (H2:SA, Contracts, Blood Money, Freedom Fighters) store primitive
# geometry as triangle STRIPS, not lists. An index block is one or more strip groups, and
# consecutive strips (plus the degenerate stitches inside a single strip) share the vertex
# stream. To feed Blender we flatten every strip into an explicit triangle list.
#
# STRIP WINDING: verified against the on-disk per-vertex normals across all four games at
# 99.7-99.9% agreement. The correct winding is (i1, i0, i2) on EVEN positions and
# (i0, i1, i2) on ODD positions - i.e. the first, third, fifth... triangle has its first two
# indices swapped relative to the textbook GL strip. Emitting the textbook order instead
# gives a fully inverted mesh (0.3% agreement), which is itself the proof.
#
# DEGENERATE STITCHES: strips are joined with zero-area triangles (a repeated index) so the
# whole block can be one continuous strip. Those MUST be dropped or Blender will choke on
# from_pydata / raise on duplicate-vertex faces. We detect them by "any two of the three
# indices are equal" and skip.

def normalize_signed_byte_vector(nx: int, ny: int, nz: int) -> tuple[float, float, float]:
    """Glacier 1 packed-normal decode: signed bytes scaled by 1/128, then explicitly normalized.

    This is NOT the Glacier 2 convention. Glacier 2 stores NTB as UNSIGNED bytes centered at 128
    and the result is already unit length (see `convert_vertex_normal`). Glacier 1's weighted
    vertex format stores SIGNED bytes whose decoded vector is deliberately not unit length, so it
    must be normalized after scaling - which is exactly why a unit-length probe finds nothing at
    the correct offset. Sourced from id-daemon's readers and kept separate to avoid the two
    conventions ever being confused for one another.

    Returns (0, 0, 1) for a zero-length input rather than raising, so a degenerate vertex cannot
    abort a whole mesh import.
    """
    x, y, z = nx / 128.0, ny / 128.0, nz / 128.0
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-9: return (0.0, 0.0, 1.0)
    return (x / length, y / length, z / length)

def strip_to_triangles(strip_indices: list[int] | tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Flatten a single triangle strip into a list of (a, b, c) triangles, Glacier 1 winding.

    Drops degenerate stitch triangles (any repeated index). One strip in -> N triangles out,
    where N <= len(strip) - 2. Safe on strips shorter than 3 (returns an empty list).

    Winding matches the games' on-disk normals: even i -> (i+1, i, i+2), odd i -> (i, i+1, i+2).
    """
    triangles: list[tuple[int, int, int]] = []
    strip_length = len(strip_indices)
    if strip_length < 3: return triangles

    for i in range(strip_length - 2):
        a, b, c = strip_indices[i], strip_indices[i + 1], strip_indices[i + 2]

        # Degenerate stitch: any two indices equal -> zero-area triangle used only to join strips.
        if a == b or b == c or a == c: continue

        # Even position flips the first edge to keep consistent facing across the strip.
        if (i & 1) == 0: triangles.append((b, a, c))
        else: triangles.append((a, b, c))

    return triangles

def strips_to_triangles(strips: list[list[int]] | list[tuple[int, ...]]) -> list[tuple[int, int, int]]:
    """Flatten an index block's worth of strips (a list of strips) into one triangle list."""
    triangles: list[tuple[int, int, int]] = []
    for strip in strips: triangles.extend(strip_to_triangles(strip))
    return triangles

def indices_to_triangles(indices: list[int] | tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Flatten a flat TRIANGLE LIST index run into (a, b, c) triples.

    Blood Money does NOT use strips: its index blocks are [ushort 1][ushort indexCount][indices]
    where indexCount is always divisible by 3 and the indices are consumed three at a time
    (confirmed on 264 of 264 index blocks in M00_main.PRM, and matching id-daemon's Hitman4
    reader which iterates indexCount/3 triangles). The earlier games DO use strips, so the two
    paths must not be mixed - running a Blood Money block through the strip flattener produces
    (n - 2) garbage triangles instead of the correct (n / 3).

    Degenerate triangles (any repeated index) are dropped, same as the strip path."""
    triangles: list[tuple[int, int, int]] = []
    for i in range(0, len(indices) - 2, 3):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        if a == b or b == c or a == c: continue
        triangles.append((a, b, c))
    return triangles

# --------------------------------
# STRING / BUFFER HELPERS
# --------------------------------

def strip_null_padding(text: str) -> str:
    """Trim trailing null bytes (and any whitespace) from a fixed-length char array string."""
    return text.split('\x00', 1)[0].strip()

def align_offset(offset: int, alignment: int = 16) -> int:
    """Round an offset up to the nearest multiple of alignment. Glacier 2 data blocks are typically 16-byte aligned."""
    if alignment <= 1: return offset
    remainder = offset % alignment
    return offset if remainder == 0 else offset + (alignment - remainder)

# =================================================================================================================================================================================

# ------------------------------
# COMPRESSION CODEC: LZ4 (BLOCK)
# ------------------------------
# A dependency-free and engine-agnostic LZ4 codec. Blender ships a vanilla CPython with no lz4 module and asking users to pip-install a C extension into
# Blender's Python is a support nightmare so we carry our own. This is the LZ4 block format (The raw compression stream: no frame magic, no stored size, no checksum)
# The same primitive Glacier streams each texture mip in, and a good default for most game-internal LZ4 you will meet.

# Block-format constants. MIN_MATCH is fixed by the spec. The final bytes of a block are always literals (A decoder-safety margin) so no match may begin within the last MATCH_SAFEGUARD bytes.
LZ4_MIN_MATCH       = 4
LZ4_HASH_LOG        = 16 # 64K-entry match-finder table; More bits = Better matches, More memory
LZ4_HASH_SIZE       = 1 << LZ4_HASH_LOG
LZ4_LAST_LITERALS   = 5  # Spec: The last 5 bytes are always literals
LZ4_MATCH_SAFEGUARD = 12 # Spec: No match may start within the final 12 bytes

def lz4_decompress_block(src: bytes, decompressed_size: int | None = None) -> bytearray:
    """Decode one LZ4 raw block. Pass decompressed_size when you know it (Glacier stores it in the header mip tables): We pre-allocate the exact output and validate the result - the fast, safe path. Omit it and we grow the buffer as we go, which still terminates correctly because a block ends with a literals-only sequence once the input is consumed. Verified byte-identical to the reference lz4.block library on real texture mips and on arbitrary data. Raises ValueError on malformed / truncated input."""
    known = decompressed_size is not None
    out = bytearray(decompressed_size) if known else bytearray()
    src_view = memoryview(src)
    src_len = len(src)
    s = 0 # Read cursor in src
    d = 0 # Write cursor in out

    try:
        while s < src_len:
            token = src[s]; s += 1

            # --- Literals: A raw run copied verbatim ---
            literal_length = token >> 4
            if literal_length == 15:
                while True:
                    add = src[s]; s += 1
                    literal_length += add
                    if add != 255: break
            if literal_length:
                end = s + literal_length
                if known: out[d:d + literal_length] = src_view[s:end]
                else: out += src_view[s:end]
                s = end; d += literal_length

            # The final sequence is literals-only; Once input is spent we are done.
            if s >= src_len: break

            # --- Match: copy match_length bytes from offset back in the output ---
            offset = src[s] | (src[s + 1] << 8); s += 2
            if offset == 0: raise ValueError("LZ4: zero match offset (corrupt block)")
            match_length = token & 0x0F
            if match_length == 15:
                while True:
                    add = src[s]; s += 1
                    match_length += add
                    if add != 255: break
            match_length += LZ4_MIN_MATCH

            start = d - offset
            if start < 0: raise ValueError("LZ4: match reaches before buffer start (corrupt block)")

            if offset >= match_length:  # non-overlapping: one slice copy (fast, runs in C)
                if known: out[d:d + match_length] = out[start:start + match_length]
                else: out += out[start:start + match_length]
                d += match_length
            else:  # overlapping (offset < length): copy byte-by-byte so just-written bytes repeat
                if known:
                    for i in range(match_length): out[d] = out[start + i]; d += 1
                else:
                    for i in range(match_length): out.append(out[start + i]); d += 1
    except IndexError:
        raise ValueError("LZ4: truncated block (ran off the end of the input)")

    if known and d != decompressed_size: raise ValueError(f"LZ4: produced {d} bytes, expected {decompressed_size}")
    return out

def lz4_compress_block(src: bytes) -> bytes:
    """Encode src into a single valid LZ4 raw block. A greedy hash-4 match finder: At each position we hash the next 4 bytes, Look up the most recent place we saw that hash and if it's in-window and truly matches, Extend the match as far as it goes then emit the pending literals followed by the match. Output decodes under any conformant LZ4 decoder (verified against the reference library and our own decoder).

    Ratio note: This is a fast greedy encoder and not LZ4HC. On highly compressible data it does great (text/RLE/zeros crush down hard); On already-compressed BCn texture payloads it lands near 1.0, A touch looser than IOI's LZHC (~0.84). That is a size tradeoff only - correctness and in-engine decodability are unaffected. Add hash-chains + lazy matching to approach LZHC ratios."""
    n = len(src)
    out = bytearray()
    if n == 0: return bytes(out)

    def emit_sequence(literal_start: int, literal_end: int, offset: int | None, match_length: int) -> None:
        """Write one sequence: [token][extra literal len][literals]([offset lo/hi][extra match len])."""
        literal_length = literal_end - literal_start
        token_index = len(out)
        out.append(0)  # placeholder token, patched once the lengths are known

        if literal_length >= 15:
            remainder = literal_length - 15
            while remainder >= 255: out.append(255); remainder -= 255
            out.append(remainder)
        out.extend(src[literal_start:literal_end])

        token_literals = 15 if literal_length >= 15 else literal_length
        if offset is None:  # trailing literals-only sequence closes the block
            out[token_index] = token_literals << 4
            return

        out.append(offset & 0xFF); out.append((offset >> 8) & 0xFF)
        encoded_match = match_length - LZ4_MIN_MATCH
        if encoded_match >= 15:
            remainder = encoded_match - 15
            while remainder >= 255: out.append(255); remainder -= 255
            out.append(remainder)
        out[token_index] = (token_literals << 4) | (15 if encoded_match >= 15 else encoded_match)

    hash_table = [-1] * LZ4_HASH_SIZE
    match_limit = n - LZ4_MATCH_SAFEGUARD  # never form a match past here (keeps trailing literals)
    anchor = 0  # start of the current pending literal run
    i = 0

    while i < match_limit:
        sequence = struct.unpack_from("<I", src, i)[0]
        h = ((sequence * 2654435761) & 0xFFFFFFFF) >> (32 - LZ4_HASH_LOG)
        candidate = hash_table[h]
        hash_table[h] = i

        # Usable only if in-window (< 64 KB back) and the 4 bytes genuinely match (hashes collide).
        if candidate >= 0 and (i - candidate) < 65536 and struct.unpack_from("<I", src, candidate)[0] == sequence:
            match_length = LZ4_MIN_MATCH
            while i + match_length < match_limit and src[candidate + match_length] == src[i + match_length]:
                match_length += 1
            emit_sequence(anchor, i, i - candidate, match_length)
            i += match_length
            anchor = i
        else: i += 1

    emit_sequence(anchor, n, None, 0)  # flush the tail as literals
    return bytes(out)

# =================================================================================================================================================================================

# ----------
# MATERIALS
# ----------

def material_name_only(material_path: str) -> str:
    """Returns the name of a material, but as its name only with no path information."""
    return Path(material_path).stem

def material_path_no_ext(material_path: str) -> str:
    """Return material path with no ".material" extension. File path gets truncated if it's over 64 characters long (Blender Limitation on versions older than 4.5 and 5.0)."""
    return str(Path(material_path).with_suffix(''))

def create_material(material_name: str, assign_material_colors: bool = True) -> bpy.types.Material:
    """Credit: REDxEYE, Modified by Dodylectable | Create a material that can be placed on an object."""
    mat = bpy.data.materials.get(material_name)
    if not mat:
        mat = bpy.data.materials.new(material_name)
        if assign_material_colors: mat.diffuse_color = (random.uniform(0.4, 1.0), random.uniform(0.4, 1.0), random.uniform(0.4, 1.0), 1.0)
    return mat

def add_material(mat: bpy.types.Material, model_obj: bpy.types.Object) -> int:
    """Credit: REDxEYE, Modified by Dodylectable | Quickly add a material to a Blender object."""
    model_data = cast(bpy.types.Mesh, model_obj.data)

    # Check if material is already assigned to avoid duplicate slots
    for idx, slot_mat in enumerate(model_data.materials):
        if slot_mat == mat: return idx

    model_data.materials.append(mat)
    return len(model_data.materials) - 1

# =================================================================================================================================================================================

# --------------
# VERTEX GROUPS
# --------------

def rename_vertex_groups_to_bone_names(obj: bpy.types.Object, bone_map: dict[int, str]) -> None:
    """Rename vertex groups from 'bone_<id>' to actual bone names."""
    for group in obj.vertex_groups:
        if group.name.startswith("bone_"):
            try:
                bone_index = int(group.name[5:]) # Fast slice off "bone_"
                bone_name = bone_map.get(bone_index)
                if bone_name: group.name = bone_name
            except ValueError: pass # Invalid integer conversion, skip cleanly

def rename_vertex_groups_to_bone_indices(obj: bpy.types.Object, bone_map: dict[str, int]) -> None:
    """Rename vertex groups from actual bone names to 'bone_<id>'."""
    for group in obj.vertex_groups:
        bone_index = bone_map.get(group.name)
        if bone_index is not None: group.name = f"bone_{bone_index}"

def handle_vertex_group_rename_to_names() -> None:
    """Switch selected mesh vertex groups to bone names."""
    cached_bone_maps = {} # Cache to prevent rebuilding maps for shared skeletons

    for obj in bpy.context.selected_objects:
        if (obj.type != 'MESH'): continue

        arm = get_attached_skeleton(obj)
        if not arm:
            print(f"Warning: Model '{obj.name}' has no skeleton attached. Skipping.")
            continue

        # Build and cache map once per armature
        if arm.name not in cached_bone_maps: cached_bone_maps[arm.name] = {bone['id']: bone.name for bone in arm.data.bones if 'id' in bone}

        rename_vertex_groups_to_bone_names(obj, cached_bone_maps[arm.name])

def handle_vertex_group_rename_to_indices() -> None:
    """Switch selected mesh vertex groups to bone indices."""
    cached_bone_maps = {}

    for obj in bpy.context.selected_objects:
        if (obj.type != 'MESH'): continue

        arm = get_attached_skeleton(obj)
        if not arm:
            print(f"Warning: Model '{obj.name}' has no skeleton attached. Skipping.")
            continue

        if arm.name not in cached_bone_maps: cached_bone_maps[arm.name] = {bone.name: bone['id'] for bone in arm.data.bones if 'id' in bone}

        rename_vertex_groups_to_bone_indices(obj, cached_bone_maps[arm.name])

def get_attached_skeleton(obj: bpy.types.Object) -> bpy.types.Object | None:
    """Return the object instance of an attached skeleton to an object if it exists. If no object was found or the type of the object is not a skeleton returns `None`."""
    # The provided object is a skeleton already.
    if (obj.type == 'ARMATURE'): return obj

    # Check through the modifiers, See if there was a skeleton set.
    for (mod) in (obj.modifiers):
        if (mod.type == 'ARMATURE') and (mod.object): return mod.object

    # Is this parented to an armature?
    if (obj.parent) and (obj.parent.type == 'ARMATURE'): return obj.parent

    # No armature found, bruh - Just return None in that case, we'll handle this ourselves
    return None

# =================================================================================================================================================================================

# ---------
# TEXTURES
# ---------
#
# Glacier stores a texture's pixel format as a single u16 render-format code in the header. The
# SAME numeric code means DIFFERENT formats across engine branches: Hitman: WoA and Hitman:
# Absolution share the classic Glacier 2 ERenderFormat enum, while 007: First Light renumbered it.
# So every lookup here is game-aware. Each descriptor carries what both sides of the pipeline need:
# the importer (rebuild a DDS -> hand to Blender) and the exporter (texconv-encode -> repack).
#   dxgi_id / dxgi_str : DXGI format for the DDS DX10 header and texconv's `-f`
#   block_size         : bytes per 4x4 block for BCn formats (0 for uncompressed)
#   bpp                : bytes per pixel for uncompressed formats (0 for BCn)
#   is_compressed      : True for the block-compressed BCn family
#
# BC5 maps to BC5_UNORM (83), never SNORM (84): these games do not use the SNORM variant, so there
# is no transcode-on-import step - Blender loads the BCn DDS directly.

def texture_format(name: str, dxgi_id: int, dxgi_str: str, block_size: int, bpp: int, is_compressed: bool) -> dict:
    """Build one render-format descriptor row."""
    return {"name": name, "dxgi_id": dxgi_id, "dxgi_str": dxgi_str, "block_size": block_size, "bpp": bpp, "is_compressed": is_compressed}

# Codes below come from the shipped ERenderFormat enums for each branch, filtered down to what
# actually turns up in mod work: the BCn family (plus their sRGB twins and BC6H for HDR), and the
# common uncompressed layouts. Deliberately omitted are the TYPELESS entries, the SNORM variants,
# and the depth/stencil D*/X* codes - none of them appear in shippable texture assets, and listing
# them would only invite a wrong `-f` argument on export.
#
# Two corrections landed here when the full enums were cross-checked against the old hand-built
# tables, both worth knowing if you have older output lying around:
#   * WoA 0x42 was labelled A8. It is R8_UNORM; A8 is 0x46. Same 1 byte/pixel either way, so mip
#     sizes were never wrong, but the DDS carried the wrong DXGI id (61 vs 65).
#   * 007FL 0x42 was labelled BC3. It is B5G6R5_UNORM; BC3 is 0x52. This one mattered - a 2 bpp
#     uncompressed format was being sized as a 16-byte-per-block compressed one. B5G6R5 is kept in
#     both tables (rather than filtered out as legacy) precisely because a file was hitting 0x42.

# Classic Glacier 2 ERenderFormat - shared by Hitman: WoA (Trilogy) and Hitman: Absolution.
GLACIER2_RENDER_FORMATS = {
    0x0A: texture_format("R16G16B16A16F",  10, "R16G16B16A16_FLOAT",    0, 8, False),
    0x18: texture_format("R10G10B10A2",    24, "R10G10B10A2_UNORM",     0, 4, False),
    0x1A: texture_format("R11G11B10F",     26, "R11G11B10_FLOAT",       0, 4, False),
    0x1C: texture_format("R8G8B8A8",       28, "R8G8B8A8_UNORM",        0, 4, False),
    0x1D: texture_format("R8G8B8A8_SRGB",  29, "R8G8B8A8_UNORM_SRGB",   0, 4, False),
    0x22: texture_format("R16G16F",        34, "R16G16_FLOAT",          0, 4, False),
    0x23: texture_format("R16G16",         35, "R16G16_UNORM",          0, 4, False),
    0x34: texture_format("R8G8",           49, "R8G8_UNORM",            0, 2, False),
    0x39: texture_format("R16F",           54, "R16_FLOAT",             0, 2, False),
    0x3B: texture_format("R16",            56, "R16_UNORM",             0, 2, False),
    0x3F: texture_format("B5G6R5",         85, "B5G6R5_UNORM",          0, 2, False),
    0x42: texture_format("R8",             61, "R8_UNORM",              0, 1, False),
    0x46: texture_format("A8",             65, "A8_UNORM",              0, 1, False),
    0x49: texture_format("BC1",            71, "BC1_UNORM",             8, 0, True),
    0x4A: texture_format("BC1_SRGB",       72, "BC1_UNORM_SRGB",        8, 0, True),
    0x4C: texture_format("BC2",            74, "BC2_UNORM",            16, 0, True),
    0x4D: texture_format("BC2_SRGB",       75, "BC2_UNORM_SRGB",       16, 0, True),
    0x4F: texture_format("BC3",            77, "BC3_UNORM",            16, 0, True),
    0x50: texture_format("BC3_SRGB",       78, "BC3_UNORM_SRGB",       16, 0, True),
    0x52: texture_format("BC4",            80, "BC4_UNORM",             8, 0, True),
    0x55: texture_format("BC5",            83, "BC5_UNORM",            16, 0, True),
    0x57: texture_format("BC6H",           95, "BC6H_UF16",            16, 0, True),
    0x5A: texture_format("BC7",            98, "BC7_UNORM",            16, 0, True),
    0x5B: texture_format("BC7_SRGB",       99, "BC7_UNORM_SRGB",       16, 0, True),
    0x62: texture_format("B8G8R8A8",       87, "B8G8R8A8_UNORM",        0, 4, False),
    0x63: texture_format("B8G8R8A8_SRGB",  91, "B8G8R8A8_UNORM_SRGB",   0, 4, False),
}

# 007: First Light renumbered the enum wholesale, so codes do NOT carry over from the table above -
# 0x42 is BC3 in one branch and B5G6R5 in the other. Always resolve through get_render_format().
# BC1 (0x4C) and BC5 (0x58) are additionally confirmed against real 007FL samples.
GLACIER2_BOND_RENDER_FORMATS = {
    0x0A: texture_format("R16G16B16A16F",  10, "R16G16B16A16_FLOAT",    0, 8, False),
    0x18: texture_format("R10G10B10A2",    24, "R10G10B10A2_UNORM",     0, 4, False),
    0x1A: texture_format("R11G11B10F",     26, "R11G11B10_FLOAT",       0, 4, False),
    0x1C: texture_format("R8G8B8A8",       28, "R8G8B8A8_UNORM",        0, 4, False),
    0x1D: texture_format("R8G8B8A8_SRGB",  29, "R8G8B8A8_UNORM_SRGB",   0, 4, False),
    0x22: texture_format("B8G8R8A8",       87, "B8G8R8A8_UNORM",        0, 4, False),
    0x23: texture_format("B8G8R8A8_SRGB",  91, "B8G8R8A8_UNORM_SRGB",   0, 4, False),
    0x25: texture_format("R16G16F",        34, "R16G16_FLOAT",          0, 4, False),
    0x26: texture_format("R16G16",         35, "R16G16_UNORM",          0, 4, False),
    0x37: texture_format("R8G8",           49, "R8G8_UNORM",            0, 2, False),
    0x3C: texture_format("R16F",           54, "R16_FLOAT",             0, 2, False),
    0x3E: texture_format("R16",            56, "R16_UNORM",             0, 2, False),
    0x42: texture_format("B5G6R5",         85, "B5G6R5_UNORM",          0, 2, False),
    0x45: texture_format("R8",             61, "R8_UNORM",              0, 1, False),
    0x49: texture_format("A8",             65, "A8_UNORM",              0, 1, False),
    0x4C: texture_format("BC1",            71, "BC1_UNORM",             8, 0, True),
    0x4D: texture_format("BC1_SRGB",       72, "BC1_UNORM_SRGB",        8, 0, True),
    0x4F: texture_format("BC2",            74, "BC2_UNORM",            16, 0, True),
    0x50: texture_format("BC2_SRGB",       75, "BC2_UNORM_SRGB",       16, 0, True),
    0x52: texture_format("BC3",            77, "BC3_UNORM",            16, 0, True),
    0x53: texture_format("BC3_SRGB",       78, "BC3_UNORM_SRGB",       16, 0, True),
    0x55: texture_format("BC4",            80, "BC4_UNORM",             8, 0, True),
    0x58: texture_format("BC5",            83, "BC5_UNORM",            16, 0, True),
    0x5B: texture_format("BC6H",           95, "BC6H_UF16",            16, 0, True),
    0x5E: texture_format("BC7",            98, "BC7_UNORM",            16, 0, True),
    0x5F: texture_format("BC7_SRGB",       99, "BC7_UNORM_SRGB",       16, 0, True),
}

def get_render_format_table(game: str) -> dict:
    """Return the format-code -> descriptor table for a given game branch."""
    return GLACIER2_BOND_RENDER_FORMATS if game == GLACIER2_BOND else GLACIER2_RENDER_FORMATS

def get_render_format(game: str, format_code: int) -> dict:
    """Resolve a header render-format code to its descriptor for `game`. Raises on unknown codes."""
    table = get_render_format_table(game)
    if format_code not in table: raise ValueError(f"Unknown {game} render format code: 0x{format_code:02X}")
    return table[format_code]

def get_render_format_code(game: str, format_name: str) -> int:
    """Reverse lookup: canonical name (e.g. 'BC5') -> the game's header code. Used on export."""
    table = get_render_format_table(game)
    for code, meta in table.items():
        if meta["name"] == format_name: return code
    raise ValueError(f"{game} has no render format named '{format_name}'")

def mip_size_for(format_meta: dict, width: int, height: int) -> int:
    """Byte size of a single mip level for a format descriptor.

    The block-ceil `(dim + 3) // 4` is what makes this correct for non-power-of-two textures
    (UI atlases, billboards) as well as clean powers of two.
    """
    if format_meta["is_compressed"]:
        return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * format_meta["block_size"]
    return width * height * format_meta["bpp"]

def calculate_mip_size(game: str, format_code: int, width: int, height: int) -> int:
    """Convenience: resolve the format for `game`, then compute its mip byte size."""
    return mip_size_for(get_render_format(game, format_code), width, height)

def get_texconv_path() -> Path:
    """Resolve and validate the path to texconv (used only on export, to encode custom BCn DDS)."""
    texconv_path = Path(__file__).parent / "binaries" / "texconv.exe"
    if not texconv_path.exists(): raise FileNotFoundError(f"Missing required binary: {texconv_path}. Ensure texconv.exe is in the plugin's binaries directory.")
    return texconv_path

# =================================================================================================================================================================================
