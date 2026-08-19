# =====================================================
#   GLACIER 1 PRM (RENDERPRIMITIVE) PARSER
#       Parses IO Interactive's Glacier 1 level
#       primitive containers used by:
#           - Hitman 2: Silent Assassin
#           - Hitman: Contracts
#           - Hitman: Blood Money (and Mini Ninjas, same tech)
#           - Freedom Fighters
#
#       Mirrors PRM.bt 1:1. If you change the binary
#       template, mirror the change here.
#
#       Unlike the Glacier 2 PRIM parser there is NO
#       auto-detection: the caller passes the game
#       constant and we branch on it. The three classic
#       titles share an anchor/reference-table container;
#       Blood Money uses a heap/block-table container.
# =====================================================

import struct
from ..io import Reader
from ..utilities import *

# ==========
# CONSTANTS
# ==========

# Per-game field layout for the CLASSIC anchor container. Every offset is relative to
# the anchor and was recovered byte-exact against real samples (see PRM.bt header).
#
#   bounds_prefix   : bytes of bounding-volume prefix sitting BEFORE the anchor
#   vertex_count    : ushort, offset from anchor
#   bounds_pointer  : uint back-pointer, must equal (anchor - bounds_prefix)
#   vertex_pointer  : uint, offset from anchor
#   index_pointer   : uint, offset from anchor
#   index_count     : uint (total ushorts in the index block), offset from anchor
GLACIER1_CLASSIC_LAYOUTS = {
    GLACIER1_H2SA: {
        "bounds_prefix":  0x3C,
        "vertex_count":   0x16,
        "bounds_pointer": 0x18,
        "vertex_pointer": 0x1C,
        "index_pointer":  0x44,
        "index_count":    0x48,
    },
    GLACIER1_HMC: {
        "bounds_prefix":  0x38,
        "vertex_count":   0x0E,
        "bounds_pointer": 0x10,
        "vertex_pointer": 0x14,
        "index_pointer":  0x34,
        "index_count":    0x38,
    },
    GLACIER1_FIGHTERS: {
        "bounds_prefix":  0x3C,
        "vertex_count":   0x0E,
        "bounds_pointer": 0x10,
        "vertex_pointer": 0x14,
        "index_pointer":  0x3C,
        "index_count":    0x40,
    },
}

# Classic vertex is a fixed 36-byte interleave: pos(3f) normal(3f) color(4ub, BGRA) uv(2f).
GLACIER1_CLASSIC_VERTEX_STRIDE = 36

# Blood Money vertex strides. This is the COMPLETE set - BMEdit's PRMVertexBufferFormat enum
# is exhaustive at 0x10/0x24/0x28/0x34, and a census of a real level confirmed no others turn
# up (40B x3107, 52B x112, 36B x95, 16B x61, plus one 80B chunk that is a doubled 40B buffer).
#   0x10 (16B) : position + colour only
#   0x24 (36B) : position + normal(3f) + UV  (same float normal the classic games carry)
#   0x28 (40B) : position + colour + UV      (static level geometry - the majority; NO normal)
#   0x34 (52B) : skinned/weighted vertex     (characters, animated props)
GLACIER1_HBM_VERTEX_STRIDE_SIMPLE   = 16
GLACIER1_HBM_VERTEX_STRIDE_NORMAL   = 36
GLACIER1_HBM_VERTEX_STRIDE_STATIC   = 40
GLACIER1_HBM_VERTEX_STRIDE_WEIGHTED = 52
GLACIER1_HBM_VERTEX_STRIDES = (16, 36, 40, 52)

# A vertex block whose real stride is a whole multiple of 0x28 (40) but not 40 itself packs
# several logical vertices per declared vertex (BMExport scales vertex_num by size/0x28 in that
# case). We reproduce that so an 80B block reads as 2x the declared count at 40B stride.
GLACIER1_HBM_STRIDE_MULTIPLE_BASE = 0x28

# A Blood Money "buffer" chunk needs at least these bytes to hold its four header fields.
GLACIER1_HBM_BUFFER_CHUNK_SIZE = 16

# Upper sanity bound on a buffer chunk's declared vertex count. Real meshes top out in the low
# thousands; anything past this is a chunk that merely looks like a buffer header.
GLACIER1_HBM_MAX_VERTICES = 200000

# Blood Money MODEL chunk graph field offsets (see PRM.resolve_model_parts).
GLACIER1_HBM_MODEL_CHUNK_SIZE      = 0x1C
GLACIER1_HBM_MODEL_SKELETON        = 0x10
GLACIER1_HBM_MODEL_PART_COUNT      = 0x14
GLACIER1_HBM_MODEL_GEOMETRY        = 0x18
GLACIER1_HBM_PART_CHUNK_SIZE       = 0x2C
GLACIER1_HBM_PART_LOD_MASK         = 0x0E
GLACIER1_HBM_PART_GEOMETRY_BUFFER  = 0x28
GLACIER1_HBM_MAX_PARTS             = 4096

# Blood Money MODEL chunk tag. The first u32 of a 0x40-byte chunk equals this when the chunk is
# a model (BMExport's is_model test). Used to enumerate models for skeleton resolution.
GLACIER1_HBM_MODEL_TAG = 0x70100
GLACIER1_HBM_MODEL_CHUNK_TOTAL_SIZE = 0x40

# Blood Money SKELETON chain (see PRM.parse_blood_money_skeleton). The model's +0x10 points at a
# skeleton pointer chunk; its +4 word points at the skeleton HEADER chunk, whose first u32 is the
# bone count followed by an array of sub-chunk indices. Each sub-chunk is one parallel per-bone
# array, discriminated by its (size / boneCount) stride - which is how they were identified
# byte-exact on a 100-bone character:
#   64 B/bone : bone definition; null-terminated name at def+BONE_NAME_OFFSET (100/100 clean)
#    4 B/bone : parent table; each entry is a BYTE offset -> divide by BONE_PARENT_STRIDE
#   28 B/bone : local transform; quaternion(4f) + position(3f) (100/100 unit quaternions)
GLACIER1_HBM_SKELETON_POINTER_FIELD = 0x04   # word inside the +0x10 pointer chunk -> header chunk
GLACIER1_HBM_BONE_DEF_STRIDE        = 64
GLACIER1_HBM_BONE_NAME_OFFSET       = 28
GLACIER1_HBM_BONE_NAME_MAX          = 32
GLACIER1_HBM_BONE_PARENT_STRIDE     = 4      # parent array: 4 bytes per bone
GLACIER1_HBM_BONE_PARENT_DIVISOR    = 48     # parent entry is a byte offset; /48 -> bone index
GLACIER1_HBM_BONE_LOCAL_STRIDE      = 28     # quat(4f) + pos(3f)
GLACIER1_HBM_MAX_BONES              = 4096

# Glacier stores bone indices pre-multiplied by 3 (the stride of its internal bone matrix rows).
# The same convention appears in Glacier 2's Absolution boneRemapValues.
GLACIER1_BONE_INDEX_DIVISOR = 3

# A part chunk's weightType field selects the vertex layout: 0 means WEIGHTED (52 byte skinned
# vertex), non-zero means static. id-daemon's reader spells this `weitype`; it is a weight TYPE
# and is named in full here so the intent survives.
GLACIER1_WEIGHT_TYPE_WEIGHTED = 0

# The set of games this parser accepts. Mini Ninjas rides on GLACIER1_HBM (same tech); K&L are
# deferred elsewhere.
GLACIER1_SUPPORTED = (GLACIER1_H2SA, GLACIER1_HMC, GLACIER1_HBM, GLACIER1_FIGHTERS)

# =====================================================
# MAIN PARSER CLASS
# =====================================================

class PRM():
    """Glacier 1 RenderPrimitive parser. Produces a fully-decoded object model of every submesh in a `.PRM` file."""
    def __init__(self, file_path: str, game: str):
        """Construct the parser and run the full parse pass."""

        super().__init__()

        # ===============================
        # == CLASS MEMBERS ==============
        # ===============================

        # -- INPUT METADATA
        self.model_file: str = file_path
        """The path to the source `.PRM` file."""

        self.game: str = game
        """Which Glacier 1 title produced this file. Drives container + layout branching."""

        # -- CONTAINER SHAPE
        self.is_blood_money: bool = (game == GLACIER1_HBM)
        """Blood Money uses the heap/block-table container; the other three use anchors."""

        # -- FILE-LEVEL HEADER (meaning differs per container; see parse methods)
        self.header: dict = {}
        """The decoded 16-byte header."""

        # -- MASTER SUBMESH LIST
        self.objects: list[dict] = []
        """Per-submesh parsed data. One entry per validated mesh. See `build_mesh_record`."""

        # -- BLOOD MONEY CHUNK STATE
        self.blocks: list[tuple] = []
        """Blood Money only: the decoded block table, one (offset, size, useCount, zero) per entry."""

        self.block_count: int = 0
        """Blood Money only: number of heap blocks."""

        self.buffer: bytes = b""
        """The raw file bytes, retained so the Blood Money model graph can random-access chunks."""

        self.mesh_index_by_reference: dict[int, int] = {}
        """Model reference -> index into `self.objects`. Keyed by PRM byte offset on the classic
        games and by buffer-chunk index on Blood Money, which is exactly what the companion GMS
        stores, so the model handler can map a prop's transform straight onto its mesh."""

        # -- BLOOD MONEY SKELETONS
        self.skeletons: dict[int, dict] = {}
        """Blood Money only: model chunk index -> decoded skeleton (bones list + name/parent maps).
        Empty on the classic games, whose skeleton layout is not yet recovered. See
        `parse_blood_money_skeleton` for the record shape."""

        self.skeleton_by_buffer: dict[int, int] = {}
        """Blood Money only: buffer chunk index -> the model chunk index that owns it. Lets the
        handler find which skeleton a weighted submesh binds to."""

        # -- BLENDER OBJECT LIST (populated by the handler, kept here for parity with PRIM)
        self.blender_objects: list[bpy.types.Object] = []
        """Master list of all built Blender objects."""

        # -- VALIDATION
        if game not in GLACIER1_SUPPORTED: raise ValueError(f"Unsupported game type for PRM parsing: '{game}'. Use GLACIER1_H2SA, GLACIER1_HMC, GLACIER1_HBM or GLACIER1_FIGHTERS.")

        # ===============================
        # == PARSE THE DATA =============
        # ===============================
        self.parse_model_file()

    # =====================================================
    # TOP-LEVEL DRIVER
    # =====================================================

    def parse_model_file(self) -> None:
        """Parse the model file. Branches on container: Blood Money heap vs classic anchors."""
        print(f"\nParsing PRM model ({self.game}): {self.model_file}\n")
        self.buffer = open(self.model_file, "rb").read()
        reader = Reader(self.buffer)

        if self.is_blood_money: self.parse_blood_money(reader)
        else: self.parse_classic(reader)

        print(f"\nPRM PARSING COMPLETE!  {len(self.objects)} submeshes parsed.\n")

    # =====================================================
    # CLASSIC CONTAINER (H2:SA / Contracts / Freedom Fighters)
    # =====================================================

    def parse_classic(self, reader: Reader) -> None:
        """Parse the anchor/reference-table container.

        Header (16 B): headerField00 (u32, unknown), referenceTableOffset (u32),
        referenceTableOffsetCopy (u32), referenceCount (u32). The reference table is a
        sorted array of u32 anchors at the tail of the file. Each anchor that passes the
        back-pointer test is a mesh; the bounds prefix sits immediately before it."""
        layout = GLACIER1_CLASSIC_LAYOUTS[self.game]

        # ---------------- HEADER ----------------
        header_field_00        = reader.uint32()
        reference_table_offset = reader.uint32()
        reference_table_copy   = reader.uint32()
        reference_count        = reader.uint32()

        self.header = {
            "header_field_00":        header_field_00,
            "reference_table_offset": reference_table_offset,
            "reference_count":        reference_count,
        }
        data_limit = reference_table_offset  # object data ends where the table begins
        print(f"Reference table: {reference_count} anchors at 0x{reference_table_offset:08X}")

        # ---------------- REFERENCE TABLE ----------------
        reader.seek(reference_table_offset)
        anchors = [reader.uint32() for _ in range(reference_count)]

        bounds_prefix   = layout["bounds_prefix"]
        vertex_count_o  = layout["vertex_count"]
        bounds_ptr_o    = layout["bounds_pointer"]
        vertex_ptr_o    = layout["vertex_pointer"]
        index_ptr_o     = layout["index_pointer"]
        index_count_o   = layout["index_count"]

        mesh_index = 0
        for anchor in anchors:
            # Freedom Fighters writes one trailing terminator entry with bit 31 set.
            if anchor & 0x80000000: continue

            # Bounds must exist and the whole header window must fit before the table.
            if anchor < bounds_prefix: continue
            if anchor + index_count_o + 4 > data_limit: continue

            # Back-pointer test: the field at (anchor + bounds_pointer) must point at the
            # bounds prefix start. This is what separates meshes from the ~30-45% of anchors
            # that are lights/portals/markers of other, un-recovered types.
            reader.seek(anchor + bounds_ptr_o)
            back_pointer = reader.uint32()
            if back_pointer != anchor - bounds_prefix: continue

            reader.seek(anchor + vertex_count_o)
            vertex_count = reader.ushort()
            reader.seek(anchor + vertex_ptr_o)
            vertex_pointer = reader.uint32()
            reader.seek(anchor + index_ptr_o)
            index_pointer = reader.uint32()
            index_count = reader.uint32()

            # Sanity gates mirroring the template's read guards.
            if vertex_count == 0 or vertex_pointer == 0 or index_pointer == 0: continue
            if vertex_pointer + GLACIER1_CLASSIC_VERTEX_STRIDE * vertex_count > data_limit: continue
            if index_count <= 2 or index_pointer + 2 * index_count > data_limit: continue

            print(f"\n--- Submesh {mesh_index} @ anchor 0x{anchor:08X} ({vertex_count} verts, {index_count} index words) ---")

            positions, normals, colors, uvs = self.parse_classic_vertices(reader, vertex_pointer, vertex_count)
            strips = self.parse_index_block(reader, index_pointer)
            triangles = strips_to_triangles(strips)

            record = self.build_mesh_record(
                anchor        = anchor,
                block_index   = None,
                vertex_stride = GLACIER1_CLASSIC_VERTEX_STRIDE,
                vertex_count  = vertex_count,
                index_count   = index_count,
                strip_count   = len(strips),
                positions     = positions,
                normals       = normals,
                colors        = colors,
                uv_channels   = [uvs],
                triangles     = triangles,
            )
            self.mesh_index_by_reference[anchor] = len(self.objects)
            self.objects.append(record)
            mesh_index += 1

    def parse_classic_vertices(self, reader: Reader, vertex_pointer: int, vertex_count: int) -> tuple[list, list, list, list]:
        """Read `vertex_count` 36-byte classic vertices: pos(3f) normal(3f) color(4ub BGRA) uv(2f)."""
        positions: list[tuple[float, float, float]] = []
        normals:   list[tuple[float, float, float]] = []
        colors:    list[tuple[float, float, float, float]] = []
        uvs:       list[tuple[float, float]] = []

        reader.seek(vertex_pointer)
        for _ in range(vertex_count):
            positions.append(reader.vec3f())
            normals.append(reader.vec3f())
            b, g, r, a = reader.vec4ub()  # stored BGRA
            colors.append(convert_vertex_color(r, g, b, a))
            uvs.append(invert_uv_map(reader.vec2f()))

        return positions, normals, colors, uvs

    # =====================================================
    # BLOOD MONEY CONTAINER (heap + block table)
    # =====================================================

    def parse_blood_money(self, reader: Reader) -> None:
        """Parse the heap/block-table container.

        Header (16 B): blockTableOffset (u32), blockCount (u32), blockTableOffsetCopy (u32),
        reservedZero (u32). The block table is a gapless partition of the heap: each entry is
        [blockOffset, blockSize, useCount, reservedZero].

        Meshes are found through the CHUNK GRAPH rather than by guessing at adjacent blocks. A
        "buffer" chunk describes one mesh:

            +0x00  uint vertexCount        (explicit - no need to infer it from the indices)
            +0x04  uint vertexBlockIndex
            +0x08  uint unknown
            +0x0C  uint faceBlockIndex

        id-daemon's reader has three extra uints before faceBlockIndex, but those are commented
        "for kane" (Kane & Lynch) and do not apply to Blood Money; probing every candidate offset
        against real data puts faceBlockIndex at +0x0C, which yields ~770 buffer chunks of which
        nearly all hold plausible vertex positions. The old adjacency heuristic found only 195."""
        # ---------------- HEADER ----------------
        block_table_offset = reader.uint32()
        block_count        = reader.uint32()
        block_table_copy   = reader.uint32()
        reserved_zero      = reader.uint32()

        self.header = {
            "block_table_offset": block_table_offset,
            "block_count":        block_count,
        }
        heap_limit = block_table_offset  # heap ends where the block table begins
        print(f"Block table: {block_count} heap blocks at 0x{block_table_offset:08X}")

        # ---------------- BLOCK TABLE ----------------
        reader.seek(block_table_offset)
        blocks = [reader.read("4I") for _ in range(block_count)]  # (offset, size, useCount, zero)
        self.blocks = blocks
        self.block_count = block_count

        # ---------------- SKELETONS (walk every model chunk's skeleton chain) ----------------
        # Done first so a weighted submesh can be tied back to the skeleton that owns it as we
        # build it. Failures are swallowed per-model: a level with one bad skeleton must still
        # import every other mesh.
        self.parse_all_blood_money_skeletons()

        # ---------------- LOCATE MESHES VIA THE CHUNK GRAPH ----------------
        mesh_index = 0
        for i in range(block_count):
            block_offset, block_size = blocks[i][0], blocks[i][1]
            if block_size < GLACIER1_HBM_BUFFER_CHUNK_SIZE: continue

            reader.seek(block_offset)
            vertex_count      = reader.uint32()
            vertex_block      = reader.uint32()
            unknown_field     = reader.uint32()
            face_block        = reader.uint32()

            if vertex_count == 0 or vertex_count > GLACIER1_HBM_MAX_VERTICES: continue
            if not (0 <= vertex_block < block_count and 0 <= face_block < block_count): continue

            vertex_block_size = blocks[vertex_block][1]
            if vertex_block_size == 0: continue

            # Resolve the true stride. When the block is a whole multiple of 0x28 wider than the
            # declared vertex count, the block packs multiple logical vertices per declared vertex
            # (BMExport's `vertex_num *= size/0x28` case) - unfold it so we read them all at 40B.
            effective_count, vertex_stride = self.resolve_vertex_stride(vertex_block_size, vertex_count)
            if vertex_stride not in GLACIER1_HBM_VERTEX_STRIDES: continue

            indices = self.parse_index_list(reader, blocks[face_block][0], blocks[face_block][1])
            if indices is None: continue
            if indices and max(indices) >= effective_count: continue

            vertex_offset = blocks[vertex_block][0]
            if vertex_offset + vertex_stride * effective_count > heap_limit: continue

            triangles = indices_to_triangles(indices)
            positions, normals, colors, uvs, bone_weights, bone_indices = self.parse_blood_money_vertices(reader, vertex_offset, effective_count, vertex_stride)

            print(f"\n--- Submesh {mesh_index} @ chunk #{i} ({vertex_stride}B stride, {effective_count} verts, {len(triangles)} tris) ---")

            record = self.build_mesh_record(
                anchor        = block_offset,
                block_index   = i,
                vertex_stride = vertex_stride,
                vertex_count  = effective_count,
                index_count   = len(indices),
                strip_count   = 1,   # Blood Money index blocks always declare a single group
                positions     = positions,
                normals       = normals,
                colors        = colors,
                uv_channels   = [uvs] if uvs else [],
                triangles     = triangles,
                bone_weights  = bone_weights,
                bone_indices  = bone_indices,
                skeleton_model = self.skeleton_by_buffer.get(i),
            )
            self.mesh_index_by_reference[i] = len(self.objects)
            self.objects.append(record)
            mesh_index += 1

    def resolve_vertex_stride(self, vertex_block_size: int, vertex_count: int) -> tuple[int, int]:
        """Return (effectiveVertexCount, stride) for a Blood Money vertex block.

        The naive stride is blockSize / declaredCount, floored to a multiple of 4 (Glacier pads
        every vertex to a 4-byte boundary, so the fractional remainder is padding). When that
        stride is a clean multiple of 0x28 but not 0x28 itself, the block actually holds several
        40-byte vertices per declared vertex, so we rescale the count and pin the stride to 40 -
        mirroring BMExport's `vertex_num *= size/0x28; size = 0x28` special case. Returns
        (0, 0) for an indivisible block so the caller can reject it."""
        if vertex_count <= 0 or vertex_block_size % vertex_count: return (0, 0)

        stride = vertex_block_size // vertex_count
        stride -= stride % 4  # drop per-vertex tail padding

        if stride != GLACIER1_HBM_STRIDE_MULTIPLE_BASE and stride % GLACIER1_HBM_STRIDE_MULTIPLE_BASE == 0:
            multiplier = stride // GLACIER1_HBM_STRIDE_MULTIPLE_BASE
            return (vertex_count * multiplier, GLACIER1_HBM_STRIDE_MULTIPLE_BASE)

        return (vertex_count, stride)

    def parse_index_list(self, reader: Reader, block_offset: int, block_size: int) -> Optional[list[int]]:
        """Read a Blood Money index block as a TRIANGLE LIST.

        Layout: [ushort groupCount][ushort indexCount][indexCount ushorts]. groupCount is 1 on
        every block in the sample and indexCount is always divisible by 3 (264/264 verified),
        which is what proves these are lists rather than the strips the earlier games use.
        Returns None when the block is not a valid index block."""
        if block_size < 4: return None

        reader.seek(block_offset)
        group_count = reader.ushort()
        index_count = reader.ushort()

        if group_count != 1: return None
        if 4 + 2 * index_count != block_size: return None
        if index_count % 3: return None
        if index_count == 0: return []

        return list(reader.read(f"{index_count}H"))

    def parse_blood_money_vertices(self, reader: Reader, vertex_offset: int, vertex_count: int, vertex_stride: int) -> tuple[list, list, list, list, Optional[list], Optional[list]]:
        """Read Blood Money vertices at the given stride.

        The stride IS the layout discriminator - there is no per-mesh flag to read. Four forms
        matter, all derived from BMExport / id-daemon readers and then checked against real bytes:

          52B WEIGHTED (skinned geometry - characters and animated props)
              position(3f) | weights(3f) | boneIndices(3ub) | pad(1ub)
              | normal(3sb) | pad(1ub) | color(4ub BGRA) | u(f) | v(f) | trailing(8)
              Bone indices are stored PRE-MULTIPLIED BY 3 and must be divided back down, the
              same convention Absolution uses for its boneRemapValues. Only three weights are
              stored; the fourth is implied as (1 - sum) and is not reconstructed here.

          40B STATIC (unweighted level geometry - the overwhelming majority)
              position(3f) | slot(4) | color(4ub BGRA) | u(f) | v(f) | fieldA(4) | fieldB(4)
              A face-normal correlation test (with a 36B float-normal control) shows the 40B
              slot holds NO usable normal in any encoding - its alpha byte is always 0x00 and the
              trailing floats are always 0.0. So we emit no normal for static meshes; Blender
              computes them. This is a confirmed negative, not an open question.

          36B NORMAL (a minority of level meshes - same interleave as the classic games)
              position(3f) | normal(3f) | color(4ub BGRA) | u(f) | v(f)
              The +12 float3 normal reads unit-length 500/500 on real data, so unlike the 40B
              slot this one IS a real normal and is applied.

          16B SIMPLE
              position(3f) | color(4ub BGRA)   (position + colour only)

        Position, colour and UV are byte-confirmed on every form. Only the weighted path emits
        skin data; the others return None for weights/indices."""
        positions:    list[tuple[float, float, float]] = []
        normals:      list[tuple[float, float, float]] = []
        colors:       list[tuple[float, float, float, float]] = []
        uvs:          list[tuple[float, float]] = []
        bone_weights: list[list[float]] = []
        bone_indices: list[list[int]] = []

        for vertex in range(vertex_count):
            base = vertex_offset + vertex_stride * vertex
            reader.seek(base)

            positions.append(reader.vec3f())

            if vertex_stride == GLACIER1_HBM_VERTEX_STRIDE_WEIGHTED:
                weights = list(reader.vec3f())
                # Stored pre-multiplied by 3; divide back to real bone indices.
                raw_bones = reader.vec3ub()
                bone_weights.append(weights)
                bone_indices.append([value // GLACIER1_BONE_INDEX_DIVISOR for value in raw_bones])
                reader.skip(1)  # 4th bone slot, unused by the three-weight layout

                nx, ny, nz = reader.vec3sb()
                normals.append(normalize_signed_byte_vector(nx, ny, nz))
                reader.skip(1)  # padding after the normal triple

                b, g, r, a = reader.vec4ub()  # colour, BGRA
                colors.append(convert_vertex_color(r, g, b, a))
                uvs.append(invert_uv_map(reader.vec2f()))
                # Trailing 8 bytes are not identified; skipped rather than guessed at.

            elif vertex_stride == GLACIER1_HBM_VERTEX_STRIDE_NORMAL:
                # Same interleave as the classic 36B vertex: a genuine float3 normal at +12.
                normals.append(reader.vec3f())
                b, g, r, a = reader.vec4ub()  # colour, BGRA
                colors.append(convert_vertex_color(r, g, b, a))
                uvs.append(invert_uv_map(reader.vec2f()))

            elif vertex_stride == GLACIER1_HBM_VERTEX_STRIDE_STATIC:
                reader.skip(4)                # slot carries no usable normal (confirmed negative)
                b, g, r, a = reader.vec4ub()  # colour, BGRA
                colors.append(convert_vertex_color(r, g, b, a))
                uvs.append(invert_uv_map(reader.vec2f()))
                # Remaining 8 bytes: two unidentified fields, always 0.0 in the sample.

            else:  # 16B simple, or any other accepted stride: position + colour only
                b, g, r, a = reader.vec4ub()
                colors.append(convert_vertex_color(r, g, b, a))

        return positions, normals, colors, uvs, (bone_weights or None), (bone_indices or None)

    # =====================================================
    # BLOOD MONEY MODEL GRAPH
    # =====================================================

    def resolve_model_parts(self, model_chunk: int) -> list[int]:
        """Blood Money only: walk a MODEL chunk down to the buffer chunks it draws.

        The companion GMS does not point at geometry directly - it points at a model chunk, and
        the model owns a part list which owns the buffers. The chain, per id-daemon's reader:

            model  +0x10 skeletonChunk | +0x14 partCount | +0x18 geometryChunk
            geometry            partCount * uint part chunk indices
            part   +0x0E lodMask | +0x0F variant | +0x12 material | +0x28 geometryBufferChunk
            geometryBuffer      +0x00 uint bufferChunk   <- the mesh we parsed

        Parts whose lodMask has bit 0 clear are lower-detail LODs and are skipped, matching the
        `(lod & 1) == 0` test in the reference reader. Returns the buffer chunk indices, which are
        the keys of `mesh_index_by_reference`. Anything malformed is skipped rather than raised:
        the graph is only ~88% resolvable on real data and a bad model must not abort the import."""
        parts: list[int] = []
        if not self.blocks or not (0 <= model_chunk < self.block_count): return parts

        block_offset, block_size = self.blocks[model_chunk][0], self.blocks[model_chunk][1]
        if block_size < GLACIER1_HBM_MODEL_CHUNK_SIZE: return parts

        data = self.buffer
        def read_u32(offset: int) -> int: return struct.unpack_from("<I", data, offset)[0]

        try:
            part_count      = read_u32(block_offset + GLACIER1_HBM_MODEL_PART_COUNT)
            geometry_chunk  = read_u32(block_offset + GLACIER1_HBM_MODEL_GEOMETRY)
        except struct.error: return parts

        if not (0 < part_count <= GLACIER1_HBM_MAX_PARTS): return parts
        if not (0 <= geometry_chunk < self.block_count): return parts
        if self.blocks[geometry_chunk][1] < 4 * part_count: return parts

        geometry_offset = self.blocks[geometry_chunk][0]
        for part in range(part_count):
            try:
                part_chunk = read_u32(geometry_offset + 4 * part)
                if not (0 <= part_chunk < self.block_count): continue
                part_offset, part_size = self.blocks[part_chunk][0], self.blocks[part_chunk][1]
                if part_size < GLACIER1_HBM_PART_CHUNK_SIZE: continue

                # Bit 0 of the LOD mask marks the highest-detail part; everything else is a
                # lower LOD or a viewmodel variant and is not built.
                lod_mask = data[part_offset + GLACIER1_HBM_PART_LOD_MASK]
                if (lod_mask & 1) == 0: continue

                buffer_holder = read_u32(part_offset + GLACIER1_HBM_PART_GEOMETRY_BUFFER)
                if not (0 <= buffer_holder < self.block_count): continue
                if self.blocks[buffer_holder][1] < 4: continue

                buffer_chunk = read_u32(self.blocks[buffer_holder][0])
                if 0 <= buffer_chunk < self.block_count: parts.append(buffer_chunk)
            except (struct.error, IndexError): continue

        return parts

    # =====================================================
    # BLOOD MONEY SKELETON
    # =====================================================

    def parse_all_blood_money_skeletons(self) -> None:
        """Enumerate every MODEL chunk and decode the skeleton it references (if any).

        A model is a 0x40-byte chunk whose first u32 is the model tag. Its +0x10 word points at a
        skeleton pointer chunk, whose +4 word points at the skeleton header. Each decoded skeleton
        is stored under its model chunk index, and every buffer chunk the model draws is tagged
        back to that model so a weighted submesh can locate its skeleton."""
        data = self.buffer
        def read_u32(offset: int) -> int: return struct.unpack_from("<I", data, offset)[0]

        skeleton_count = 0
        for i in range(self.block_count):
            offset, size = self.blocks[i][0], self.blocks[i][1]
            if size != GLACIER1_HBM_MODEL_CHUNK_TOTAL_SIZE: continue
            try:
                if read_u32(offset) != GLACIER1_HBM_MODEL_TAG: continue
                skeleton_pointer = read_u32(offset + GLACIER1_HBM_MODEL_SKELETON)
            except struct.error: continue
            if not (0 < skeleton_pointer < self.block_count): continue

            skeleton = self.parse_blood_money_skeleton(skeleton_pointer)
            if skeleton is None: continue

            self.skeletons[i] = skeleton
            skeleton_count += 1

            # Tie every buffer chunk this model draws back to the model, so the handler can bind.
            for buffer_chunk in self.resolve_model_parts(i):
                self.skeleton_by_buffer[buffer_chunk] = i

        if skeleton_count: print(f"Skeletons: decoded {skeleton_count} model skeletons.")

    def parse_blood_money_skeleton(self, skeleton_pointer_chunk: int) -> Optional[dict]:
        """Decode one Blood Money skeleton from its pointer chunk.

        pointer chunk +4 -> header chunk. Header chunk: [u32 boneCount][u32 subChunkIndex ...].
        We identify the three sub-chunks we need by their (size / boneCount) stride rather than by
        position, since their order is not guaranteed:

            64 B/bone -> bone definitions; null-terminated name at def+28
             4 B/bone -> parent table; each entry a byte offset, /48 -> parent bone index
            28 B/bone -> local transform; quaternion(4f) + position(3f)

        Returns a dict with `bone_count`, `names`, `parents`, `local_rotations` (x,y,z,w) and
        `local_positions`, or None if the chunk is not a well-formed skeleton header. All three
        arrays were validated byte-exact on a 100-bone character (100/100 clean names, 100% valid
        parents, 100/100 unit quaternions composing to an anatomically correct bind pose)."""
        data = self.buffer
        def read_u32(offset: int) -> int: return struct.unpack_from("<I", data, offset)[0]

        if not (0 <= skeleton_pointer_chunk < self.block_count): return None
        pointer_offset, pointer_size = self.blocks[skeleton_pointer_chunk][0], self.blocks[skeleton_pointer_chunk][1]
        if pointer_size < 8: return None

        try:
            header_chunk = read_u32(pointer_offset + GLACIER1_HBM_SKELETON_POINTER_FIELD)
        except struct.error: return None
        if not (0 <= header_chunk < self.block_count): return None

        header_offset, header_size = self.blocks[header_chunk][0], self.blocks[header_chunk][1]
        if header_size < 8: return None

        try:
            bone_count = read_u32(header_offset)
        except struct.error: return None
        if not (0 < bone_count <= GLACIER1_HBM_MAX_BONES): return None

        # Header sub-chunk index array follows the count. Resolve each by its per-bone stride.
        sub_chunks = []
        word_count = header_size // 4
        for word in range(1, word_count):
            try: candidate = read_u32(header_offset + 4 * word)
            except struct.error: break
            if 0 <= candidate < self.block_count: sub_chunks.append(candidate)

        def find_sub_chunk(target_stride: int) -> Optional[int]:
            """First referenced sub-chunk whose size is exactly boneCount * target_stride."""
            for chunk in sub_chunks:
                if self.blocks[chunk][1] == bone_count * target_stride: return chunk
            return None

        def_chunk    = find_sub_chunk(GLACIER1_HBM_BONE_DEF_STRIDE)
        parent_chunk = find_sub_chunk(GLACIER1_HBM_BONE_PARENT_STRIDE)
        local_chunk  = find_sub_chunk(GLACIER1_HBM_BONE_LOCAL_STRIDE)
        if def_chunk is None or parent_chunk is None or local_chunk is None: return None

        # --- NAMES (def chunk, name at def+28) ---
        names: list[str] = []
        def_offset = self.blocks[def_chunk][0]
        for bone in range(bone_count):
            name_start = def_offset + GLACIER1_HBM_BONE_DEF_STRIDE * bone + GLACIER1_HBM_BONE_NAME_OFFSET
            raw = data[name_start:name_start + GLACIER1_HBM_BONE_NAME_MAX]
            terminator = raw.find(b"\x00")
            name = raw[:terminator] if terminator >= 0 else raw
            names.append(name.decode("ascii", errors="replace") or f"bone_{bone}")

        # --- PARENTS (parent chunk, byte offset / 48) ---
        parent_offset = self.blocks[parent_chunk][0]
        raw_parents = struct.unpack_from(f"<{bone_count}i", data, parent_offset)
        parents = [value // GLACIER1_HBM_BONE_PARENT_DIVISOR for value in raw_parents]
        parents = [parent if 0 <= parent < bone_count else -1 for parent in parents]

        # --- LOCAL TRANSFORMS (local chunk, quat(4f)+pos(3f)) ---
        local_rotations: list[tuple[float, float, float, float]] = []
        local_positions: list[tuple[float, float, float]] = []
        local_offset = self.blocks[local_chunk][0]
        for bone in range(bone_count):
            base = local_offset + GLACIER1_HBM_BONE_LOCAL_STRIDE * bone
            qx, qy, qz, qw, px, py, pz = struct.unpack_from("<7f", data, base)
            local_rotations.append((qx, qy, qz, qw))
            local_positions.append((px, py, pz))

        print(f"  Skeleton: {bone_count} bones (root '{names[0]}')")
        return {
            "bone_count":      bone_count,
            "names":           names,
            "parents":         parents,
            "local_rotations": local_rotations,  # (x, y, z, w) per bone
            "local_positions": local_positions,  # (x, y, z) per bone
        }

    # =====================================================
    # SHARED: INDEX BLOCK
    # =====================================================

    def parse_index_block(self, reader: Reader, index_pointer: int) -> list[list[int]]:
        """Read an index block into a list of strips. Layout: [ushort stripCount] then
        stripCount * ([ushort indexCount][indexCount ushorts]). Shared by every game."""
        reader.seek(index_pointer)
        strip_count = reader.ushort()

        strips: list[list[int]] = []
        for _ in range(strip_count):
            index_count = reader.ushort()
            if index_count == 0:
                strips.append([])
                continue
            strips.append(list(reader.read(f"{index_count}H")))

        return strips

    # =====================================================
    # MESH RECORD BUILDER (unified output shape)
    # =====================================================

    def build_mesh_record(self, anchor: int, block_index: Optional[int], vertex_stride: int, vertex_count: int, index_count: int, strip_count: int, positions: list, normals: list, colors: list, uv_channels: list, triangles: list, bone_weights: Optional[list] = None, bone_indices: Optional[list] = None, skeleton_model: Optional[int] = None) -> dict:
        """Pack one submesh's decoded data into the unified record dict the model handler consumes.

        Kept deliberately flat and game-agnostic: classic and Blood Money both funnel through here,
        so the handler never has to know which container produced the mesh."""
        return {
            "game":          self.game,
            "anchor":        anchor,        # classic: PRM byte offset | HBM: buffer chunk offset
            "block_index":   block_index,   # HBM only: index into the block table (else None)
            "vertex_stride": vertex_stride,
            "vertex_count":  vertex_count,
            "index_count":   index_count,
            "strip_count":   strip_count,
            "positions":     positions,
            "normals":       normals,       # may be empty (static HBM verts have no trusted normal)
            "vertex_colors": colors,
            "uv_channels":   uv_channels,   # list of channels; each channel is a list of (u, v)
            "triangles":     triangles,     # flattened, de-stitched, correct winding
            "bone_weights":  bone_weights,  # per-vertex float weights, or None when unweighted
            "bone_indices":  bone_indices,  # per-vertex bone indices, or None when unweighted
            "is_weighted":   bone_weights is not None,
            "skeleton_model": skeleton_model,  # HBM: owning model chunk index, or None
        }
