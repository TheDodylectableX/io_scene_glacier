# =====================================================
#   GLACIER 2 PRIM (RENDERPRIMITIVE) PARSER
#       Parses IOI Interactive's Glacier 2
#       RenderPrimitive model format used by:
#           - Hitman: Absolution
#           - Hitman: World of Assassination (Trilogy)
#           - 007: First Light (Bond)
#
#       Mirrors PRIM.bt 1:1. If you change the binary
#       template, mirror the change here.
# =====================================================

import struct
from ..io import Reader
from ..utilities import *

# ==========
# CONSTANTS
# ==========

# PRIM_OBJECT_TYPE (ushort) - The top-level kind tag for a PRIM record.
PRIM_OBJECT_TYPE_UNKNOWN       = 0
PRIM_OBJECT_TYPE_OBJECT_HEADER = 1
PRIM_OBJECT_TYPE_MESH          = 2
PRIM_OBJECT_TYPE_DECAL         = 3
PRIM_OBJECT_TYPE_SPRITES       = 4
PRIM_OBJECT_TYPE_SHAPE         = 5
PRIM_OBJECT_TYPE_UNUSED        = 6

# PRIM_OBJECT_SUBTYPE (ubyte) - Per-mesh layout discriminator.
# In 007FL: The WEIGHTED layout differs from the LINKED layout.
# In Absolution and The WoA Trilogy: The subtype mostly governs how many UV maps/channels the mesh carries.
PRIM_SUBTYPE_STANDARD      = 0
PRIM_SUBTYPE_LINKED        = 1
PRIM_SUBTYPE_WEIGHTED      = 2
PRIM_SUBTYPE_STANDARD_UV_2 = 3
PRIM_SUBTYPE_STANDARD_UV_3 = 4
PRIM_SUBTYPE_STANDARD_UV_4 = 5

PRIM_SUBTYPE_NAMES = {
    PRIM_SUBTYPE_STANDARD:      "STANDARD",
    PRIM_SUBTYPE_LINKED:        "LINKED",
    PRIM_SUBTYPE_WEIGHTED:      "WEIGHTED",
    PRIM_SUBTYPE_STANDARD_UV_2: "STANDARD_UV_2",
    PRIM_SUBTYPE_STANDARD_UV_3: "STANDARD_UV_3",
    PRIM_SUBTYPE_STANDARD_UV_4: "STANDARD_UV_4",
}

# UV channel count implied by subtype in the Absolution / Trilogy layout. 007FL ignores this and always uses 1 map/channel for LINKED and WEIGHTED meshes except hair meshes which can have 2.
TRILOGY_SUBTYPE_UV_CHANNEL_COUNT = {
    PRIM_SUBTYPE_STANDARD:      1,
    PRIM_SUBTYPE_LINKED:        1,
    PRIM_SUBTYPE_WEIGHTED:      1,
    PRIM_SUBTYPE_STANDARD_UV_2: 2,
    PRIM_SUBTYPE_STANDARD_UV_3: 3,
    PRIM_SUBTYPE_STANDARD_UV_4: 4,
}

# BORG Resource Sentinel/Marker: 0xFFFFFFFF means "No embedded skeleton, The BORG asset is referenced externally by resource hash".
BORG_RESOURCE_NONE = 0xFFFFFFFF

# =====================================================
# MAIN PARSER CLASS
# =====================================================

class PRIM():
    """Glacier 2 RenderPrimitive parser. Parses a fully-decoded object model of every mesh in a .PRIM file."""
    def __init__(self, file_path: str, game: str):
        """Construct the parser and run the full parse pass."""

        super().__init__()

        # ===============================
        # == CLASS MEMBERS ==============
        # ===============================

        # -- INPUT METADATA
        self.model_file: str = file_path
        """The path to the source .PRIM file."""

        self.game: str = game
        """Which Glacier 2 title produced this file. Drives parser branching: GLACIER2_TRILOGY or GLACIER2_BOND."""

        # -- FILE-LEVEL HEADER
        self.header_offset: int = 0
        """u64 pointer to PRIM_OBJECT_HEADER read from the prolog. The object header sits at the END of the file."""

        self.object_header: dict = {}
        """The decoded PRIM_OBJECT_HEADER. Carries the file-level flags, bone-rig resource index, object count and overall bounding box."""

        # -- BLENDER OBJECT LIST
        self.blender_objects: list[bpy.types.Object] = []
        """Master list of all Blender objects."""

        # -- MASTER MESH LIST
        self.objects: list[dict] = []
        """Per-mesh parsed data. One entry per object listed in objectOffsets[]. See build_mesh_record for the shape of each entry."""

        # -- VALIDATION
        if game not in (GLACIER2_ABSOLUTION, GLACIER2_TRILOGY, GLACIER2_BOND): raise ValueError(f"Unsupported game type for PRIM parsing: '{game}'. Use GLACIER2_ABSOLUTION, GLACIER2_TRILOGY or GLACIER2_BOND.")

        # ===============================
        # == PARSE THE DATA =============
        # ===============================
        self.parse_model_file()

    # =====================================================
    # TOP-LEVEL DRIVER
    # =====================================================

    def parse_model_file(self) -> None:
        """Parse the model file. Parse order strictly matches PRIM.bt."""
        print(f"\nParsing PRIM model ({self.game}): {self.model_file}\n")
        file_bytes = open(self.model_file, "rb").read()
        reader = Reader(file_bytes)

        # ----------------------------------------------------------------
        # HEADER - Pointer/Offset to PRIM_OBJECT_HEADER + 8 bytes padding
        # ----------------------------------------------------------------
        self.header_offset = reader.uint64()
        print(f"Object Header Offset: 0x{self.header_offset:08X}")

        # Sanity-check the prolog before seeking on it. A partial extraction, a raw stream that
        # was never a PRIM, or a big-endian console build all land here with an offset pointing
        # somewhere past the end of the file - without this check the first field read fails deep
        # inside the reader with a struct error that says nothing about the actual problem.
        if self.header_offset == 0 or self.header_offset + 4 > len(file_bytes):
            raise ValueError(
                f"Not a valid PRIM: the header pointer at the start of the file reads "
                f"0x{self.header_offset:X}, which is outside the file ({len(file_bytes):,} bytes). "
                f"This usually means the file is a partial extraction, isn't a PRIM at all, or is "
                f"a big-endian console build (which this importer doesn't support)."
            )

        # -----------------------------------------
        # PRIM_OBJECT_HEADER (file footer)
        # -----------------------------------------
        reader.seek(self.header_offset)
        self.parse_object_header(reader)

        # -----------------------------------------
        # OBJECT OFFSET TABLE
        # -----------------------------------------
        object_count = self.object_header["object_count"]
        reader.seek(self.object_header["object_table_offset"])
        object_offsets = [reader.uint32() for _ in range(object_count)]
        print(f"\nReading {object_count} object offsets from 0x{self.object_header['object_table_offset']:08X}")

        # -----------------------------------------
        # MESHES - one PrimMesh per object offset
        # -----------------------------------------
        is_weighted = self.object_header["flags"]["is_weighted_object"]
        for i, object_offset in enumerate(object_offsets):
            print(f"\n--- Mesh {i + 1} of {object_count} @ 0x{object_offset:08X} ---")
            reader.seek(object_offset)
            mesh_record = self.parse_prim_mesh(reader, is_weighted)
            self.objects.append(mesh_record)

        print(f"\nPRIM PARSING COMPLETE!  {object_count} meshes parsed.\n")

    # =====================================================
    # FILE-LEVEL HEADER
    # =====================================================

    def parse_object_header(self, reader: Reader) -> None:
        """Parse PRIM_OBJECT_HEADER (the file footer pointed to by the prolog u64)."""
        # PRIM_HEADER (4 B): drawDestination (u8), packType (u8), Type (u16)
        draw_destination = reader.ubyte()
        pack_type        = reader.ubyte()
        record_type      = reader.ushort()

        # PRIM_OBJECT_HEADER_FLAGS (u32 bitfield)
        flags_u32 = reader.uint32()
        flags = self.decode_object_header_flags(flags_u32)

        # Bond inserts an extra u32 here (purpose unconfirmed - 0xFFFFFFFF or 0x00000000 in observed samples).
        # Hypothesis: high 32 bits of a 64-bit bone-rig reference (then boneRigResourceIndex would be the low 32 bits).
        unknown_padding_bond = reader.uint32() if self.game == GLACIER2_BOND else 0

        bone_rig_resource_index    = reader.uint32()
        object_count               = reader.uint32()
        object_table_offset        = reader.uint32()
        total_bounding_box_minimum = reader.vec3f()
        total_bounding_box_maximum = reader.vec3f()

        self.object_header = {
            "record_type":             record_type,
            "draw_destination":        draw_destination,
            "pack_type":               pack_type,
            "flags_raw":               flags_u32,
            "flags":                   flags,
            "unknown_padding_bond":    unknown_padding_bond,
            "bone_rig_resource_index": bone_rig_resource_index,
            "object_count":            object_count,
            "object_table_offset":     object_table_offset,
            "bounding_box_min":        total_bounding_box_minimum,
            "bounding_box_max":        total_bounding_box_maximum,
        }

        print(f"  Record Type: {record_type} | Object Count: {object_count}")
        print(f"  Flags: weighted={flags['is_weighted_object']} | linked={flags['is_linked_object']} | hasBones={flags['has_bones']} | highRes={flags['has_high_resolution']}")
        print(f"  Bone Rig Resource Index: 0x{bone_rig_resource_index:08X}" + ("  (no embedded skeleton)" if bone_rig_resource_index == BORG_RESOURCE_NONE else ""))
        print(f"  Object Table Offset: 0x{object_table_offset:08X}")

    def decode_object_header_flags(self, flags_u32: int) -> dict:
        """Unpack the file-level PRIM_OBJECT_HEADER_FLAGS u32 bitfield (LSB-first, mirroring 010 Editor)."""
        return {
            "has_bones":           bool(flags_u32 & (1 << 0)),
            "has_frames":          bool(flags_u32 & (1 << 1)),
            "is_linked_object":    bool(flags_u32 & (1 << 2)),
            "is_weighted_object":  bool(flags_u32 & (1 << 3)),
            # Bits 4-7 reserved
            "use_bounds":          bool(flags_u32 & (1 << 8)),
            "has_high_resolution": bool(flags_u32 & (1 << 9)),
            # Bits 10-31 reserved
        }

    def decode_object_flags(self, flags_u8: int) -> dict:
        """Unpack the per-mesh PRIM_OBJECT_FLAGS u8 bitfield (LSB-first)."""
        return {
            "is_x_axis_locked":    bool(flags_u8 & (1 << 0)),
            "is_y_axis_locked":    bool(flags_u8 & (1 << 1)),
            "is_z_axis_locked":    bool(flags_u8 & (1 << 2)),
            "is_high_resolution":  bool(flags_u8 & (1 << 3)),
            "has_ps3_edge":        bool(flags_u8 & (1 << 4)),
            "use_color_1":         bool(flags_u8 & (1 << 5)),
            "has_no_physics_prop": bool(flags_u8 & (1 << 6)),
            # Bit 7 reserved
        }

    # =====================================================
    # PRIM_OBJECT (44 B common header used by PrimMesh and PrimSubMesh)
    # =====================================================

    def parse_prim_object(self, reader: Reader) -> dict:
        """Parse the 44-byte PRIM_OBJECT block (PRIM_HEADER + per-mesh metadata + AABB)."""
        # PRIM_HEADER (4 B)
        draw_destination = reader.ubyte()
        pack_type        = reader.ubyte()
        record_type      = reader.ushort()
        sub_type         = reader.ubyte()
        # The properties byte and the LOD-mask byte occupy offsets +5 and +6 but their ORDER is swapped between the two games (verified on shipping assets):
        #   WoA  : +5 = properties (observed 0)            , +6 = lod_mask (1,2,4,8,16,0xE0,... per LOD level)
        #   007FL: +5 = lod_mask (1,2,4,8,16,0x20,0xFF,...), +6 = properties (0 / 0x20)
        if self.game == GLACIER2_BOND:
            lod_mask      = reader.ubyte()
            properties_u8 = reader.ubyte()
        else:
            properties_u8 = reader.ubyte()
            lod_mask      = reader.ubyte()
        variant_id        = reader.ubyte()
        z_bias            = reader.ubyte()
        z_offset          = reader.ubyte()
        material_id       = reader.ushort()
        wire_color        = reader.int32()
        color_1           = reader.vec4ub()
        bbox_min          = reader.vec3f()
        bbox_max          = reader.vec3f()

        return {
            "draw_destination": draw_destination,
            "pack_type":        pack_type,
            "record_type":      record_type,
            "sub_type":         sub_type,
            "sub_type_name":    PRIM_SUBTYPE_NAMES.get(sub_type, f"UNKNOWN_{sub_type}"),
            "properties_raw":   properties_u8,
            "properties":       self.decode_object_flags(properties_u8),
            "lod_mask":         lod_mask,
            "lod_index":        ((lod_mask & -lod_mask).bit_length() - 1) if lod_mask else 0,
            "variant_id":       variant_id,
            "z_bias":           z_bias,
            "z_offset":         z_offset,
            "material_id":      material_id,
            "wire_color":       wire_color,
            "color_1":          color_1,
            "bbox_min":         bbox_min,
            "bbox_max":         bbox_max,
        }

    # ==================================
    # PRIM_MESH - Game-aware dispatcher
    # ==================================

    def parse_prim_mesh(self, reader: Reader, is_weighted: bool) -> dict:
        """Parse one mesh from the file, dispatching on game."""
        if self.game == GLACIER2_BOND:        return self.parse_prim_mesh_bond(reader, is_weighted)
        if self.game == GLACIER2_ABSOLUTION:  return self.parse_prim_mesh_absolution(reader, is_weighted)
        return self.parse_prim_mesh_trilogy(reader, is_weighted)

    # ---------------------------------------------------------------------
    # TRILOGY / HITMAN PrimMesh
    # The mesh struct here is a thin shell - the actual geometry lives one
    # indirection deeper in PRIM_SUBMESH_HITMAN, which subMeshTableOffset
    # points at via a single-entry u32 table.
    # ---------------------------------------------------------------------

    def parse_prim_mesh_trilogy(self, reader: Reader, is_weighted: bool) -> dict:
        """Parse a Trilogy PrimMesh + the inner PrimSubMesh it points to."""
        # ----- PRIM_OBJECT (44 bytes) -----
        prim_object = self.parse_prim_object(reader)
        print(f"  PrimMesh: subType={prim_object['sub_type_name']} | materialID={prim_object['material_id']}")

        # ----- Trilogy mesh body: subMeshTable pointer + quantization params + cloth -----
        sub_mesh_table_offset = reader.uint32()
        vertex_position_scale = reader.vec4f()
        vertex_position_bias  = reader.vec4f()
        uv_coord_scale        = reader.vec2f()
        uv_coord_bias         = reader.vec2f()
        cloth_id_raw          = reader.uint32()

        # ----- Optional weighted trailer (4 fields / 16 bytes) -----
        # Field ORDER differs from Bond: copyBones first, boneIndices/boneInfo last.
        copy_bones_count        = 0
        copy_bones_offset       = 0
        bone_indices_offset     = 0
        bone_information_offset = 0
        if is_weighted:
            copy_bones_count        = reader.uint32()
            copy_bones_offset       = reader.uint32()
            bone_indices_offset     = reader.uint32()
            bone_information_offset = reader.uint32()

        # ----- Follow subMeshTable -> single uint entry -> PRIM_SUBMESH_HITMAN -----
        saved_position = reader.tell()
        reader.seek(sub_mesh_table_offset)
        sub_mesh_offset = reader.uint32() # Always a single entry in observed Trilogy files
        reader.seek(sub_mesh_offset)

        submesh = self.parse_prim_submesh_trilogy(reader, is_weighted, prim_object)
        reader.seek(saved_position)

        # ----- Build the unified mesh record from submesh data + outer-mesh metadata -----
        record = self.build_mesh_record(
            prim_object        = prim_object,
            sub_type           = prim_object["sub_type"],
            vertex_count       = submesh["vertex_count"],
            index_count        = submesh["index_count"],
            position_scale     = vertex_position_scale,
            position_bias      = vertex_position_bias,
            uv_scale           = uv_coord_scale,
            uv_bias            = uv_coord_bias,
            cloth_id_raw       = cloth_id_raw,
            positions          = submesh["positions"],
            normals            = submesh["normals"],
            tangents           = submesh["tangents"],
            bitangents         = submesh["bitangents"],
            uv_channels        = submesh["uv_channels"],
            vertex_colors      = submesh["vertex_colors"],
            triangles          = submesh["triangles"],
            bone_weights       = submesh["bone_weights"],
            bone_local_indices = submesh["bone_local_indices"],
            collision          = submesh["collision"],
        )

        # ----- Walk weighted trailer pointers (BoneInfo + BoneIndices) -----
        if is_weighted:
            if bone_information_offset != 0:
                reader.seek(bone_information_offset)
                record["bone_info"] = self.parse_bone_info(reader)

            if bone_indices_offset != 0:
                reader.seek(bone_indices_offset)
                record["bone_palette"] = self.parse_bone_indices_buffer(reader)

            record["copy_bones_count"] = copy_bones_count
            if copy_bones_offset != 0 and copy_bones_count > 0:
                reader.seek(copy_bones_offset)
                record["copy_bones_data"] = bytes(reader.read_bytes(copy_bones_count))

        return record

    def parse_prim_submesh_trilogy(self, reader: Reader, is_weighted: bool, parent_object: dict) -> dict:
        """Parse a PRIM_SUBMESH_HITMAN block + dive into its vertex/index buffers."""
        # ----- Submesh's own PRIM_OBJECT (44 bytes) -----
        submesh_object = self.parse_prim_object(reader)

        vertex_count           = reader.uint32()
        vertex_buffer_offset   = reader.uint32()
        index_count            = reader.uint32()
        additional_index_count = reader.uint32()
        index_buffer_offset    = reader.uint32()
        collision_offset       = reader.uint32()
        cloth_offset           = reader.uint32()
        uv_channel_count       = reader.uint32()

        print(f"  PrimSubMesh: verts={vertex_count} | indices={index_count} (+{additional_index_count}) | UV channels={uv_channel_count}")

        # ----- Index buffer (primary + trailing additional indices) -----
        reader.seek(index_buffer_offset)
        triangles = self.parse_index_buffer(reader, index_count + additional_index_count)

        # ----- Vertex buffer (HITMAN_VERTEX_BUFFER) -----
        # readColor gate, mirrored from format.py: see PRIM.bt comment in PRIM_SUBMESH_HITMAN.
        read_color = ((not parent_object["properties"]["use_color_1"]) or is_weighted) and (not submesh_object["properties"]["use_color_1"])
        is_high_resolution = parent_object["properties"]["is_high_resolution"]

        reader.seek(vertex_buffer_offset)
        vertex_data = self.parse_vertex_buffer_trilogy(
            reader             = reader,
            vertex_count       = vertex_count,
            uv_channel_count   = uv_channel_count,
            is_weighted        = is_weighted,
            is_high_resolution = is_high_resolution,
            read_color         = read_color,
        )

        # ----- Optional broad-phase collision -----
        collision = None
        if collision_offset != 0:
            reader.seek(collision_offset)
            collision = self.parse_box_coli(reader)

        # ----- Cloth blob - format not yet decoded; note its presence only -----
        cloth_present = (cloth_offset != 0)
        if cloth_present: print(f"  Cloth data present at 0x{cloth_offset:08X} (not decoded)")

        return {
            "vertex_count":       vertex_count,
            "index_count":        index_count + additional_index_count,
            "positions":          vertex_data["positions"],
            "normals":            vertex_data["normals"],
            "tangents":           vertex_data["tangents"],
            "bitangents":         vertex_data["bitangents"],
            "uv_channels":        vertex_data["uv_channels"],
            "vertex_colors":      vertex_data["vertex_colors"],
            "bone_weights":       vertex_data["bone_weights"],
            "bone_local_indices": vertex_data["bone_local_indices"],
            "triangles":          triangles,
            "collision":          collision,
        }

    # ---------------------------------------------------------------------
    # ABSOLUTION PrimMesh
    # Key differences from WoA:
    #   * PrimMesh body has TWO extra uint fields (clothId + a flags/pad word) between the subMeshTable pointer and posScale so scale/bias sit 8 bytes later than WoA.
    #   * PrimSubMesh has NO additionalIndexCount field.
    #   * Vertex stream order is subtype-specific:
    #       - STANDARD : interleaved per vertex - pos, Normal, COLOR, Tangent, Binormal, UV*n (note: colour sits BETWEEN normal and tangent)
    #       - LINKED   : planar - all positions, then per vertex Normal, Tangent, Binormal, UV*n, COLOR
    #       - WEIGHTED : planar - all positions, then all (4 weights + 4 boneRemap), then per vertex Normal, Tangent, Binormal, UV*n, COLOR
    #   * WEIGHTED skinning is 8 B/vert: 4 u8 weights (sum 255) + 4 u8 boneRemapValues, with the weightA==1 -> weightA=0, weightD+=1 fixup. boneRemapValues resolve to global bone indices via BoneInfo.bone_remap (GetBoneIndex(value/3)) at weight-apply time.
    #   * STANDARD reads PrimMesh (No trailer); LINKED and WEIGHTED read PrimMeshWeighted (4-field trailer: numCopyBones, copyBonesOffset, boneIndices, boneInfo) and a BoneInfo block.
    # ---------------------------------------------------------------------

    def parse_prim_mesh_absolution(self, reader: Reader, is_weighted: bool) -> dict:
        """Parse one Absolution mesh + its inner submesh, dispatching on subType."""
        # ----- PRIM_OBJECT (44 bytes including color1) -----
        prim_object = self.parse_prim_object(reader)
        sub_type = prim_object["sub_type"]
        print(f"  PrimMesh: subType={prim_object['sub_type_name']} | materialID={prim_object['material_id']} | LOD=0x{prim_object['lod_mask']:02X}")

        # ----- PrimMesh body -----
        sub_mesh_table_offset = reader.uint32()
        cloth_id_raw          = reader.uint32() # f1: clothID
        mesh_flags_raw        = reader.uint32() # f2: Flags / pad (0x10000 on props, 0 on chars)
        vertex_position_scale = reader.vec4f()
        vertex_position_bias  = reader.vec4f()
        uv_coord_scale        = reader.vec2f()
        uv_coord_bias         = reader.vec2f()

        # ----- Weighted/Linked trailer (PrimMeshWeighted) -----
        # Both LINKED and WEIGHTED subtypes carry the trailer and a BoneInfo block; STANDARD doesn't.
        has_trailer = sub_type in (PRIM_SUBTYPE_LINKED, PRIM_SUBTYPE_WEIGHTED)
        copy_bones_count = copy_bones_offset = bone_indices_offset = bone_information_offset = 0
        if has_trailer:
            copy_bones_count        = reader.uint32()
            copy_bones_offset       = reader.uint32()
            bone_indices_offset     = reader.uint32()
            bone_information_offset = reader.uint32()

        # ----- Follow subMeshTable -> single uint entry -> PrimSubMesh -----
        saved_position = reader.tell()
        reader.seek(sub_mesh_table_offset)
        sub_mesh_offset = reader.uint32()
        reader.seek(sub_mesh_offset)

        submesh = self.parse_prim_submesh_absolution(reader, sub_type, is_weighted, vertex_position_scale, vertex_position_bias, uv_coord_scale, uv_coord_bias)
        reader.seek(saved_position)

        record = self.build_mesh_record(
            prim_object        = prim_object,
            sub_type           = sub_type,
            vertex_count       = submesh["vertex_count"],
            index_count        = submesh["index_count"],
            position_scale     = vertex_position_scale,
            position_bias      = vertex_position_bias,
            uv_scale           = uv_coord_scale,
            uv_bias            = uv_coord_bias,
            cloth_id_raw       = cloth_id_raw,
            positions          = submesh["positions"],
            normals            = submesh["normals"],
            tangents           = submesh["tangents"],
            bitangents         = submesh["bitangents"],
            uv_channels        = submesh["uv_channels"],
            vertex_colors      = submesh["vertex_colors"],
            triangles          = submesh["triangles"],
            bone_weights       = submesh["bone_weights"],
            bone_local_indices = submesh["bone_local_indices"],
            collision          = submesh["collision"],
        )

        # Absolution-only asset-kind word (0x10000 on props/scenery, 0 on characters). Kept on the
        # record so the Mesh Properties panel can preset its toggle and an exporter can write it
        # back; what the bit actually means inside the engine is still unconfirmed.
        record["mesh_flags_raw"] = mesh_flags_raw
        record["object_metadata"]["mesh_flags_raw"] = mesh_flags_raw

        # ----- BoneInfo (needed to resolve boneRemapValues -> global bone indices) -----
        # Mirrors IOI's WeightedMesh/LinkedMesh::Deserialize, which reads ONLY the BoneInfo block from the trailer.
        if has_trailer and bone_information_offset != 0:
            reader.seek(bone_information_offset)
            record["bone_info"] = self.parse_bone_info(reader)
        if has_trailer:
            record["copy_bones_count"]    = copy_bones_count
            record["copy_bones_offset"]   = copy_bones_offset
            record["bone_indices_offset"] = bone_indices_offset

            # ----- boneIndices pool (runtime bone-batch data; captured for export round-trips) -----
            # ABSOLUTION-SPECIFIC encoding (byte-verified on shipping character models - NOT the WoA
            # u32-overlap convention):
            #   u16 elements[0]         = TOTAL element count of the buffer, SELF-INCLUSIVE
            #                             (region ends exactly at offset + 2 * elements[0])
            #   then per BoneInfo accel entry k, laid contiguously:
            #     u16 segmentCount      = accel[k].indices_count + 1 (self-inclusive, at accel offset - 1)
            #     u16 values[count]     = VERTEX indices (verified against per-vertex skinning data)
            #   accel[k].offset is the ELEMENT index of the first value; segments chain with exactly
            #   one count element between them. This is derived GPU batching data - an exporter must
            #   REGENERATE it from the skinning it writes, not copy it.
            if bone_indices_offset != 0:
                reader.seek(bone_indices_offset)
                record["bone_indices_pool"] = self.parse_bone_indices_absolution(reader)

            # ----- copyBones (structure confirmed, semantics open) -----
            if copy_bones_count > 0 and copy_bones_offset != 0:
                reader.seek(copy_bones_offset)
                record["copy_bones"] = self.parse_copy_bones_absolution(reader, copy_bones_count)

        return record

    def parse_bone_indices_absolution(self, reader: Reader) -> list[int]:
        """Parse the Absolution boneIndices pool: u16 total element count (self-inclusive), then the
        remaining u16 elements (per-segment counts interleaved with vertex-index values - see the
        structure comment at the call site). Returned raw so research and future export round-trip
        tooling can inspect it; the importer itself never needs it."""
        total_elements = reader.ushort()
        if total_elements < 1: return [total_elements]
        return [total_elements] + list(reader.read(f"{total_elements - 1}H"))

    def parse_copy_bones_absolution(self, reader: Reader, count: int) -> list[tuple[int, int]]:
        """Parse copyBones: count pairs of u32. Byte-verified structural facts: both values are
        always multiples of 12 in shipping samples, reading like (lengthBytes, offsetBytes) over a
        12 bytes stride target buffer. Which buffer they address is unconfirmed - kept as honest raw
        pairs until an in-engine test settles it."""
        pairs: list[tuple[int, int]] = []
        for _ in range(count):
            first  = reader.uint32()
            second = reader.uint32()
            pairs.append((first, second))
        return pairs

    def parse_prim_submesh_absolution(self, reader: Reader, sub_type: int, is_weighted: bool, position_scale: tuple, position_bias: tuple, uv_scale: tuple, uv_bias: tuple) -> dict:
        """Parse an Absolution PrimSubMesh + its vertex/index buffers (no additionalIndexCount)."""
        # ----- Submesh's own PRIM_OBJECT (44 bytes) -----
        submesh_object = self.parse_prim_object(reader)

        vertex_count         = reader.uint32()
        vertex_buffer_offset = reader.uint32()
        index_count          = reader.uint32()
        index_buffer_offset  = reader.uint32()
        collision_offset     = reader.uint32()
        cloth_offset         = reader.uint32()
        uv_channel_count     = reader.uint32()
        if uv_channel_count == 0: uv_channel_count = 1 # Zero indexed

        print(f"  PrimSubMesh: verts={vertex_count} | indices={index_count} | UV channels={uv_channel_count}")

        # ----- Index buffer -----
        reader.seek(index_buffer_offset)
        triangles = self.parse_index_buffer(reader, index_count)

        # ----- Vertex buffer (Subtype-specific stream layout) -----
        reader.seek(vertex_buffer_offset)
        vertex_data = self.parse_vertex_buffer_absolution(reader, sub_type, vertex_count, uv_channel_count, is_weighted, position_scale, position_bias, uv_scale, uv_bias)

        # ----- Optional collision -----
        collision = None
        if collision_offset != 0:
            reader.seek(collision_offset)
            collision = self.parse_box_coli(reader)

        return {
            "vertex_count":       vertex_count,
            "index_count":        index_count,
            "positions":          vertex_data["positions"],
            "normals":            vertex_data["normals"],
            "tangents":           vertex_data["tangents"],
            "bitangents":         vertex_data["bitangents"],
            "uv_channels":        vertex_data["uv_channels"],
            "vertex_colors":      vertex_data["vertex_colors"],
            "bone_weights":       vertex_data["bone_weights"],
            "bone_local_indices": vertex_data["bone_local_indices"],
            "triangles":          triangles,
            "collision":          collision,
        }

    def parse_vertex_buffer_absolution(self, reader: Reader, sub_type: int, vertex_count: int, uv_channel_count: int, is_weighted: bool, position_scale: tuple, position_bias: tuple, uv_scale: tuple, uv_bias: tuple) -> dict:
        """Decode an Absolution vertex buffer. Stream layout depends on subType.

        STANDARD: Interleaved per vertex; Position, Normal, COLOR, Tangent, Binormal, UV*n
        LINKED  : Planar - All positions then per vertex Normal, Tangent, Binormal, UV*n, COLOR
        WEIGHTED: Planar - All positions then all (4 weights + 4 boneRemap), then per vertex Normal, Tangent, Binormal, UV*n, COLOR"""

        positions:     list[tuple[float, float, float]] = []
        normals:       list[tuple[float, float, float]] = []
        tangents:      list[tuple[float, float, float]] = []
        bitangents:    list[tuple[float, float, float]] = []
        uv_channels:   list[list[tuple[float, float]]]  = [[] for _ in range(uv_channel_count)]
        vertex_colors: list[tuple[float, float, float, float]] = []
        bone_weights:       Optional[list[list[int]]] = None
        bone_local_indices: Optional[list[list[int]]] = None

        def read_position() -> tuple:
            packed = reader.vec4ss()
            return dequantize_position(packed, position_scale, position_bias)

        def read_ntb() -> None:
            n = reader.vec4ub(); t = reader.vec4ub(); b = reader.vec4ub()
            normals.append(convert_vertex_normal(n[0], n[1], n[2]))
            tangents.append(convert_vertex_normal(t[0], t[1], t[2]))
            bitangents.append(convert_vertex_normal(b[0], b[1], b[2]))

        def read_uvs() -> None:
            for ch in range(uv_channel_count):
                uv = dequantize_uv(reader.vec2ss(), uv_scale, uv_bias)
                uv_channels[ch].append(invert_uv_map(uv))

        def read_color() -> None:
            r, g, b, a = reader.vec4ub()
            vertex_colors.append(convert_vertex_color(r, g, b, a))

        if sub_type == PRIM_SUBTYPE_STANDARD: # Interleaved per vertex: position, Normal, COLOR, Tangent, Binormal, UV*n
            for _ in range(vertex_count):
                positions.append(read_position())
                n = reader.vec4ub()
                normals.append(convert_vertex_normal(n[0], n[1], n[2]))
                read_color()
                t = reader.vec4ub(); b = reader.vec4ub()
                tangents.append(convert_vertex_normal(t[0], t[1], t[2]))
                bitangents.append(convert_vertex_normal(b[0], b[1], b[2]))
                read_uvs()
        else: # LINKED / WEIGHTED: planar positions first.
            for _ in range(vertex_count):
                positions.append(read_position())

            if is_weighted: # WEIGHTED inserts the 8 B/vert skinning stream between positions and the NTB block.
                bone_weights = []
                bone_local_indices = []
                for _ in range(vertex_count):
                    raw = reader.read("8B")
                    weight_a, weight_b, weight_c, weight_d = raw[0], raw[1], raw[2], raw[3]
                    if weight_a == 1: # sentinel: zero it, hand the unit to D (keeps sum at 255)
                        weight_a = 0
                        weight_d += 1
                    bone_weights.append([weight_a, weight_b, weight_c, weight_d])  # raw u8; handler /255s
                    bone_local_indices.append([raw[4], raw[5], raw[6], raw[7]])

            # Per vertex: Normal, Tangent, Binormal, UV*n, COLOR
            for _ in range(vertex_count):
                read_ntb()
                read_uvs()
                read_color()

        return {
            "positions":          positions,
            "normals":            normals,
            "tangents":           tangents,
            "bitangents":         bitangents,
            "uv_channels":        uv_channels,
            "vertex_colors":      vertex_colors if vertex_colors else None,
            "bone_weights":       bone_weights,
            "bone_local_indices": bone_local_indices,
        }

    # -----------------------------------------------------------------------------------------------------------------------------------
    # BOND / 007FL PrimMesh
    # The submesh indirection is flattened away - All buffer offsets live directly on the PrimMesh struct (28 B post-PRIM_OBJECT block).
    # -----------------------------------------------------------------------------------------------------------------------------------

    def parse_prim_mesh_bond(self, reader: Reader, is_weighted: bool) -> dict:
        """Parse a Bond PrimMesh (no submesh indirection)."""
        # ----- PRIM_OBJECT (44 bytes) -----
        prim_object = self.parse_prim_object(reader)

        # ----- 7 uints = 28 bytes of flattened submesh fields -----
        vertex_count             = reader.uint32()
        vertex_buffer_offset     = reader.uint32()
        index_count              = reader.uint32()
        unknown_0c               = reader.uint32() # Equivalent of Trilogy's additionalIndexCount
        index_buffer_offset      = reader.uint32()
        auxiliary_stream_offset  = reader.uint32() # Points at BoxColi collision, NOT a per-vertex stream
        cloth_data_offset        = reader.uint32() # Absolute offset of the cloth-simulation data blob; 0 on non-cloth meshes. Pairs with a nonzero clothID.

        # ----- Quantization parameters & cloth ID (Same layout as Trilogy) -----
        vertex_position_scale = reader.vec4f()
        vertex_position_bias  = reader.vec4f()
        uv_coord_scale        = reader.vec2f()
        uv_coord_bias         = reader.vec2f()
        cloth_id_raw          = reader.uint32()

        # ----- Optional weighted trailer (5 fields / 20 bytes) -----
        # Field ORDER differs from Trilogy: boneIndices/boneInfo FIRST, copyBones in the middle.
        bone_indices_offset         = 0
        bone_information_offset     = 0
        copy_bones_count            = 0
        copy_bones_offset           = 0
        per_vertex_skinning_offset  = 0
        if is_weighted:
            bone_indices_offset        = reader.uint32()
            bone_information_offset    = reader.uint32()
            copy_bones_count           = reader.uint32()
            copy_bones_offset          = reader.uint32()
            per_vertex_skinning_offset = reader.uint32()

        print(f"  PrimMesh: subType={prim_object['sub_type_name']} | verts={vertex_count} | indices={index_count} "
              f"| mat={prim_object['material_id']} | LOD={prim_object['lod_index']} (mask=0x{prim_object['lod_mask']:02X}) "
              f"| props=0x{prim_object['properties_raw']:02X} | cloth=0x{cloth_id_raw:08X}")

        # ----- Index buffer (sits BEFORE positions in the file) -----
        triangles: list[tuple[int, int, int]] = []
        if index_count > 0:
            reader.seek(index_buffer_offset)
            triangles = self.parse_index_buffer(reader, index_count)

        # ----- Vertex buffer (positions + attribute stream) -----
        positions:    list[tuple[float, float, float]] = []
        position_w:   list[int] = [] # 007FL: The W component is the fourth bone index
        normals:      list[tuple[float, float, float]] = []
        tangents:     list[tuple[float, float, float]] = []
        bitangents:   list[tuple[float, float, float]] = []
        uv_channels:  list[list[tuple[float, float]]]  = []
        vertex_colors: Optional[list[tuple[float, float, float, float]]] = None
        if vertex_count > 0: # Position stream - int16x4 per vertex, 8 B/vert. XYZ = quantized position; W = 4th bone index.
            reader.seek(vertex_buffer_offset)
            positions, position_w = self.parse_position_stream_bond(reader, vertex_count, vertex_position_scale, vertex_position_bias)

            # The vertex buffer runs from vertex_buffer_offset up to the FIRST block that starts
            # after it. The weighted attribute parser uses this boundary to size the per-vertex
            # attribute region exactly (16- vs 20-byte NTB+UV records), so derive it from every
            # known trailing offset - not just the aux/collision one, which isn't always adjacent.
            trailing = [o for o in (index_buffer_offset, auxiliary_stream_offset, cloth_data_offset, bone_indices_offset, bone_information_offset, copy_bones_offset, per_vertex_skinning_offset)
                        if o > vertex_buffer_offset]
            vertex_buffer_end = min(trailing) if trailing else auxiliary_stream_offset

            # Vertex attribute stream - layout varies by subType. FTell is already at the right place.
            attribute_data = self.parse_vertex_attributes_bond(
                reader            = reader,
                sub_type          = prim_object["sub_type"],
                vertex_count      = vertex_count,
                uv_scale          = uv_coord_scale,
                uv_bias           = uv_coord_bias,
                aux_stream_offset = auxiliary_stream_offset,
                vertex_buffer_end = vertex_buffer_end,
            )
            normals       = attribute_data["normals"]
            tangents      = attribute_data["tangents"]
            bitangents    = attribute_data["bitangents"]
            uv_channels   = attribute_data["uv_channels"]
            vertex_colors = attribute_data["vertex_colors"]

        # ----- BoxColi collision @ auxiliaryStreamOffset -----
        collision = None
        if auxiliary_stream_offset != 0:
            reader.seek(auxiliary_stream_offset)
            collision = self.parse_box_coli(reader)

        # ----- Build the unified mesh record from everything we just parsed -----
        record = self.build_mesh_record(
            prim_object        = prim_object,
            sub_type           = prim_object["sub_type"],
            vertex_count       = vertex_count,
            index_count        = index_count,
            position_scale     = vertex_position_scale,
            position_bias      = vertex_position_bias,
            uv_scale           = uv_coord_scale,
            uv_bias            = uv_coord_bias,
            cloth_id_raw       = cloth_id_raw,
            positions          = positions,
            normals            = normals,
            tangents           = tangents,
            bitangents         = bitangents,
            uv_channels        = uv_channels,
            vertex_colors      = vertex_colors,
            triangles          = triangles,
            bone_weights       = None,
            bone_local_indices = None,
            collision          = collision,
        )

        # Center-of-Rotation stream. Weighted 007FL meshes only - the linked/standard layouts have
        # no such stream - so it rides on the record separately rather than through the shared
        # record builder, which is common to all three games.
        if "centers_of_rotation" in attribute_data:
            record["centers_of_rotation"]      = attribute_data["centers_of_rotation"]
            record["center_of_rotation_lanes"] = attribute_data["center_of_rotation_lanes"]
            record["center_of_rotation_bytes"] = attribute_data["center_of_rotation_bytes"]

        # ----- Weighted-mesh data blocks (Walked in FILE order, NOT trailer-field order) -----
        if is_weighted:
            if bone_information_offset != 0:
                reader.seek(bone_information_offset)
                record["bone_info"] = self.parse_bone_info(reader)

            if bone_indices_offset != 0:
                reader.seek(bone_indices_offset)
                record["bone_palette"] = self.parse_bone_indices_buffer(reader)

            record["copy_bones_count"] = copy_bones_count
            if copy_bones_offset != 0 and copy_bones_count > 0:
                reader.seek(copy_bones_offset)
                record["copy_bones_data"] = bytes(reader.read_bytes(copy_bones_count))

            # Per-vertex skinning - 8 B/vert: 4 u8 weights + a 32-bit word packing three 10-bit
            # bone indices. The 4th bone index comes from the position stream's W lane.
            if per_vertex_skinning_offset != 0 and vertex_count > 0:
                reader.seek(per_vertex_skinning_offset)
                weights, indices = self.parse_per_vertex_skinning_bond(reader, vertex_count, position_w)
                record["bone_weights"]       = weights
                record["bone_local_indices"] = indices

        return record

    # =============================
    # INDEX BUFFER (Shared format)
    # =============================

    def parse_index_buffer(self, reader: Reader, index_count: int) -> list[tuple[int, int, int]]:
        """Parse `index_count` u16 indices and group them into triangles (3 indices each).

        Triangle winding is REVERSED on read - Glacier writes CW triangles; Blender expects CCW."""
        if index_count == 0: return []
        triangle_count = index_count // 3
        triangles: list[tuple[int, int, int]] = []
        for _ in range(triangle_count):
            a, b, c = reader.vec3us()
            triangles.append((a, b, c)) # Flip winding for Blender's CCW convention
        return triangles

    # ============================================
    # TRILOGY VERTEX BUFFER (Single packed stream)
    # ============================================

    def parse_vertex_buffer_trilogy(self, reader: Reader, vertex_count: int, uv_channel_count: int, is_weighted: bool, is_high_resolution: bool, read_color: bool) -> dict:
        """Parse the four sequential sub-streams of a HITMAN_VERTEX_BUFFER.

        Sub-stream order in file:
            1. Positions          - VEC3F (high-res) or VEC4SS quantized
            2. Weights & joints   - HITMAN_WEIGHTS_AND_JOINTS (12 B/vert, weighted only)
            3. NTB + UVs          - VEC4SB*3 + VEC2SS*uvCount (interleaved per vertex)
            4. Vertex colors      - VEC4UB (color-gate dependent)

        Note: this method needs the parent-mesh's quantization scale/bias to decode positions and UVs.
        We get them from the caller's `position_scale/bias` etc. - but the wrapper-level dispatch fills
        those into the mesh record AFTER this parse. To avoid threading them through, we read raw
        ints here and decode them in `build_mesh_record` using the parent's scale/bias.

        However for code clarity we DO decode positions and UVs here using the values we have on hand;
        the wrapper passes them down via a closure pattern (see `parse_prim_mesh_trilogy`)."""

        # NOTE: This method reads RAW quantized values. The Trilogy path needs the parent PrimMesh's quantization vectors to decode them - Those live one stack-frame up.
        # We extract them via the reader's position context: The caller has already navigated the file pointer here, so we just read sequentially.
        # Scale/bias are passed in via a stash on the reader object (added below in `parse_prim_mesh_trilogy`) so this method stays focused on stream parsing.
        position_scale = getattr(reader, "_position_scale", (1.0, 1.0, 1.0, 1.0))
        position_bias  = getattr(reader, "_position_bias",  (0.0, 0.0, 0.0, 0.0))
        uv_scale       = getattr(reader, "_uv_scale",       (1.0, 1.0))
        uv_bias        = getattr(reader, "_uv_bias",        (0.0, 0.0))

        # ----- 1. Positions -----
        positions: list[tuple[float, float, float]] = []
        if is_high_resolution:
            for _ in range(vertex_count): positions.append(reader.vec3f())
        else:
            for _ in range(vertex_count):
                packed = reader.vec4ss()
                positions.append(dequantize_position(packed, position_scale, position_bias))

        # ----- 2. Weights and Indices -----
        # Trilogy encodes up to 6 bone influences per vertex (4 main + 2 supplementary) in 12 bytes/vert.
        bone_weights:       Optional[list[list[float]]] = None
        bone_local_indices: Optional[list[list[int]]] = None
        if is_weighted:
            bone_weights = []
            bone_local_indices = []
            for _ in range(vertex_count): # Layout: w0_A, w0_B, w0_C, w0_D, j0_A, j0_B, j0_C, j0_D, w1_A, w1_B, j1_A, j1_B
                raw = reader.read("12B")
                weights_u8 = (raw[0], raw[1], raw[2], raw[3], raw[8], raw[9])
                joints_u8  = (raw[4], raw[5], raw[6], raw[7], raw[10], raw[11])
                bone_weights.append(list(weights_u8))  # RAW u8 - handler does the single /255
                bone_local_indices.append(list(joints_u8))

        # ----- 3. NTB + UVs (Interleaved per vertex) -----
        normals:     list[tuple[float, float, float]]  = []
        tangents:    list[tuple[float, float, float]]  = []
        bitangents:  list[tuple[float, float, float]]  = []
        uv_channels: list[list[tuple[float, float]]]   = [[] for _ in range(uv_channel_count)]
        for _ in range(vertex_count):
            n = reader.vec4ub()
            t = reader.vec4ub()
            b = reader.vec4ub()
            normals.append(convert_vertex_normal(n[0], n[1], n[2]))
            tangents.append(convert_vertex_normal(t[0], t[1], t[2]))
            bitangents.append(convert_vertex_normal(b[0], b[1], b[2]))

            for ch in range(uv_channel_count):
                packed_uv = reader.vec2ss() # int16x2
                uv = dequantize_uv(packed_uv, uv_scale, uv_bias)
                uv_channels[ch].append(invert_uv_map(uv))

        # ----- 4. Vertex Colors (Only when colour gate passes) -----
        vertex_colors: Optional[list[tuple[float, float, float, float]]] = None
        if read_color:
            vertex_colors = []
            for _ in range(vertex_count):
                r, g, b, a = reader.vec4ub()
                vertex_colors.append(convert_vertex_color(r, g, b, a))

        return {
            "positions":          positions,
            "normals":            normals,
            "tangents":           tangents,
            "bitangents":         bitangents,
            "uv_channels":        uv_channels,
            "vertex_colors":      vertex_colors,
            "bone_weights":       bone_weights,
            "bone_local_indices": bone_local_indices,
        }

    # ======================
    # BOND POSITION STREAM
    # ======================

    def parse_position_stream_bond(self, reader: Reader, vertex_count: int, position_scale: tuple, position_bias: tuple) -> tuple[list[tuple[float, float, float]], list[int]]:
        """Read `vertex_count` int16x4 packed positions; decode XYZ to world space and KEEP W.

        CRITICAL (007FL skinning research): the W lane of each int16x4 position is NOT padding - it is the FOURTH bone index for weighted meshes (signed i16, full bone range, 0 when the vertex has fewer than 4 influences); zero out-of-range indices observed when weight[3] > 0. Non-weighted meshes carry 0 or 32767 here."""

        raw = reader.read_bytes(vertex_count * 8)
        positions:  list[tuple[float, float, float]] = []
        position_w: list[int] = []
        for packed in struct.iter_unpack("<4h", raw):
            positions.append(dequantize_position(packed, position_scale, position_bias))
            position_w.append(packed[3])
        return positions, position_w

    # =====================================================
    # BOND VERTEX ATTRIBUTE DISPATCH (subType-driven)
    #
    # Add new subtype layouts to this dispatch as research confirms them.
    # The interface contract: each parser receives a Reader positioned at the start of the post-position attribute region and the metadata it needs then returns a dict matching the keys below.
    # =====================================================

    def parse_vertex_attributes_bond(self, reader: Reader, sub_type: int, vertex_count: int, uv_scale: tuple, uv_bias: tuple, aux_stream_offset: int, vertex_buffer_end: int = 0) -> dict:
        """Dispatch to the right Bond attribute-stream parser based on subType.

        007FL weighted vertex buffers are PLANAR (stream-major): the position stream is followed
        by separate per-attribute streams, NOT one interleaved record per vertex. The NTB+UV
        record width (16 vs 20 B) is derived directly from the region size, so no heuristic
        detection is needed - the file tells us. See parse_bond_attrs_weighted for the layout."""
        empty = self.empty_attribute_dict()

        if sub_type == PRIM_SUBTYPE_WEIGHTED: return self.parse_bond_attrs_weighted(reader, vertex_count, uv_scale, uv_bias, vertex_buffer_end)
        if sub_type == PRIM_SUBTYPE_LINKED:   return self.parse_bond_attrs_linked(reader, vertex_count, uv_scale, uv_bias)
        if sub_type == PRIM_SUBTYPE_STANDARD: return self.parse_bond_attrs_standard(reader, vertex_count, uv_scale, uv_bias, aux_stream_offset)

        # ----- UNVERIFIED SUBTYPES (STANDARD_UV_2/3/4) -----
        # Falls back to a raw byte skip sized against (aux_stream_offset - current_position) when aux is after the vertex buffer; Otherwise assumes a LINKED-like 16 B/vert layout.
        start = reader.tell()
        fallback_bytes = (aux_stream_offset - start) if aux_stream_offset > start else (vertex_count * 16)
        if fallback_bytes > 0:
            print(f"  WARNING: Unverified Bond subType {sub_type} ({PRIM_SUBTYPE_NAMES.get(sub_type, '?')}). Skipping {fallback_bytes} attribute bytes.")
            reader.read_bytes(fallback_bytes)
        return empty

    def empty_attribute_dict(self) -> dict:
        """Default shape of a parsed-attribute dict, used when a subType is unrecognized."""
        return {
            "normals":       [],
            "tangents":      [],
            "bitangents":    [],
            "uv_channels":   [[]],
            "vertex_colors": None,
        }

    # ---------------------------------------------------------------------------------
    # Bond STANDARD layout - 16 B/vert interleaved NTB+UV then a 4 B/vert color stream
    # ---------------------------------------------------------------------------------

    def parse_bond_attrs_standard(self, reader: Reader, vertex_count: int, uv_scale: tuple, uv_bias: tuple, aux_stream_offset: int = 0) -> dict:
        """STANDARD layout (subType=0): the LINKED 16 B/vert interleaved NTB+UV block, followed
        by a SEPARATE 4 B/vert RGBA color stream (then padding up to the aux stream).

        Cloth props carry the color stream (cloth masks commonly live in vertex color
        channels); plain props may pack exactly 16 B/vert with no color data at all, which is
        why presence is detected from the region size rather than assumed.

        Note: cloth meshes also carry a nonzero `unknown_18` mesh field, which is the absolute
        offset of the cloth-simulation data blob (after the aux/BoxColi block). That blob is
        captured separately by the mesh parser; it does not affect this attribute layout."""

        base = self.parse_bond_attrs_linked(reader, vertex_count, uv_scale, uv_bias)

        # The color stream is OPTIONAL: some STANDARD meshes (e.g. mirror props) pack exactly 16 bytes/vert of attributes with NO trailing color data. Reading colors unconditionally overruns into the next block (or past end-of-file on the last object). Detect presence by the room left between the cursor and the aux stream.
        remaining = aux_stream_offset - reader.tell() if aux_stream_offset else 0
        if remaining >= vertex_count * 4:
            color_bytes = reader.read_bytes(vertex_count * 4)
            base["vertex_colors"] = [convert_vertex_color(r, g, b, a)
                                     for r, g, b, a in struct.iter_unpack("<4B", color_bytes)]
        return base

    # ---------------------------------------------------------------------
    # Bond LINKED layout - 16 B/vert interleaved (Normal + Tangent + Bitangent + UV)
    # ---------------------------------------------------------------------

    def parse_bond_attrs_linked(self, reader: Reader, vertex_count: int, uv_scale: tuple, uv_bias: tuple) -> dict:
        """LINKED layout (subType=1): 16 B/vert interleaved NTB + UV. Confirmed on linked prop samples."""
        raw = reader.read_bytes(vertex_count * 16)
        normals    = []
        tangents   = []
        bitangents = []
        uv_channel = []

        for rec in struct.iter_unpack("<12B2h", raw):
            normals.append(convert_vertex_normal(rec[0], rec[1], rec[2]))
            tangents.append(convert_vertex_normal(rec[4], rec[5], rec[6]))
            bitangents.append(convert_vertex_normal(rec[8], rec[9], rec[10]))
            uv = dequantize_uv((rec[12], rec[13]), uv_scale, uv_bias)
            uv_channel.append(invert_uv_map(uv))

        return {
            "normals":       normals,
            "tangents":      tangents,
            "bitangents":    bitangents,
            "uv_channels":   [uv_channel],
            "vertex_colors": None, # LINKED layout doesn't carry vertex color
        }

    # ------------------------------------------------------------------------------
    # Bond WEIGHTED layout (PLANAR / stream-major) Scalp, sim planes, and haircards
    # ------------------------------------------------------------------------------

    def parse_bond_attrs_weighted(self, reader: Reader, vertex_count: int, uv_scale: tuple, uv_bias: tuple, vertex_buffer_end: int = 0) -> dict:
        """WEIGHTED layout (subType=2). The vertex buffer is PLANAR (stream-major): the position
        stream (parsed earlier) is followed by these contiguous per-vertex streams, in order:

          CoR   : 8 bytes /vert   int16x4 Center of Rotation (see the CENTER OF ROTATION note below)
          NTB+UV: 16 or 20 bytes/vert record stream:
                  +0  Normal     VEC4UB, Decoded (2x/255)-1
                  +4  Tangent    VEC4UB
                  +8  Bitangent  VEC4UB
                  +12 UV0        VEC2SS, Dequantized via uv scale/bias
                  +16 UV1        VEC2SS  -- PRESENT ONLY on the 20-byte record (haircards)
          Color:  4 bytes/vert   VEC4UB RGBA vertex color

        ===== MESH CLASS DISCRIMINATOR (verified on Theresa hair/head + body/prop samples) =====
        The high u16 of clothId classifies the mesh: (clothId >> 16) == 1 for every standard /
        sim-plane / scalp / face mesh (single-UV 16-byte records), == 2 for HAIRCARDS ONLY
        (dual-UV 20-byte records). The record width correlates 1:1 with the class on every
        sample examined, so the class is the authorable switch a from-scratch exporter must set;
        the region-size derivation below stays as a cross-check. The low u16 remains the cloth
        blob index (0 = no cloth data).

        ===== NORMALS ON CARD MESHES (verified byte-level, Theresa samples) =====
        The +0 normal slot is the true shading normal on EVERY mesh class - unit length, and
        signed-correlated 0.92-0.998 with winding-derived geometric normals on lashes / brows.
        Two card-specific facts explain "broken-looking" viewport shading WITHOUT any parse bug:
          * Eyelashes / eyebrows are fully double-sided: exactly half the vertices duplicate a
            position with the OPPOSING normal (front + back sheets, each consistent with its own
            winding). The data is correct; coincident alpha sheets simply z-fight in preview.
          * Haircards store the FRONT-face normal on BOTH sheets (half the vertices oppose their
            own winding) - intentional for the game's two-sided hair shading model. Blender's
            default lighting shows the back sheet dark; the import is faithful.

        ===== CENTER OF ROTATION (the 8-byte stream between positions and NTB) =====
        8 bytes/vert = int16 x4. Lanes 0-2 dequantize with the SAME position scale/bias as the
        position stream and land 2-40 cm from the vertex, on every weighted mesh class (body,
        face, lashes, hair cards). What pins the meaning down:

          * Vertices bound rigidly to one bone share an IDENTICAL point per bone - spread of
            exactly 0.0, not merely close.
          * Vertices that share a bone+weight signature share an identical point too (37 of 40
            multi-vertex signature groups on a Bond body mesh), while vertices blending different
            joints vary smoothly between them.

        That is optimized Center-of-Rotation skinning: each vertex carries the point it should
        rotate about so blended joints don't collapse at shoulders and wrists. Because the value
        is a function of the skin weights and the rest mesh - not extra authored input - it is
        REBUILDABLE on export rather than something we have to preserve from a donor file.

        Lane 3 is a small positive scalar (observed up to ~6700). Every distinct CoR maps to
        exactly one lane-3 value, but one lane-3 value covers many CoRs, so it groups vertices
        more coarsely than the CoR does - consistent with a cluster or batch id. Kept raw; an
        in-engine perturbation test is the cheap way to settle it.

        The record width (16 vs 20) is NOT guessed - it is derived from the actual region size:
            attr_bytes   = (vertex_buffer_end - attribute_start) / vertex_count
            record_bytes = attr_bytes - 8 (CoR) - 4 (color)
        giving 16 for the single-UV layout (scalp, sim planes, brows, lashes) and 20 for the
        dual-UV haircard layout. vertex_buffer_end is the first block offset after the vertex
        buffer (the mesh parser passes the minimum of all trailing offsets).

        Reading this buffer as one interleaved record per vertex scrambles byte columns across
        the three independent streams and corrupts normals/UVs/colors - stream-major is the only
        correct model. Verified: unit-length, correctly oriented normals and in-range UVs on 100%
        of non-degenerate verts across scalp, proxy-plane, and hair-card meshes (cursor lands
        exactly at the vertex-buffer end on every object of full hair files)."""
        start = reader.tell()

        # Derive the record width from the region size (Deterministic; no heuristics).
        record_bytes     = 16
        uv_channel_count = 1
        if vertex_buffer_end > start and vertex_count:
            attr_bytes = (vertex_buffer_end - start) // vertex_count
            if attr_bytes >= 8 + 20 + 4: # CoR(8) + 20-byte record + color(4)
                record_bytes     = 20
                uv_channel_count = 2

        # ----- Center of Rotation: 8 bytes/vert, int16x4 -----
        # Lanes 0-2 are the CoR position quantized against the POSITION scale/bias (not the UV
        # one); lane 3 is a coarse grouping scalar. Decoded here so the data is usable rather than
        # opaque, and kept raw alongside it for byte-exact round trips.
        center_of_rotation_bytes = bytes(reader.read_bytes(vertex_count * 8))
        centers_of_rotation: list[tuple[int, int, int]] = []
        center_of_rotation_lanes: list[int] = []
        for cx, cy, cz, lane in struct.iter_unpack("<4h", center_of_rotation_bytes):
            centers_of_rotation.append((cx, cy, cz))
            center_of_rotation_lanes.append(lane)

        # ----- NTB + UV record stream (16 or 20 B/vert) -----
        raw = reader.read_bytes(vertex_count * record_bytes)
        fmt = "<12B2h" if record_bytes == 16 else "<12B4h"
        normals    = []
        tangents   = []
        bitangents = []
        uv0        = []
        uv1        = []
        for rec in struct.iter_unpack(fmt, raw):
            normals.append(convert_vertex_normal(rec[0], rec[1], rec[2]))
            tangents.append(convert_vertex_normal(rec[4], rec[5], rec[6]))
            bitangents.append(convert_vertex_normal(rec[8], rec[9], rec[10]))
            uv0.append(invert_uv_map(dequantize_uv((rec[12], rec[13]), uv_scale, uv_bias)))
            if uv_channel_count == 2: uv1.append(invert_uv_map(dequantize_uv((rec[14], rec[15]), uv_scale, uv_bias)))

        uv_channels = [uv0, uv1] if uv_channel_count == 2 else [uv0]

        # ----- Color: 4 B/vert VEC4UB RGBA -----
        color_bytes = reader.read_bytes(vertex_count * 4)
        vertex_colors = [convert_vertex_color(r, g, b, a)
                         for r, g, b, a in struct.iter_unpack("<4B", color_bytes)]

        return {
            "normals":             normals,
            "tangents":            tangents,
            "bitangents":          bitangents,
            "uv_channels":         uv_channels,
            "vertex_colors":       vertex_colors,
            # Quantized ints, not world units: dequantize with the mesh's position scale/bias to
            # place them. Raw bytes ride along so an exporter can round-trip byte-exactly.
            "centers_of_rotation":      centers_of_rotation,
            "center_of_rotation_lanes": center_of_rotation_lanes,
            "center_of_rotation_bytes": center_of_rotation_bytes,
        }

    # ===================================================================
    # BOND PER-VERTEX SKINNING (8 bytes/vert at perVertexSkinningOffset)
    # ===================================================================

    def parse_per_vertex_skinning_bond(self, reader: Reader, vertex_count: int, position_w: list[int]) -> tuple[list[list[int]], list[list[int]]]:
        """Read and DECODE 007FL skinning records.

        ===== 007: FIRST LIGHT SKINNING FORMAT (reverse-engineered, fully verified) =====
        Each vertex has 4 bone influences. Per-vertex skinning record is 8 bytes:

            ubyte weight[4] : influence weights, sum to 255
            uint packedBones: little-endian word packing THREE 10-bit bone indices:
                                 bone0 = packedBones         & 0x3FF (pairs with weight[0])
                                 bone1 = (packedBones >> 10) & 0x3FF (pairs with weight[1])
                                 bone2 = (packedBones >> 20) & 0x3FF (pairs with weight[2])
                                 bits 30-31 unused

        The FOURTH bone index doesn't live in this stream - it is the W lane of the int16x4
        quantized position (signed i16, full bone range), pairing with weight[3]. When a vertex
        has fewer than 4 influences, the unused weight bytes are 0 and the unused bone fields
        read as 0 (GROUND) - they are skipped because their weight is 0.

        WHY 10 BITS: 007FL character rigs exceed 256 bones (e.g. 560-bone body rigs with face
        bones at indices 236-460), so u8 joints are impossible. 10 bits addresses 1024 bones.
        This packing is also why earlier byte-wise interpretations looked "almost right":
        byte 0 equals bone0 whenever bone0 < 256 and bone1's low bits are clear, so simple props
        decoded fine while faces/limbs (bones >= 256 or dense blends) shattered.

        VERIFICATION: across dedicated-rig and shared-rig shipping assets, decoded blends form
        correct anatomical chains (adjacent spine/jaw/brow bones) with exact left/right symmetry
        and ZERO out-of-range indices over 305k+ influences.

        Weights are returned as RAW u8 (the handler does the single /255).
        Bone indices are returned FULLY DECODED - downstream code uses them directly."""
        raw = reader.read_bytes(vertex_count * 8)
        bone_weights:       list[list[int]] = []
        bone_local_indices: list[list[int]] = []
        w_count = len(position_w)

        for vertex_index, (w0, w1, w2, w3, packed_bones) in enumerate(struct.iter_unpack("<4BI", raw)):
            bone_weights.append([w0, w1, w2, w3])
            bone_local_indices.append([
                packed_bones         & 0x3FF,
                (packed_bones >> 10) & 0x3FF,
                (packed_bones >> 20) & 0x3FF,
                position_w[vertex_index] if vertex_index < w_count else 0,
            ])

        return bone_weights, bone_local_indices

    # ===========================================
    # BoneInfo (Shared between Trilogy and Bond)
    # ===========================================

    def parse_bone_info(self, reader: Reader) -> dict:
        """Parse a BONE_INFO record: u16 totalSize + u16 accelCount + 255 B remap + 1 B pad + accel entries.

        `totalSize` is the TOTAL allocated region size in bytes (including trailing reserved 0xFF padding).
        We consume the tail so subsequent FSeek lands at the right place."""
        start_offset       = reader.tell()
        total_size         = reader.ushort()
        accel_entry_count  = reader.ushort()
        bone_remap         = list(reader.read_bytes(255))
        reader.skip(1) # alignmentPadding

        # 0xFF entries are unused slots in the bone remap table.
        accel_entries: list[dict] = []
        for _ in range(accel_entry_count):
            offset_val    = reader.uint32()
            indices_count = reader.uint32()
            accel_entries.append({"offset": offset_val, "indices_count": indices_count})

        # Consume any trailing 0xFF padding to leave the cursor at the end of the totalSize region.
        parsed_bytes = reader.tell() - start_offset
        tail_bytes   = total_size - parsed_bytes
        if tail_bytes > 0: reader.skip(tail_bytes)

        return {
            "total_size":        total_size,
            "accel_entry_count": accel_entry_count,
            "bone_remap":        bone_remap,    # 255-entry: local palette idx -> global bone ID (0xFF = unused)
            "accel_entries":     accel_entries, # per-entry: (offset into bone_palette, count of indices to consume)
        }

    # =====================================================================
    # BoneIndices buffer (Shared; Count overlaps with the first 2 indices)
    # =====================================================================

    def parse_bone_indices_buffer(self, reader: Reader) -> list[int]:
        """Parse the count-prefixed u16 bone-indices buffer.

        IOI's space-saving trick: the u32 count and the first 2 u16 indices OVERLAP -
        the bytes that encode `indexCount` ALSO act as indices[0] (low u16) and
        indices[1] (high u16). Then (indexCount - 2) more u16s follow.

        Edge cases:
          - indexCount == 0: empty buffer.
          - indexCount == 1: only indices[0] is valid (= count & 0xFFFF); no follow-up u16s.
          - indexCount == 2: both indices live inside the count word; no follow-up u16s.
          - indexCount >= 3: read (indexCount - 2) more u16s."""

        index_count = reader.uint32()
        if index_count == 0: return []

        indices: list[int] = []
        if index_count >= 1: indices.append(index_count & 0xFFFF)
        if index_count >= 2: indices.append((index_count >> 16) & 0xFFFF)

        remaining = index_count - 2
        if remaining > 0:
            for _ in range(remaining): indices.append(reader.ushort())

        return indices

    # ====================================================
    # BoxColi - bullet / projectile broad-phase collision
    # ====================================================

    def parse_box_coli(self, reader: Reader) -> dict:
        """Parse a BOX_COLI block (broad-phase collision boxes quantized into mesh-bbox space)."""
        chunk_count          = reader.ushort()
        triangles_per_chunk  = reader.ushort()

        entries: list[dict] = []
        for _ in range(chunk_count):
            box_min = reader.vec3ub() # Quantized ubyte*3 in mesh-bbox space
            box_max = reader.vec3ub()
            entries.append({"box_min": box_min, "box_max": box_max})

        return {
            "chunk_count":         chunk_count,
            "triangles_per_chunk": triangles_per_chunk,
            "entries":             entries,
        }

    # =====================================================
    # MESH RECORD BUILDER (Unified output shape)
    # =====================================================

    def build_mesh_record(self, prim_object: dict, sub_type: int, vertex_count: int, index_count: int, position_scale: tuple, position_bias: tuple, uv_scale: tuple, uv_bias: tuple, cloth_id_raw: int, positions: list, normals: list, tangents: list, bitangents: list, uv_channels: list, vertex_colors: Optional[list], triangles: list, bone_weights: Optional[list], bone_local_indices: Optional[list], collision: Optional[dict]) -> dict:
        """Pack one mesh's worth of decoded data into the unified per-object record dict.

        This is the shape model_handler.import_prim_model() (and any future exporter) expect.
        Trilogy and Bond paths both funnel through here so downstream code is game-agnostic."""
        return {
            "object_metadata":    prim_object,
            "sub_type":           sub_type,
            "sub_type_name":      PRIM_SUBTYPE_NAMES.get(sub_type, f"UNKNOWN_{sub_type}"),
            # Surface LOD info at the top level so the model handler can name meshes without reaching into object_metadata (where it lives but the handler doesn't look).
            "lod_mask":           prim_object.get("lod_mask", 0),
            "lod_index":          prim_object.get("lod_index", 0),
            "material_id":        prim_object.get("material_id", 0),
            "vertex_count":       vertex_count,
            "index_count":        index_count,
            "scale_bias": {
                "position_scale": position_scale,
                "position_bias":  position_bias,
                "uv_scale":       uv_scale,
                "uv_bias":        uv_bias,
            },
            "cloth_id_raw":       cloth_id_raw,
            "positions":          positions,
            "normals":            normals,
            "tangents":           tangents,
            "bitangents":         bitangents,
            "uv_channels":        uv_channels,
            "vertex_colors":      vertex_colors,
            "triangles":          triangles,
            "bone_weights":       bone_weights,
            "bone_local_indices": bone_local_indices,
            "bone_info":          None,  # Populated by the weighted-trailer walk if applicable
            "bone_palette":       None,  # Populated by the weighted-trailer walk if applicable
            "copy_bones_count":   0,
            "copy_bones_data":    None,
            "collision":          collision,
        }

# =====================================================================================================================================================
# TRILOGY VERTEX-BUFFER QUANTIZATION CONTEXT HELPER
# The Hitman vertex buffer reader needs the parent mesh's quantization scale/bias
# vectors to decode positions and UVs. Rather than thread them through several
# function signatures, we stash them on the Reader instance just before diving in.
# This keeps the parser methods focused on stream layout, not parameter forwarding.
# =====================================================================================================================================================

def stash_quantization_on_reader(reader: Reader, position_scale, position_bias, uv_scale, uv_bias) -> None:
    """Attach per-mesh quantization vectors to the reader for downstream decoders to read."""
    reader._position_scale = position_scale
    reader._position_bias  = position_bias
    reader._uv_scale       = uv_scale
    reader._uv_bias        = uv_bias

# Patch PRIM.parse_prim_submesh_trilogy to call stash_quantization_on_reader before reading the vertex buffer. We do this here (post-class definition) by overriding the method to wrap it - Keeps the original method readable above without a verbose preamble. The stash is read back inside parse_vertex_buffer_trilogy via getattr() with safe defaults.
_original_parse_submesh_trilogy = PRIM.parse_prim_submesh_trilogy
def parse_prim_submesh_trilogy_wrapped(self: PRIM, reader: Reader, is_weighted: bool, parent_object: dict) -> dict:
    # The parent PrimMesh has already read its quantization vectors. We need to expose them to the submesh's vertex-buffer parser. Pull them from the caller's frame would be hacky so we re-fetch them by repositioning: the caller saved its position before recursing into the submesh, but we don't have easy access here. Instead the calling site
    # parse_prim_mesh_trilogy stashes them on the reader BEFORE calling us.
    return _original_parse_submesh_trilogy(self, reader, is_weighted, parent_object)
PRIM.parse_prim_submesh_trilogy = parse_prim_submesh_trilogy_wrapped

# Re-wire parse_prim_mesh_trilogy so it stashes quantization vectors on the reader before diving into the submesh. We override the method by inserting the stash call at the right point. Cleanest way: Replace the method with a version that does the stash.
_original_parse_mesh_trilogy = PRIM.parse_prim_mesh_trilogy
def parse_prim_mesh_trilogy_wrapped(self: PRIM, reader: Reader, is_weighted: bool) -> dict:
    # We need to capture the quantization vectors AFTER they're read but BEFORE the submesh parser descends into the vertex buffer. The original method reads them mid-flow; the cleanest mirror is to duplicate the read sequence here, stash, then dispatch into the rest of the workflow by calling the submesh parser directly with the values we just read.
    # Rather than duplicate code, we do a focused re-implementation:

    prim_object = self.parse_prim_object(reader)
    print(f"  PrimMesh: subType={prim_object['sub_type_name']} | materialID={prim_object['material_id']}")

    sub_mesh_table_offset = reader.uint32()
    vertex_position_scale = reader.vec4f()
    vertex_position_bias  = reader.vec4f()
    uv_coord_scale        = reader.vec2f()
    uv_coord_bias         = reader.vec2f()
    cloth_id_raw          = reader.uint32()

    copy_bones_count        = 0
    copy_bones_offset       = 0
    bone_indices_offset     = 0
    bone_information_offset = 0
    if is_weighted:
        copy_bones_count        = reader.uint32()
        copy_bones_offset       = reader.uint32()
        bone_indices_offset     = reader.uint32()
        bone_information_offset = reader.uint32()

    # Stash quantization vectors so the inner submesh + vertex-buffer parsers can find them.
    stash_quantization_on_reader(reader, vertex_position_scale, vertex_position_bias, uv_coord_scale, uv_coord_bias)

    saved_position = reader.tell()
    reader.seek(sub_mesh_table_offset)
    sub_mesh_offset = reader.uint32()
    reader.seek(sub_mesh_offset)

    submesh = self.parse_prim_submesh_trilogy(reader, is_weighted, prim_object)
    reader.seek(saved_position)

    record = self.build_mesh_record(
        prim_object         = prim_object,
        sub_type            = prim_object["sub_type"],
        vertex_count        = submesh["vertex_count"],
        index_count         = submesh["index_count"],
        position_scale      = vertex_position_scale,
        position_bias       = vertex_position_bias,
        uv_scale            = uv_coord_scale,
        uv_bias             = uv_coord_bias,
        cloth_id_raw        = cloth_id_raw,
        positions           = submesh["positions"],
        normals             = submesh["normals"],
        tangents            = submesh["tangents"],
        bitangents          = submesh["bitangents"],
        uv_channels         = submesh["uv_channels"],
        vertex_colors       = submesh["vertex_colors"],
        triangles           = submesh["triangles"],
        bone_weights        = submesh["bone_weights"],
        bone_local_indices  = submesh["bone_local_indices"],
        collision           = submesh["collision"],
    )

    if is_weighted:
        if bone_information_offset != 0:
            reader.seek(bone_information_offset)
            record["bone_info"] = self.parse_bone_info(reader)
        if bone_indices_offset != 0:
            reader.seek(bone_indices_offset)
            record["bone_palette"] = self.parse_bone_indices_buffer(reader)
        record["copy_bones_count"] = copy_bones_count
        if copy_bones_offset != 0 and copy_bones_count > 0:
            reader.seek(copy_bones_offset)
            record["copy_bones_data"] = bytes(reader.read_bytes(copy_bones_count))

    return record

PRIM.parse_prim_mesh_trilogy = parse_prim_mesh_trilogy_wrapped

# =====================================================================================================================================================
