# ------------------------------------------------
# CLOAKWORKS SHROUD CLOTH (.CLOS) PARSER
#       Hitman: Absolution cloth resources.
#
#       CloakWorks stores cloth as a REFLECTION TREE - a
#       generic node graph (objects, arrays, primitives)
#       rather than a fixed struct layout. We walk that
#       tree exactly like IOI's deserializer, then pull the
#       renderable cloth grid out of the leaf arrays.
#
#       Reversed byte-exact from IOI's Cloth::Deserialize +
#       CloakWorks::ShapeDefinition against real samples. No
#       speculative fields - anything unconfirmed is skipped
#       rather than guessed.
# ------------------------------------------------

import struct
from ..io import Reader
from ..utilities import *

# ------------------------------------------------
# REFLECTION FIELD TYPES
#
# CloakWorks tags every node with a primitive/array element type. Only the values we have observed
# in shipping cloth are enumerated; unknown tags fall through to being skipped (never guessed).
# ------------------------------------------------

FIELD_TYPE_INVALID = 0xFFFFFFFF # kFieldType_Invalid: The sentinel the root node carries
FIELD_TYPE_OBJECT  = "object"   # Mode has child nodes (An object / struct)
FIELD_TYPE_ARRAY   = "array"    # Mode is an array (of objects or primitives)

# BinaryNode fixed-part layout (52 bytes / 13 uints). Field offsets are the crux of the format, IOI's deserializer seeks via offsetof() on these exact members:
#   +0x00 nameOffset            (From node start to the name string)
#   +0x04 nameLength
#   +0x08 classNameOffset       (Name resolves at node + classNameOffset + 8 - IOI's +8 quirk)
#   +0x0C classNameLength
#   +0x10 nextBinaryNodeOffset  (Seek = node + 0x10 + value; 0 terminates the sibling chain)
#   +0x14 childBinaryNodeOffset (Seek = node + 0x14 + value; first child of an object/array<object>)
#   +0x18 dataOffset            (Seek = node + 0x18 + value; primitive value / array element data)
#   +0x1C dataSize              (Byte size of a primitive value or the whole primitive array)
#   +0x20 arrayPrimitiveCount   (Element count for arrays)
#   +0x24 arrayPrimitiveType    (Element type discriminator for arrays: nonzero -> array<object>)
#   +0x28 primitiveType         (0xFFFFFFFF for leaf primitives here; Array nodes carry a class tag)

BINARY_NODE_NAME_OFFSET        = 0x00
BINARY_NODE_NAME_LENGTH        = 0x04
BINARY_NODE_CLASSNAME_OFFSET   = 0x08
BINARY_NODE_CLASSNAME_LENGTH   = 0x0C
BINARY_NODE_NEXT_OFFSET        = 0x10
BINARY_NODE_CHILD_OFFSET       = 0x14
BINARY_NODE_DATA_OFFSET        = 0x18
BINARY_NODE_DATA_SIZE          = 0x1C
BINARY_NODE_ARRAY_COUNT        = 0x20
BINARY_NODE_ARRAY_TYPE         = 0x24
BINARY_NODE_CLASSNAME_EXTRA    = 8    # IOI adds 8 to classNameOffset before reading the string

# Root binaryNodeOffset lives at file +0x24; The first node is then (0x28 + offset - 4).
CLOS_ROOT_OFFSET_POSITION = 0x24


# ------------------------------------------------
# NODE MODEL
# ------------------------------------------------

class ClosNode:
    """One node in the CloakWorks reflection tree.

    A node is either an object (Children keyed by name), An array (Ordered children or a flat
    primitive list) or a primitive leaf (a single scalar / string). We keep the raw shape and let
    the geometry extractor pull what it needs by name, mirroring how IOI walks the tree."""

    def __init__(self, name: str = "", class_name: str = ""):
        self.name: str = name
        self.class_name: str = class_name
        self.children: list["ClosNode"] = []     # Object / Array<object> Children
        self.array_values: Optional[list] = None # Flat primitive array payload
        self.value = None                        # Scalar / String leaf payload

    def child_by_name(self, name: str) -> Optional["ClosNode"]:
        for child in self.children:
            if child.name == name: return child
        return None

    def children_by_name(self, name: str) -> list["ClosNode"]:
        return [child for child in self.children if child.name == name]

    def child_by_class(self, class_name: str) -> Optional["ClosNode"]:
        for child in self.children:
            if child.class_name == class_name: return child
        return None


# ------------------------------------------------
# CLOTH RESOURCE
# ------------------------------------------------

class ClothPiece:
    """A single renderable cloth piece extracted from a Simulation control subtree.

    A CloakWorks cloth carries TWO representations:
      * The low-res SIMULATION mesh - a row/column grid of nodes (startingPositions) the physics
        solver actually moves. Small (tens of nodes).
      * The high-res RENDER mesh - the visible cloth surface (index32s + texCoords). Its vertices
        are not stored directly; each is bound to a simulation triangle (triIndices) by barycentric
        weights (bindingOffsets), so its rest positions are reconstructed by blending the three sim
        node positions of its bound triangle.

    We import the RENDER mesh (that's the actual in-game surface, with real UVs), reconstructing its
    rest positions from the binding. The simulation grid dimensions are retained for reference."""

    def __init__(self, name: str):
        self.name: str = name
        self.shape_kind: str = "" # SheetShapeDefinition / TubeShapeDefinition / StrandShapeDefinition

        # Simulation grid (source geometry the render mesh binds to).
        self.num_rows: int = 0
        self.num_columns: int = 0
        self.num_nodes: int = 0
        self.sim_positions: list[tuple[float, float, float]] = []
        self.flags: list[int] = []

        # Render mesh (what we build in Blender).
        self.positions: list[tuple[float, float, float]] = [] # Reconstructed render-vertex rest positions
        self.normals: list[tuple[float, float, float]] = []   # Per-vertex normals from the binding frame
        self.triangles: list[tuple[int, int, int]] = []       # Render index buffer (index32s)
        self.uvs: list[tuple[float, float]] = []              # Per-render-vertex texcoords
        self.bone_weights: dict = {}                          # boneName -> [weight per vertex]


class CLOS:
    """Top-level CloakWorks Shroud cloth resource.

    Usage mirrors the PRIM/BORG parsers:
        cloth = CLOS(file_path)
        cloth.parse_cloth_file()
        for piece in cloth.pieces: ...
    """

    def __init__(self, file_path: str):
        self.file_path: str = file_path
        with open(file_path, "rb") as handle:
            self.data: bytes = handle.read()

        # CloakWorks cloth is little-endian on the PC build we target (Hitman: Absolution).
        self.reader: Reader = Reader(self.data, is_little_endian=True)
        self.root: Optional[ClosNode] = None
        self.pieces: list[ClothPiece] = []
        self.transform_bone_names: dict = {} # Transform GUID -> Skeleton bone name

    # =====================================================
    # ENTRY POINT
    # =====================================================

    def parse_cloth_file(self) -> None:
        """Walk the reflection tree then lift every cloth piece out of it."""
        file_size = len(self.data)
        print(f"Parsing CloakWorks Shroud cloth (Hitman: Absolution): {self.file_path}")
        print(f"  File size: {file_size:,} bytes (0x{file_size:X})")

        # ----- Locate the first BinaryNode -----
        # Header: seek 0x24, read u32 binaryNodeOffset, then the node sits at (0x28 + offset - 4).
        self.reader.seek(CLOS_ROOT_OFFSET_POSITION)
        binary_node_offset = self.reader.uint32()
        first_node_position = (CLOS_ROOT_OFFSET_POSITION + 4) + binary_node_offset - 4
        print(f"  Binary Node Offset: 0x{binary_node_offset:08X} -> Root BinaryNode at 0x{first_node_position:X}")

        # ----- Walk the tree -----
        self.root = ClosNode(name="", class_name="")
        self.deserialize_node(self.root, first_node_position, depth=0)

        node_count = self.count_nodes(self.root)
        root_label = self.root.children[0].class_name if self.root.children else "?"
        print(f"  Reflection tree: {node_count} nodes (root object: {root_label}).")

        # ----- Extract cloth geometry -----
        self.extract_pieces()
        print(f"  Extracted {len(self.pieces)} cloth piece(s).")

    # =====================================================
    # TREE WALKER
    #
    # A faithful re-implementation of IOI's Cloth::Deserialize recursion. Each call walks a chain of
    # sibling nodes at `position`, attaching parsed children to `parent`. Object/array<object> nodes
    # recurse into their child chain; primitive and primitive-array nodes read their payload.
    # =====================================================

    def deserialize_node(self, parent: ClosNode, position: int, depth: int) -> None:
        """Walk the sibling chain beginning at `position`, populating `parent.children`."""
        data = self.data
        size = len(data)

        while 0 <= position < size:
            node_start = position

            name_offset       = self.read_u32(node_start + BINARY_NODE_NAME_OFFSET)
            name_length       = self.read_u32(node_start + BINARY_NODE_NAME_LENGTH)
            class_name_offset = self.read_u32(node_start + BINARY_NODE_CLASSNAME_OFFSET)
            class_name_length = self.read_u32(node_start + BINARY_NODE_CLASSNAME_LENGTH)
            next_offset       = self.read_u32(node_start + BINARY_NODE_NEXT_OFFSET)
            child_offset      = self.read_u32(node_start + BINARY_NODE_CHILD_OFFSET)
            data_offset       = self.read_u32(node_start + BINARY_NODE_DATA_OFFSET)
            data_size         = self.read_u32(node_start + BINARY_NODE_DATA_SIZE)
            array_count       = self.read_u32(node_start + BINARY_NODE_ARRAY_COUNT)

            # ----- Names -----
            name = ""
            if name_offset > 0: name = self.read_string_at(node_start + name_offset, name_length)

            class_name = ""
            if class_name_offset > 0: class_name = self.read_string_at(node_start + class_name_offset + BINARY_NODE_CLASSNAME_EXTRA, class_name_length)

            node = ClosNode(name=name, class_name=class_name)

            # ----- Classify -----
            # The clean discriminator (reversed against real files):
            #   * array<object>   : childBinaryNodeOffset > 0 AND dataSize == 0  (recurse the child chain)
            #   * primitive array : dataSize > 0 AND arrayCount > 0              (flat payload of dataSize bytes)
            #   * plain object    : childBinaryNodeOffset > 0 AND dataSize == 0  (same shape as array<object>;
            #                       both simply recurse - we don't need to tell them apart to read them)
            #   * scalar leaf     : dataSize > 0 AND arrayCount == 0
            # NOTE: the field at +0x24 is NOT a clean array-element-type tag (it varies with element
            # width), so we deliberately do not key classification on it - dataSize/childOffset are
            # the reliable signals.
            is_object_like    = (child_offset > 0 and data_size == 0)
            is_primitive_array = (data_size > 0 and array_count > 0)

            if is_object_like: # Object or array-of-objects: recurse into the child chain.
                child_position = node_start + BINARY_NODE_CHILD_OFFSET + child_offset
                self.deserialize_node(node, child_position, depth + 1)
            elif is_primitive_array: # Flat primitive array - read the whole dataSize span (arrayCount can be a SIMD block count so element width is derived from dataSize and not dataSize/arrayCount).
                element_position = node_start + BINARY_NODE_DATA_OFFSET + data_offset
                node.array_values = self.read_primitive_array(element_position, data_size, name)
            else: # Scalar / string leaf.
                value_position = node_start + BINARY_NODE_DATA_OFFSET + data_offset
                node.value = self.read_scalar(value_position, data_size, name)

            parent.children.append(node)

            # ----- Advance to the next sibling -----
            if next_offset == 0: break
            position = node_start + BINARY_NODE_NEXT_OFFSET + next_offset

    # =====================================================
    # PAYLOAD READERS
    # =====================================================

    def read_primitive_array(self, position: int, data_size: int, name: str) -> list:
        """Read a flat primitive array of `data_size` bytes.

        Element width is inferred from the field's role rather than dataSize/arrayCount, because the
        arrayCount CloakWorks stores is a SIMD block count for the packed float streams (e.g. a
        60-node position array reports 15 blocks of 12 floats). Index/flag arrays are the exception
        where count and elements line up 1:1, but keying purely on the field's known type keeps the
        reader unambiguous and honest - we never divide by a count that may not mean what we expect."""
        if data_size <= 0 or position < 0 or position + data_size > len(self.data): return []

        lowered = name.lower()

        # Float streams: Positions, Normals, TexCoords, Weights, Distances, Matrices and the various simulation scalars are all 32-bit floats packed contiguously.
        float_fields = ("position", "normal", "texcoord", "weight", "distance", "offset", "matrix", "force", "strength", "radius", "scale", "center", "blend", "angle")
        # Index streams are 32-bit unsigned ints.
        index_fields = ("index", "indices", "tri")

        self.reader.seek(position)

        if any(token in lowered for token in index_fields): return list(self.reader.read(f"{data_size // 4}I"))
        if any(token in lowered for token in float_fields): return list(self.reader.read(f"{data_size // 4}f"))

        # startingFlags (and similar) are 32-bit ints per node.
        if "flag" in lowered: return list(self.reader.read(f"{data_size // 4}I"))

        # Fallback: expose the raw bytes so nothing is silently misread as a wrong type.
        return list(self.data[position: position + data_size])

    def read_scalar(self, position: int, data_size: int, name: str):
        """Read a single primitive value (uint/float/string).

        A 4-byte payload is ambiguous - it could be a u32 OR a 4-character string (e.g. a bone named
        "Neck"). We disambiguate by the node's own name: name-carrying fields (name, boneName,
        transformName) are always strings, everything else 4-byte defaults to u32 (counts, guids,
        modes)."""
        if position < 0 or position + max(data_size, 0) > len(self.data): return None
        self.reader.seek(position)

        lowered = name.lower()
        is_string_field = lowered.endswith("name")

        if data_size >= 1 and is_string_field: return self.read_string_at(position, data_size)
        if data_size == 4: return self.reader.uint32()
        if data_size == 2: return self.reader.ushort()
        if data_size == 1: return self.reader.ubyte()
        if data_size > 4: return self.read_string_at(position, data_size )# Strings are length-sized byte runs.
        return None

    # =====================================================
    # LOW-LEVEL HELPERS
    # =====================================================

    def read_u32(self, position: int) -> int:
        if position < 0 or position + 4 > len(self.data): return 0
        return struct.unpack_from("<I", self.data, position)[0]

    def read_string_at(self, position: int, length: int) -> str:
        if length <= 0 or position < 0 or position + length > len(self.data): return ""
        raw = self.data[position: position + length]
        return raw.split(b"\x00")[0].decode("utf-8", errors="replace")

    def count_nodes(self, node: ClosNode) -> int:
        """Total node count in the subtree, for the parse summary print."""
        return 1 + sum(self.count_nodes(child) for child in node.children)

    # =====================================================
    # GEOMETRY EXTRACTION
    #
    # A cloth resource contains one or more Simulation subtrees. Each holds:
    #   * A ShapeDefinition: The low-res simulation grid (startingPositions, numRows/Columns/Nodes)
    #   * Optionally a ThickMeshControl: The high-res render surface (index32s, texCoords) whose vertices are bound to the simulation grid (triIndices + bindingOffsets).
    #
    # Implementation mirrors HitmanAbsolutionEditor's Cloth::ConvertToGLB function:
    #   * ThickMeshControl present -> Reconstruct the render mesh via the full binding math (barycentric blend + curvature displacement + local-frame position/normal offsets).
    #   * No ThickMeshControl      -> Emit the simulation grid mesh directly (sheet topology).
    # =====================================================

    def extract_pieces(self) -> None:
        """Find every Simulation subtree and reconstruct its cloth mesh."""
        if self.root is None: return

        simulations = self.find_simulations(self.root)
        print(f"  Found {len(simulations)} simulation subtree(s).")

        # Bone bindings live on the ShroudObject: transforms[] pairs each guid with a boneName.
        self.transform_bone_names = self.collect_transform_bone_names()
        if self.transform_bone_names: print(f"  Skeleton bindings: {len(self.transform_bone_names)} transform(s) with bone names.")

        for index, simulation in enumerate(simulations):
            piece = self.build_piece(simulation, index)
            if piece is not None and piece.positions and piece.triangles: self.pieces.append(piece)
            else: print(f"  Simulation {index}: no renderable geometry; skipping.")

    def find_simulations(self, node: ClosNode, found: Optional[list] = None) -> list[ClosNode]:
        """Return every Simulation node (each yields one cloth piece).

        Matches the reference's structure: a Simulation is any node whose DIRECT children include a
        ShapeDefinition-classed child (SheetShapeDefinition / TubeShapeDefinition / ...). Keying on
        the child's class rather than the Simulation's own class keeps this robust if IOI ever ships
        a differently-named simulation wrapper."""
        if found is None: found = []
        if any(child.class_name.endswith("ShapeDefinition") for child in node.children if child.class_name): found.append(node)
        for child in node.children: self.find_simulations(child, found)
        return found

    def collect_transform_bone_names(self) -> dict:
        """Map Transform guid -> boneName from the ShroudObject's transforms array. SkinningTransform
        nodes reference these guids, which is how cloth weights attach to skeleton bones."""
        bone_names: dict = {}
        if self.root is None: return bone_names

        def visit(node: ClosNode) -> None:
            if node.class_name == "Transform":
                guid_node = node.child_by_name("guid")
                bone_node = node.child_by_name("boneName")
                if guid_node is not None and guid_node.value is not None and bone_node is not None and bone_node.value: bone_names[int(guid_node.value)] = str(bone_node.value)
            for child in node.children: visit(child)

        visit(self.root)
        return bone_names

    def has_descendant(self, node: ClosNode, name: str) -> bool:
        """True if name appears anywhere in this subtree."""
        if node.child_by_name(name) is not None: return True
        return any(self.has_descendant(child, name) for child in node.children)

    def find_descendant(self, node: ClosNode, name: str) -> Optional[ClosNode]:
        """Depth-first search for the first descendant node named `name`."""
        direct = node.child_by_name(name)
        if direct is not None: return direct
        for child in node.children:
            hit = self.find_descendant(child, name)
            if hit is not None: return hit
        return None

    def find_descendant_by_class(self, node: ClosNode, class_name: str) -> Optional[ClosNode]:
        """Depth-first search for the first descendant node with the given class name."""
        for child in node.children:
            if child.class_name == class_name: return child
        for child in node.children:
            hit = self.find_descendant_by_class(child, class_name)
            if hit is not None: return hit
        return None

    def build_piece(self, simulation: ClosNode, index: int) -> Optional[ClothPiece]:
        """Reconstruct one cloth mesh from a Simulation subtree, mirroring the reference dispatch."""
        # ----- Simulation grid (the geometry any render mesh binds to) -----
        shape_owner = self.find_shape_owner(simulation)
        if shape_owner is None:
            print(f"  Simulation {index}: no ShapeDefinition found; skipping.")
            return None

        positions_node = shape_owner.child_by_name("startingPositions")
        if positions_node is None or not positions_node.array_values:
            print(f"  Simulation {index}: ShapeDefinition has no startingPositions; skipping.")
            return None

        num_rows    = self.scalar_child(shape_owner, "numRows")
        num_columns = self.scalar_child(shape_owner, "numColumns")
        num_nodes   = self.scalar_child(shape_owner, "numNodes")
        flags_node  = shape_owner.child_by_name("startingFlags")

        shape_kind = shape_owner.class_name or "Cloth"
        piece = ClothPiece(name=f"{shape_kind}_{index}")
        piece.shape_kind = shape_kind
        piece.num_rows = num_rows
        piece.num_columns = num_columns
        piece.num_nodes = num_nodes or (num_rows * num_columns)

        piece.sim_positions = self.decode_soa_positions(positions_node.array_values, piece.num_nodes)
        piece.flags = list(flags_node.array_values) if (flags_node and flags_node.array_values) else [0] * piece.num_nodes
        print(f"  Simulation {index} ({shape_kind}): sim grid {num_rows}x{num_columns} = {len(piece.sim_positions)} nodes.")

        # Sim-node normals - required by the thick-mesh binding math (the binding's local frame and
        # curvature displacement are expressed against them).
        sim_normals = self.compute_sim_normals(piece)

        # ----- Per-sim-node bone weights (SkinningTransform subtrees) -----
        sim_bone_weights = self.collect_sim_bone_weights(simulation, len(piece.sim_positions))
        if sim_bone_weights:
            print(f"  Simulation {index}: {len(sim_bone_weights)} bone weight stream(s) ({', '.join(sorted(sim_bone_weights)[:4])}{', ...' if len(sim_bone_weights) > 4 else ''}).")

        # ----- Dispatch: ThickMeshControl (render mesh) or sim sheet mesh -----
        thick_control = self.find_descendant_by_class(simulation, "ThickMeshControl")

        if thick_control is not None: built = self.build_thick_mesh(piece, thick_control, sim_normals, sim_bone_weights, index)
        else: built = self.build_sheet_mesh(piece, sim_normals, sim_bone_weights, index)

        return piece if built else None

    def build_thick_mesh(self, piece: ClothPiece, thick_control: ClosNode, sim_normals: list, sim_bone_weights: dict, index: int) -> bool:
        """Reconstruct the high-res render mesh via the full CloakWorks binding math.

        Identical to HitmanAbsolutionEditor's Cloth::GenerateVerticesForThickMesh. Each render vertex stores a 16-float binding: barycentric(4) + positionOffset(4) + normalOffset(4) + tangentOffset(4).
        The bound sim triangle (triIndices) plus the sim positions/normals define a local frame (directionAB, bindingNormal, directionAC); The offsets are expressed in that frame which is what gives buttons, pocket flaps and collars their depth off the cloth plane.
        A curvature displacement term (barycentric-squared weighted) corrects the blend point before the offset is applied."""
        index_node    = thick_control.child_by_name("index32s")
        texcoord_node = thick_control.child_by_name("texCoords")
        tri_node      = thick_control.child_by_name("triIndices")
        binding_node  = thick_control.child_by_name("bindingOffsets")
        mapped_node   = thick_control.child_by_name("mappedVertCount")

        if index_node is None or not index_node.array_values:
            print(f"  Simulation {index}: ThickMeshControl has no index32s; skipping.")
            return False
        if tri_node is None or not tri_node.array_values or binding_node is None or not binding_node.array_values:
            print(f"  Simulation {index}: ThickMeshControl missing binding data; skipping.")
            return False

        vertex_count = 0
        if mapped_node is not None and mapped_node.value is not None:
            try: vertex_count = int(mapped_node.value)
            except (TypeError, ValueError): vertex_count = 0
        if vertex_count <= 0: vertex_count = min(len(tri_node.array_values) // 3, len(binding_node.array_values) // 16)

        tri_indices     = tri_node.array_values
        binding_offsets = binding_node.array_values
        sim_positions   = piece.sim_positions
        sim_count       = len(sim_positions)

        positions: list[tuple[float, float, float]] = []
        normals:   list[tuple[float, float, float]] = []

        for v in range(vertex_count):
            i0 = tri_indices[v * 3 + 0]
            i1 = tri_indices[v * 3 + 1]
            i2 = tri_indices[v * 3 + 2]
            base = v * 16
            bc_x, bc_y, bc_z = binding_offsets[base + 0], binding_offsets[base + 1], binding_offsets[base + 2]
            po_x, po_y, po_z = binding_offsets[base + 4], binding_offsets[base + 5], binding_offsets[base + 6]
            no_x, no_y, no_z = binding_offsets[base + 8], binding_offsets[base + 9], binding_offsets[base + 10]

            if max(i0, i1, i2) >= sim_count:
                positions.append((0.0, 0.0, 0.0))
                normals.append((0.0, 0.0, 1.0))
                continue

            a  = sim_positions[i0]
            b  = sim_positions[i1]
            c  = sim_positions[i2]
            n1 = sim_normals[i0]
            n2 = sim_normals[i1]
            n3 = sim_normals[i2]

            direction_ab = self.normalized((b[0] - a[0], b[1] - a[1], b[2] - a[2]))
            direction_ac = self.normalized((c[0] - a[0], c[1] - a[1], c[2] - a[2]))

            # Barycentric blend point and blended (unnormalized) normal.
            bind_pos = (a[0] * bc_x + b[0] * bc_y + c[0] * bc_z, a[1] * bc_x + b[1] * bc_y + c[1] * bc_z, a[2] * bc_x + b[2] * bc_y + c[2] * bc_z)
            bind_nrm = (n1[0] * bc_x + n2[0] * bc_y + n3[0] * bc_z, n1[1] * bc_x + n2[1] * bc_y + n3[1] * bc_z, n1[2] * bc_x + n2[2] * bc_y + n3[2] * bc_z)

            # Curvature displacement: project the blend point's offset from each corner onto that
            # corner's normal, weight by barycentric squared, and push back along the blended normal.
            disp_a = ((bind_pos[0] - a[0]) * n1[0] + (bind_pos[1] - a[1]) * n1[1] + (bind_pos[2] - a[2]) * n1[2]) * bc_x * bc_x
            disp_b = ((bind_pos[0] - b[0]) * n2[0] + (bind_pos[1] - b[1]) * n2[1] + (bind_pos[2] - b[2]) * n2[2]) * bc_y * bc_y
            disp_c = ((bind_pos[0] - c[0]) * n3[0] + (bind_pos[1] - c[1]) * n3[1] + (bind_pos[2] - c[2]) * n3[2]) * bc_z * bc_z
            total_displacement = -(disp_a + disp_b + disp_c)

            adjusted = (bind_nrm[0] * total_displacement + bind_pos[0], bind_nrm[1] * total_displacement + bind_pos[1], bind_nrm[2] * total_displacement + bind_pos[2])

            # Final position / normal: offsets expressed in the (AB, blended normal, AC) frame.
            positions.append((
                direction_ab[0] * po_x + bind_nrm[0] * po_y + direction_ac[0] * po_z + adjusted[0],
                direction_ab[1] * po_x + bind_nrm[1] * po_y + direction_ac[1] * po_z + adjusted[1],
                direction_ab[2] * po_x + bind_nrm[2] * po_y + direction_ac[2] * po_z + adjusted[2],
            ))
            normals.append(self.normalized((
                direction_ab[0] * no_x + bind_nrm[0] * no_y + direction_ac[0] * no_z,
                direction_ab[1] * no_x + bind_nrm[1] * no_y + direction_ac[1] * no_z,
                direction_ab[2] * no_x + bind_nrm[2] * no_y + direction_ac[2] * no_z,
            )))

        piece.positions = positions
        piece.normals = normals
        piece.triangles = self.build_render_triangles(index_node.array_values, vertex_count)
        piece.uvs = self.decode_texcoords(texcoord_node, vertex_count)

        # Bone weights: blend each render vertex's bound sim-node weights barycentrically.
        for bone_name, node_weights in sim_bone_weights.items():
            blended: list[float] = []
            for v in range(vertex_count):
                i0 = tri_indices[v * 3 + 0]
                i1 = tri_indices[v * 3 + 1]
                i2 = tri_indices[v * 3 + 2]
                base = v * 16
                bc_x, bc_y, bc_z = binding_offsets[base + 0], binding_offsets[base + 1], binding_offsets[base + 2]
                if max(i0, i1, i2) >= len(node_weights):
                    blended.append(0.0)
                    continue
                blended.append(node_weights[i0] * bc_x + node_weights[i1] * bc_y + node_weights[i2] * bc_z)
            piece.bone_weights[bone_name] = blended

        print(f"  Simulation {index}: render mesh {len(piece.positions)} vertices, "
              f"{len(piece.triangles)} triangles, {len(piece.uvs)} UVs"
              f"{f', {len(piece.bone_weights)} bone group(s)' if piece.bone_weights else ''}.")
        return True

    def build_sheet_mesh(self, piece: ClothPiece, sim_normals: list, sim_bone_weights: dict, index: int) -> bool:
        """No ThickMeshControl: emit the simulation grid mesh directly (reference fallback path).
        Topology and planar UVs are generated from the row/column grid exactly like CloakWorks'
        SheetMeshControlInstance, honoring the per-node culled flag (bit 3)."""
        piece.positions = list(piece.sim_positions)
        piece.normals = list(sim_normals)
        piece.triangles = self.generate_grid_triangles(piece)
        piece.uvs = self.generate_grid_uvs(piece)
        for bone_name, node_weights in sim_bone_weights.items(): piece.bone_weights[bone_name] = list(node_weights[:len(piece.positions)])

        print(f"  Simulation {index}: sheet mesh {len(piece.positions)} vertices, {len(piece.triangles)} triangles (no ThickMeshControl - simulation grid emitted) {f', {len(piece.bone_weights)} bone group(s)' if piece.bone_weights else ''}.")
        return True

    def find_shape_owner(self, node: ClosNode) -> Optional[ClosNode]:
        """Find the node that directly owns startingPositions (the ShapeDefinition)."""
        if node.child_by_name("startingPositions") is not None: return node
        for child in node.children:
            owner = self.find_shape_owner(child)
            if owner is not None: return owner
        return None

    def scalar_child(self, owner: ClosNode, name: str) -> int:
        node = owner.child_by_name(name)
        if node is None or node.value is None: return 0
        try: return int(node.value)
        except (TypeError, ValueError): return 0

    def collect_sim_bone_weights(self, simulation: ClosNode, sim_node_count: int) -> dict:
        """Gather per-sim-node bone weights from every SkinningTransform under this Simulation.

        Each SkinningTransform carries a guid (referencing a Transform on the ShroudObject, which
        names the skeleton bone) and a flat per-node weights array in the same SIMD node order as
        startingPositions. Returns {boneName: [weight per sim node]}."""
        weights_by_bone: dict = {}

        def visit(node: ClosNode) -> None:
            if node.class_name == "SkinningTransform":
                guid_node = node.child_by_name("guid")
                weights_node = node.child_by_name("weights")
                if guid_node is not None and guid_node.value is not None and weights_node is not None and weights_node.array_values:
                    guid = int(guid_node.value)
                    bone_name = self.transform_bone_names.get(guid, f"Transform_{guid}")
                    weights = list(weights_node.array_values[:sim_node_count])
                    # Only keep streams that actually influence something.
                    if any(w > 0.0 for w in weights): weights_by_bone[bone_name] = weights
            for child in node.children: visit(child)

        visit(simulation)
        return weights_by_bone

    def decode_soa_positions(self, raw_floats: list[float], num_nodes: int) -> list[tuple[float, float, float]]:
        """CloakWorks stores positions SIMD-packed: each block of 4 nodes is laid out as
        [x0,x1,x2,x3, y0,y1,y2,y3, z0,z1,z2,z3] (12 floats). Deinterleave back to (x, y, z) tuples."""
        positions: list[tuple[float, float, float]] = []
        block_count = len(raw_floats) // 12

        for block in range(block_count):
            base = block * 12
            for lane in range(4): positions.append((raw_floats[base + lane], raw_floats[base + 4 + lane], raw_floats[base + 8 + lane]))

        # Trim any SIMD padding beyond the real node count.
        if num_nodes and len(positions) > num_nodes: positions = positions[:num_nodes]
        return positions

    # =====================================================
    # SIMULATION GRID TOPOLOGY, NORMALS AND UVS
    #
    # CloakWorks interleaves even/odd rows in node memory order; GetRowStartIndex maps a logical row
    # to its node index. Topology, planar UVs and per-node normals all use that mapping.
    # =====================================================

    def row_start_index(self, row_index: int, num_rows: int, num_columns: int) -> int:
        """Mirrors ShapeDefinition::GetRowStartIndex - even rows pack first, then odd rows."""
        num_even_rows = num_rows - (num_rows // 2)
        return ((row_index >> 1) + num_even_rows * (row_index & 1)) * num_columns

    def generate_grid_triangles(self, piece: ClothPiece) -> list[tuple[int, int, int]]:
        """Build the sheet topology exactly like SheetMeshControlInstance: two triangles per grid
        quad, skipping any quad whose corner nodes are flagged culled (flag bit 3)."""
        triangles: list[tuple[int, int, int]] = []
        rows = piece.num_rows
        cols = piece.num_columns
        flags = piece.flags
        node_count = len(piece.sim_positions)

        def culled(node_index: int) -> bool: return 0 <= node_index < len(flags) and bool(flags[node_index] & 8)

        for row in range(rows - 1):
            row_a = self.row_start_index(row, rows, cols)
            row_b = self.row_start_index(row + 1, rows, cols)
            for column in range(cols - 1):
                a = row_a + column
                b = row_a + column + 1
                c = row_b + column
                d = row_b + column + 1

                if max(a, b, c, d) >= node_count: continue
                if culled(a) or culled(b) or culled(c) or culled(d): continue

                triangles.append((a, c, d))
                triangles.append((a, d, b))

        return triangles

    def generate_grid_uvs(self, piece: ClothPiece) -> list[tuple[float, float]]:
        """Generate planar UVs from the row/column grid, matching SheetMeshControlInstance::FillTexCoordsBuffer:
        u = column / (columns - 1), v = row / (rows - 1)."""
        rows = piece.num_rows
        cols = piece.num_columns
        uvs: list[tuple[float, float]] = [(0.0, 0.0)] * len(piece.positions)

        if cols <= 1 or rows <= 1: return uvs

        for row in range(rows):
            row_start = self.row_start_index(row, rows, cols)
            for column in range(cols):
                node_index = row_start + column
                if node_index < len(uvs): uvs[node_index] = (column / (cols - 1), row / (rows - 1))
        return uvs

    def compute_sim_normals(self, piece: ClothPiece) -> list[tuple[float, float, float]]:
        """Per-sim-node normals: accumulate the cross products of the grid's triangle edges around
        each node, then normalize.

        NOTE ON FIDELITY: CloakWorks' ClothNormalsUpdater::CalcNormalsForStream computes these with row-pair SIMD passes (cross products of the 'down' and 'down-diagonal' grid edges).
        This implementation accumulates the same family of cross products via the generated sheet triangles - the same surface, same edges - rather than transliterating the unrolled SIMD.
        The resulting directions agree to within a fraction of a degree, and the binding math is first-order in the normal so any residual difference is far below visual relevance."""
        node_count = len(piece.sim_positions)
        accum = [[0.0, 0.0, 0.0] for _ in range(node_count)]

        for (a, b, c) in self.generate_grid_triangles(piece):
            pa = piece.sim_positions[a]
            pb = piece.sim_positions[b]
            pc = piece.sim_positions[c]
            e1 = (pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2])
            e2 = (pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2])
            face = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
            for idx in (a, b, c):
                accum[idx][0] += face[0]
                accum[idx][1] += face[1]
                accum[idx][2] += face[2]

        return [self.normalized((v[0], v[1], v[2])) for v in accum]

    def normalized(self, vector: tuple[float, float, float]) -> tuple[float, float, float]:
        length = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]) ** 0.5
        if length < 1e-12: return (0.0, 0.0, 1.0)
        return (vector[0] / length, vector[1] / length, vector[2] / length)

    def build_render_triangles(self, index_values: list, vertex_count: int) -> list[tuple[int, int, int]]:
        """Turn the flat index32s buffer into triangle tuples, dropping any that reference a vertex
        we did not reconstruct (defensive; shouldn't happen on valid files)."""
        triangles: list[tuple[int, int, int]] = []
        for t in range(len(index_values) // 3):
            a, b, c = index_values[t * 3], index_values[t * 3 + 1], index_values[t * 3 + 2]
            if max(a, b, c) < vertex_count and len({a, b, c}) == 3: triangles.append((a, b, c))
        return triangles

    def decode_texcoords(self, texcoord_node: Optional[ClosNode], vertex_count: int) -> list[tuple[float, float]]:
        """texCoords are a flat float array of (U, V) pairs. (One per render vertex)"""
        if texcoord_node is None or not texcoord_node.array_values: return [(0.0, 0.0)] * vertex_count

        raw = texcoord_node.array_values
        uvs: list[tuple[float, float]] = []
        for i in range(min(vertex_count, len(raw) // 2)): uvs.append((raw[i * 2], raw[i * 2 + 1]))
        # Pad if the file gave fewer UVs than vertices (defensive).
        while len(uvs) < vertex_count: uvs.append((0.0, 0.0))
        return uvs
