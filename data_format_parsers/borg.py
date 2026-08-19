# =====================================================
#   GLACIER 2 BORG (BONERIG) PARSER
#       Parses IOI Interactive's Glacier 2 BoneRig
#       format used by:
#           - Hitman: Absolution
#           - Hitman: World of Assassination (Trilogy)
#           - 007: First Light (Bond)
#
#       Mirrors BORG.bt 1:1. If you change the binary
#       template, mirror the change here.
# =====================================================

import struct
from ..io import Reader
from ..utilities import *

# =====================================================
# CONSTANTS - mirrored from BORG.bt enums
# =====================================================

# Bone constraint type discriminator. Same numeric values in both games;
# only the storage width differs (u8 in Trilogy, u16 in Bond).
CONSTRAINT_TYPE_NONE   = 0
CONSTRAINT_TYPE_LOOKAT = 1
CONSTRAINT_TYPE_ROTATE = 2

# The boneName field is a 34-byte ASCII slot, null-padded.
BORG_BONE_NAME_LENGTH = 34

# =====================================================
# MAIN PARSER CLASS
# =====================================================

class BORG():
    """Glacier 2 BoneRig parser. Builds a fully-populated object model of the skeleton from a `.BORG` file."""
    def __init__(self, file_path: str, game: str):
        """Construct the parser and run the full parse pass.

        Args:
            file_path: Absolute path to the `.BORG` file to read.
            game: Either `GLACIER2_TRILOGY` (Hitman WoA) or `GLACIER2_BOND` (007 First Light).
                  Drives header size, bone-constraint index width, and trailing-field handling.
        """
        super().__init__()

        # ===============================
        # == CLASS MEMBERS ==============
        # ===============================

        # -- INPUT METADATA
        self.skeleton_file: str = file_path
        """The path to the source `.BORG` file."""

        self.game: str = game
        """Which Glacier 2 title produced this file. Drives parser branching - GLACIER2_TRILOGY or GLACIER2_BOND."""

        # -- HEADER FIELDS
        self.main_header_offset: int = 0
        """The u64 pointer to the BoneRigHeader read from the prolog. The header sits at the END of the file."""

        self.number_of_bones: int = 0
        """Total number of bones in the rig. Drives the length of every per-bone array."""

        self.number_of_animated_bones: int = 0
        """Subset of bones not driven by a constraint - i.e. bones that participate in animation directly."""

        self.bone_definitions_offset: int = 0
        self.bind_pose_offset: int = 0
        self.bind_pose_inv_global_matrices_offset: int = 0
        self.bone_constraints_header_offset: int = 0
        self.pose_bone_header_offset: int = 0
        self.invert_global_bones_offset: int = 0
        """Always 0 in observed files; an unused leftover from an older version of the format."""

        self.bone_map_offset: int = 0
        """Trilogy-only u64. Always 0 in observed files; dropped entirely in Bond."""

        # -- BONE DEFINITIONS (64 B per bone)
        self.bone_definitions: list[dict] = []
        """Per-bone metadata. Each entry:
           {
               'center': (x, y, z),         # Local-space bounding sphere / OBB center
               'parent_index': int,         # -1 for root
               'size': (x, y, z),           # Local-space half-extents
               'name': str,                 # ASCII; null-padding stripped
               'body_part': int,            # Logical body region tag; 0 = unspecified
           }
        """

        # -- BIND POSES (32 B per bone)
        self.bind_poses: list[dict] = []
        """Per-bone bind-pose Scale-Vector-Quaternion in PARENT-LOCAL space. Each entry:
           {
               'rotation': (qx, qy, qz, qw),     # Unit quaternion
               'position': (px, py, pz, pw),     # pw is padding (always 1.0 in observed files)
           }
        """

        # -- INVERSE GLOBAL MATRICES (48 B per bone)
        self.inverse_global_matrices: list[dict] = []
        """Per-bone inverse-bind world matrix as 4 rows x 3 cols. Each entry:
           {
               'row_0': (m00, m01, m02),       # Rotation/scale 3x3 matrix row 0
               'row_1': (m10, m11, m12),
               'row_2': (m20, m21, m22),
               'translation': (tx, ty, tz),    # World-space translation
           }
        """

        # -- BONE CONSTRAINTS
        self.constraint_count: int = 0
        """Number of bone-constraint records present in the file."""

        self.bone_constraints: list[dict] = []
        """Variable-length list of LOOKAT and/or ROTATE constraints. Each entry carries a 'type' key:
              type == CONSTRAINT_TYPE_LOOKAT -> aim-at-target constraint with up-axis control
              type == CONSTRAINT_TYPE_ROTATE -> twist constraint driven by a reference bone
           See `parse_constraint_lookat` / `parse_constraint_rotate` for the full field list.
        """

        # -- POSE BONE SYSTEM (optional - all-zero header when unused)
        self.pose_header: Optional[dict] = None
        """Sub-header for the optional pose / face-bone system. None if the rig doesn't use poses."""

        self.pose_bones: list[dict] = []
        """All POSE_BONE entries (48 B each). Empty if poseBoneCountTotal == 0."""

        self.pose_bone_indices: list[int] = []
        """Which bones each pose drives (one u32 per POSE_BONE entry)."""

        self.pose_entry_indices: list[int] = []
        """Per-pose start indices into PoseBoneIndices (one u32 per pose)."""

        self.pose_bone_counts: list[int] = []
        """Per-pose bone counts (one u32 per pose)."""

        self.pose_names: list[str] = []
        """Per-pose name strings (parsed from the variable-length cstring list)."""

        self.pose_names_entry_indices: list[int] = []
        """Per-pose name offsets into the names list (one u32 per pose)."""

        self.face_bone_indices: list[int] = []
        """Subset of bones that drive facial poses (one u32 per face-bone)."""

        # -- VALIDATION
        if game not in (GLACIER2_ABSOLUTION, GLACIER2_TRILOGY, GLACIER2_BOND):
            raise ValueError(f"Unsupported game type for BORG parsing: '{game}'. Use GLACIER2_ABSOLUTION, GLACIER2_TRILOGY or GLACIER2_BOND.")

        # ===============================
        # == PARSE THE DATA =============
        # ===============================
        self.parse_skeleton_file()

    # =====================================================
    # TOP-LEVEL DRIVER
    # =====================================================

    def parse_skeleton_file(self) -> None:
        """Parse the skeleton file. Parse order strictly matches BORG.bt."""
        print(f"\nParsing BORG skeleton ({self.game}): {self.skeleton_file}\n")
        reader = Reader(open(self.skeleton_file, "rb").read())

        # -----------------------------------------
        # PROLOG - u64 pointer to main header + 8 bytes padding
        # -----------------------------------------
        self.main_header_offset = reader.uint64()
        reader.skip(8) # 8 bytes alignment padding (always zeros)
        print(f"Main Header Offset: 0x{self.main_header_offset:08X}")

        # -----------------------------------------
        # MAIN HEADER (Trilogy = 40 B, Bond = 32 B)
        # -----------------------------------------
        reader.seek(self.main_header_offset)
        self.parse_main_header(reader)

        # -----------------------------------------
        # BONE DEFINITIONS - 64 B per bone
        # -----------------------------------------
        if self.bone_definitions_offset != 0 and self.number_of_bones > 0:
            reader.seek(self.bone_definitions_offset)
            self.parse_bone_definitions(reader)

        # -----------------------------------------
        # BIND POSES (SVQ) - 32 B per bone
        # -----------------------------------------
        if self.bind_pose_offset != 0 and self.number_of_bones > 0:
            reader.seek(self.bind_pose_offset)
            self.parse_bind_poses(reader)

        # -----------------------------------------
        # INVERSE GLOBAL MATRICES - 48 B per bone
        # -----------------------------------------
        if self.bind_pose_inv_global_matrices_offset != 0 and self.number_of_bones > 0:
            reader.seek(self.bind_pose_inv_global_matrices_offset)
            self.parse_inverse_global_matrices(reader)

        # -----------------------------------------
        # BONE CONSTRAINTS - variable-length records
        # -----------------------------------------
        if self.bone_constraints_header_offset != 0:
            reader.seek(self.bone_constraints_header_offset)
            self.parse_bone_constraints(reader)

        # -----------------------------------------
        # POSE BONE SYSTEM (optional)
        # -----------------------------------------
        if self.pose_bone_header_offset != 0:
            reader.seek(self.pose_bone_header_offset)
            self.parse_pose_bone_block(reader)

        print(f"\nBORG PARSING COMPLETE!  {self.number_of_bones} bones, {self.constraint_count} constraints.\n")

    # =====================================================
    # SECTION PARSERS
    # =====================================================

    def parse_main_header(self, reader: Reader) -> None:
        """Parse the BoneRig main header (40 B on Trilogy, 32 B on Bond)."""
        self.number_of_bones                       = reader.uint32()
        self.number_of_animated_bones              = reader.uint32()
        self.bone_definitions_offset               = reader.uint32()
        self.bind_pose_offset                      = reader.uint32()
        self.bind_pose_inv_global_matrices_offset  = reader.uint32()
        self.bone_constraints_header_offset        = reader.uint32()
        self.pose_bone_header_offset               = reader.uint32()
        self.invert_global_bones_offset            = reader.uint32()

        # Trilogy AND Absolution carry an extra u64 boneMapOffset (always zero in observed files).
        # Bond drops that field entirely - the header is 32 bytes flat. Absolution's BORG is
        # byte-for-byte the WoA layout (verified: 64 B bone defs, 32 B SVQ bind poses, 48 B inverse
        # global matrices, u8 constraint indices, name at def+28).
        if self.game in (GLACIER2_TRILOGY, GLACIER2_ABSOLUTION): self.bone_map_offset = reader.uint64()

        print(f"  Bones: {self.number_of_bones} (animated: {self.number_of_animated_bones})")
        print(f"  Bone Defs Offset:        0x{self.bone_definitions_offset:08X}")
        print(f"  Bind Pose Offset:        0x{self.bind_pose_offset:08X}")
        print(f"  Inv Global Mats Offset:  0x{self.bind_pose_inv_global_matrices_offset:08X}")
        print(f"  Constraints Offset:      0x{self.bone_constraints_header_offset:08X}")
        print(f"  Pose Bone Hdr Offset:    0x{self.pose_bone_header_offset:08X}")

    def parse_bone_definitions(self, reader: Reader) -> None:
        """Parse `numberOfBones` BONE_DEFINITION records (64 B each)."""
        print(f"\nReading {self.number_of_bones} bone definitions...")
        # PERFORMANCE: one bulk read + struct.iter_unpack instead of five reader calls per bone.
        # Record layout (64 B): VEC3F center, int32 parent, VEC3F size, char[34] name, int16 bodyPart.
        raw = reader.read_bytes(self.number_of_bones * 64)
        defs: list[dict] = []

        for rec in struct.iter_unpack("<3fi3f34sh", raw):
            defs.append({
                "center":       (rec[0], rec[1], rec[2]),
                "parent_index": rec[3],
                "size":         (rec[4], rec[5], rec[6]),
                "name":         strip_null_padding(rec[7].decode("utf-8", errors="replace")),
                "body_part":    rec[8],
            })

        self.bone_definitions = defs

    def parse_bind_poses(self, reader: Reader) -> None:
        """Parse `numberOfBones` BIND_POSE_SVQ records (32 B each: VEC4F rotation + VEC4F position)."""
        print(f"\nReading {self.number_of_bones} bind poses (SVQ)...")
        # PERFORMANCE: bulk read; each record is VEC4F rotation (unit quaternion xyzw)
        # followed by VEC4F position (xyz, w = 1.0 padding).
        raw = reader.read_bytes(self.number_of_bones * 32)
        self.bind_poses = [
            {"rotation": rec[0:4], "position": rec[4:8]}
            for rec in struct.iter_unpack("<8f", raw)
        ]

    def parse_inverse_global_matrices(self, reader: Reader) -> None:
        """Parse `numberOfBones` INV_GLOBAL_MATRIX_4X3 records (48 B each: 4 rows of VEC3F)."""
        print(f"\nReading {self.number_of_bones} inverse-global matrices...")
        # PERFORMANCE: bulk read; each record is 4 rows of VEC3F (rotation rows 0-2 + translation).
        raw = reader.read_bytes(self.number_of_bones * 48)
        self.inverse_global_matrices = [
            {"row_0": rec[0:3], "row_1": rec[3:6], "row_2": rec[6:9], "translation": rec[9:12]}
            for rec in struct.iter_unpack("<12f", raw)
        ]

    def parse_bone_constraints(self, reader: Reader) -> None:
        """Parse the bone-constraint block: u32 count followed by `count` variable-length records.

        Each record begins with a type field whose width depends on the game:
          - Trilogy: u8 type discriminator (then u8-typed indices)
          - Bond:    u16 type discriminator (then u16-typed indices)
        """
        self.constraint_count = reader.uint32()
        print(f"\nReading {self.constraint_count} bone constraints...")

        if self.constraint_count == 0: return

        constraints: list[dict] = []
        is_bond = (self.game == GLACIER2_BOND)

        for _ in range(self.constraint_count):
            # Peek the type without advancing - the per-type parser will read it again.
            saved_offset = reader.tell()
            type_value = reader.ushort() if is_bond else reader.ubyte()
            reader.seek(saved_offset)

            if   type_value == CONSTRAINT_TYPE_LOOKAT: constraints.append(self.parse_constraint_lookat(reader, is_bond))
            elif type_value == CONSTRAINT_TYPE_ROTATE: constraints.append(self.parse_constraint_rotate(reader, is_bond))
            else:
                print(f"  WARNING: Unrecognized constraint type {type_value} - aborting constraint parse.")
                break

        self.bone_constraints = constraints

    def parse_constraint_lookat(self, reader: Reader, is_bond: bool) -> dict:
        """Parse a LOOKAT (aim-at-target) constraint. Trilogy = 56 B (u8 indices), Bond = 68 B (u16 indices).

        WARNING: The Bond layout for LOOKAT is PREDICTED in the binary template; no Bond LOOKAT
        samples have been confirmed yet. Treat results from Bond LOOKAT records with skepticism.
        """
        # Index widths and their reader methods - keeps the field list below readable.
        if is_bond: read_idx = reader.ushort
        else:       read_idx = reader.ubyte

        constraint_type        = read_idx() # Always CONSTRAINT_TYPE_LOOKAT (== 1)
        bone_index             = read_idx()
        target_count           = read_idx()
        look_at_axis           = read_idx()
        up_bone_alignment_axis = read_idx()
        look_at_flip           = read_idx()
        up_flip                = read_idx()
        up_node_control        = read_idx()
        up_node_parent_index   = read_idx()
        target_parent_index_0  = read_idx()
        target_parent_index_1  = read_idx()
        alignment_padding      = read_idx()

        bone_targets_weights = reader.vec2f()
        target_position_0    = reader.vec3f()
        target_position_1    = reader.vec3f()
        up_position          = reader.vec3f()

        return {
            "type":                   CONSTRAINT_TYPE_LOOKAT,
            "bone_index":             bone_index,
            "target_count":           target_count,
            "look_at_axis":           look_at_axis,
            "up_bone_alignment_axis": up_bone_alignment_axis,
            "look_at_flip":           look_at_flip,
            "up_flip":                up_flip,
            "up_node_control":        up_node_control,
            "up_node_parent_index":   up_node_parent_index,
            "target_parent_indices":  (target_parent_index_0, target_parent_index_1),
            "alignment_padding":      alignment_padding,
            "bone_targets_weights":   bone_targets_weights,
            "target_position_0":      target_position_0,
            "target_position_1":      target_position_1,
            "up_position":            up_position,
        }

    def parse_constraint_rotate(self, reader: Reader, is_bond: bool) -> dict:
        """Parse a ROTATE (twist) constraint. Trilogy = 8 B, Bond = 12 B.

        Drives `bone_index` by `reference_bone_index * twist_weight`. Heavy use in Bond -
        all 18 constraints in the 0180310C89670206.BORG sample are ROTATE (twist bones
        for L/R femur, shoulder, elbow joints).
        """
        if is_bond:
            constraint_type      = reader.ushort()
            bone_index           = reader.ushort()
            reference_bone_index = reader.ushort()
            reader.skip(2) # Reserved padding to maintain 4-byte alignment before the float
        else:
            constraint_type      = reader.ubyte()
            bone_index           = reader.ubyte()
            reference_bone_index = reader.ubyte()
            reader.skip(1) # Reserved padding to maintain 4-byte alignment before the float

        twist_weight = reader.float32() # Fraction of reference-bone rotation applied; range [-1, 1]

        return {
            "type":                 CONSTRAINT_TYPE_ROTATE,
            "bone_index":           bone_index,
            "reference_bone_index": reference_bone_index,
            "twist_weight":         twist_weight,
        }

    def parse_pose_bone_block(self, reader: Reader) -> None:
        """Parse the optional pose / face-bone block. All sub-arrays are guarded by their own counts and offsets - the file isn't required to keep these adjacent."""
        # ----- POSE_BONE_HEADER (40 bytes / 10 u32) -----
        pose_bone_array_offset         = reader.uint32()
        pose_bone_index_array_offset   = reader.uint32()
        pose_bone_count_total          = reader.uint32()
        pose_entry_index_array_offset  = reader.uint32()
        pose_bone_count_array_offset   = reader.uint32()
        pose_count                     = reader.uint32()
        names_list_offset              = reader.uint32()
        names_entry_index_array_offset = reader.uint32()
        face_bone_index_array_offset   = reader.uint32()
        face_bone_count                = reader.uint32()

        self.pose_header = {
            "pose_bone_array_offset":         pose_bone_array_offset,
            "pose_bone_index_array_offset":   pose_bone_index_array_offset,
            "pose_bone_count_total":          pose_bone_count_total,
            "pose_entry_index_array_offset":  pose_entry_index_array_offset,
            "pose_bone_count_array_offset":   pose_bone_count_array_offset,
            "pose_count":                     pose_count,
            "names_list_offset":              names_list_offset,
            "names_entry_index_array_offset": names_entry_index_array_offset,
            "face_bone_index_array_offset":   face_bone_index_array_offset,
            "face_bone_count":                face_bone_count,
        }

        # If the rig doesn't use poses every field above is 0 - bail out cleanly.
        if pose_bone_count_total == 0 and pose_count == 0 and face_bone_count == 0:
            print("\nPose-bone block is empty - rig doesn't use poses.")
            return

        print(f"\nReading pose-bone block: {pose_bone_count_total} pose bones | {pose_count} poses | {face_bone_count} face bones")

        # ----- POSE_BONE array (48 B each: VEC4F rot + VEC4F pos + VEC4F scale) -----
        if pose_bone_count_total > 0 and pose_bone_array_offset != 0:
            reader.seek(pose_bone_array_offset)
            for _ in range(pose_bone_count_total):
                rotation = reader.vec4f()
                position = reader.vec4f()
                scale    = reader.vec4f()
                self.pose_bones.append({"rotation": rotation, "position": position, "scale": scale})

        # ----- Pose-bone indices: which bones each pose drives -----
        if pose_bone_count_total > 0 and pose_bone_index_array_offset != 0:
            reader.seek(pose_bone_index_array_offset)
            self.pose_bone_indices = [reader.uint32() for _ in range(pose_bone_count_total)]

        # ----- Per-pose entry start indices into pose_bone_indices -----
        if pose_count > 0 and pose_entry_index_array_offset != 0:
            reader.seek(pose_entry_index_array_offset)
            self.pose_entry_indices = [reader.uint32() for _ in range(pose_count)]

        # ----- Per-pose bone counts -----
        if pose_count > 0 and pose_bone_count_array_offset != 0:
            reader.seek(pose_bone_count_array_offset)
            self.pose_bone_counts = [reader.uint32() for _ in range(pose_count)]

        # ----- Names list (variable-length cstrings packed back-to-back) -----
        if pose_count > 0 and names_list_offset != 0:
            reader.seek(names_list_offset)
            self.pose_names = [reader.read_null_terminated_string() for _ in range(pose_count)]

        # ----- Per-pose name offsets into the names list (relative) -----
        if pose_count > 0 and names_entry_index_array_offset != 0:
            reader.seek(names_entry_index_array_offset)
            self.pose_names_entry_indices = [reader.uint32() for _ in range(pose_count)]

        # ----- Face bone indices (subset of bones that drive facial poses) -----
        if face_bone_count > 0 and face_bone_index_array_offset != 0:
            reader.seek(face_bone_index_array_offset)
            self.face_bone_indices = [reader.uint32() for _ in range(face_bone_count)]

    # =====================================================
    # CONVENIENCE ACCESSORS
    # =====================================================

    def get_bone_name(self, index: int) -> str:
        """Look up a bone name by skeleton index, returning a placeholder if out-of-range."""
        if 0 <= index < len(self.bone_definitions): return self.bone_definitions[index]["name"]
        return f"bone_{index}"

    def get_parent_index(self, index: int) -> int:
        """Return the parent bone index for the given bone, or -1 if root / out-of-range."""
        if 0 <= index < len(self.bone_definitions): return self.bone_definitions[index]["parent_index"]
        return -1

# =====================================================================================================================================================
