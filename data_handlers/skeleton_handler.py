# -----------------------------------------------------------
#   SKELETON IMPORTER / BUILDER
#       Takes the parsed BORG (BoneRig) data and
#       builds it into Blender's scene as an Armature.
#
#   Bones are placed DIRECTLY at their bind pose: each edit bone's full
#   orientation comes from the composed bind-pose matrix (assigned via
#   EditBone.matrix), so the armature's rest pose IS the on-disk bind
#   pose - no pose-channel offsets, no bone-aiming rewrites, no borrowed
#   glTF machinery. Bone length is a purely cosmetic pick (distance to
#   the nearest child) and never affects deformation.
# -----------------------------------------------------------

import mathutils
from ..utilities import *
from ..data_format_parsers.borg import *

# ----------------------------------------
# CONSTANTS
# ----------------------------------------

# Minimum bone length in Blender units. Edit bones with zero length get auto-deleted by Blender,
# so we clamp to a small positive value when a leaf bone has nothing to derive a length from.
MIN_BONE_LENGTH = 0.001

# Default bone length for leaves and childless bones (cosmetic only).
DEFAULT_BONE_LENGTH = 0.05

# Glacier authors content in a left-handed Y-up Z-forward basis (DirectX convention); Blender is
# right-handed Z-up Y-forward. The basis change happens at the per-bone TRS level via
# swizzle_bone_trs, NOT via a global matrix applied at the object level - composing a
# per-bone-swizzled local with the parent's swizzled world cleanly produces a Blender-space world
# transform without any "rotate then unrotate" gymnastics.
#
# Additionally, the ROOT bone gets a one-time -90 degree X rotation applied to its bind-pose
# rotation. Because all descendants inherit through the chain, this single root tweak rotates the
# entire rig from "lying on its back" to "standing up".
ROOT_AXIS_FIX_EULER = mathutils.Euler((math.radians(-90.0), 0.0, 0.0), 'XYZ')

# ----------------------------------------
# BONE WORKSPACE
# ----------------------------------------

class BoneNode:
    """Per-bone scratchpad used while building the armature.

    Carries exactly one set of transforms - the on-disk bind pose after coordinate swizzle and
    root fix - first as a parent-relative TRS, then composed into an armature-space matrix. There
    is no second "display" transform: what you see in edit mode IS the rest pose."""
    __slots__ = (
        "name", "parent", "children",
        "bind_translation", "bind_rotation",
        "bind_arma_matrix", "bone_length", "bl_bone_name",
    )

    def __init__(self):
        self.name              = ""
        self.parent            = -1
        self.children: list[int] = []
        self.bind_translation  = Vector((0, 0, 0))
        self.bind_rotation     = Quaternion((1, 0, 0, 0))
        self.bind_arma_matrix  = Matrix.Identity(4)
        self.bone_length       = DEFAULT_BONE_LENGTH
        self.bl_bone_name      = ""

# ----------------------------------------
# COORDINATE HELPERS
# ----------------------------------------

def swizzle_bone_trs(bind_pose: dict) -> tuple[Vector, Quaternion]:
    """Apply the Glacier-to-Blender coordinate-system change to a per-bone local TRS.

    Glacier (left-handed, Y-up, Z-forward) -> Blender (right-handed, Z-up, Y-forward).

    Position swizzle:   (x, y, z)    -> (x, -z, y)
    Quaternion swizzle: (x, y, z, w) -> (-x, z, -y, w); mathutils Quaternion takes (w, x, y, z),
                        so the constructor call is (w, -x, z, -y).

    The handedness flip (the negation on the rotation axes) and the up/forward axis swap are
    folded into one operation. BORG bind poses carry no scale, so none is returned - scale only
    ever appears via constraints and pose corrections, which are surfaced separately."""
    px, py, pz, _  = bind_pose["position"]
    rx, ry, rz, rw = bind_pose["rotation"]

    translation = Vector((px, -pz, py))
    rotation    = Quaternion((rw, -rx, rz, -ry))
    return translation, rotation

# ----------------------------------------
# BONE TREE PREPARATION
# ----------------------------------------

def init_bones(borg: BORG) -> dict[int, BoneNode]:
    """Allocate BoneNode workspaces from BORG bone definitions + bind poses.

    Applies the coordinate swizzle per bone. The root bone gets the additional -90 degree X
    rotation folded into its rotation, which propagates to every descendant via the bind-pose
    chain and stands the rig upright in Blender."""
    bones: dict[int, BoneNode] = {}

    if not borg.bone_definitions:
        return bones

    if len(borg.bind_poses) < len(borg.bone_definitions):
        print(f"   WARNING: BORG has {len(borg.bone_definitions)} bones but only {len(borg.bind_poses)} bind poses.")

    # Pass 1: allocate one workspace per bone, fill in TRS and parent index.
    for i, bone_def in enumerate(borg.bone_definitions):
        node = BoneNode()
        bones[i] = node

        # Use the parser's decoded name when present, falling back to a synthetic name for any
        # bone with an empty name field (defensive - haven't seen this in real files).
        node.name = bone_def["name"] if bone_def["name"] else f"bone_{i}"

        # Per-bone TRS swizzle. If bind poses are missing for this bone, fall back to identity so
        # the rest of the pipeline still produces something sensible (orphan bone at origin).
        if i < len(borg.bind_poses):
            node.bind_translation, node.bind_rotation = swizzle_bone_trs(borg.bind_poses[i])
        else:
            node.bind_translation, node.bind_rotation = Vector((0, 0, 0)), Quaternion((1, 0, 0, 0))

        # Root rotation fix - only on bone 0; it propagates to every descendant via the chain.
        if i == 0:
            node.bind_rotation.rotate(ROOT_AXIS_FIX_EULER)

        node.parent = bone_def["parent_index"]

    # Pass 2: build the children lists by walking parent links.
    bone_count = len(borg.bone_definitions)
    for i, bone_def in enumerate(borg.bone_definitions):
        parent_index = bone_def["parent_index"]
        if 0 <= parent_index < bone_count:
            bones[parent_index].children.append(i)

    return bones

def compose_armature_matrices(bones: dict[int, BoneNode]) -> None:
    """Compose each bone's parent-relative bind TRS into an armature-space 4x4 matrix.

    Straight hierarchy walk: world(bone) = world(parent) @ local(bone). The result is the exact
    bind pose in Blender space - the matrix each edit bone will be placed with."""
    def compose(bone_id: int, parent_matrix: Matrix) -> None:
        bone = bones[bone_id]
        local_matrix = Matrix.Translation(bone.bind_translation) @ bone.bind_rotation.to_matrix().to_4x4()
        bone.bind_arma_matrix = parent_matrix @ local_matrix
        for child_id in bone.children:
            compose(child_id, bone.bind_arma_matrix)

    # Walk from every root (parent index out of range). Real BORGs have a single root at index 0
    # (GROUND), but orphaned bones should still land somewhere sensible rather than crash.
    bone_count = len(bones)
    for bone_id, bone in bones.items():
        if bone.parent < 0 or bone.parent >= bone_count:
            compose(bone_id, Matrix.Identity(4))

def pick_bone_lengths(bones: dict[int, BoneNode]) -> None:
    """Choose a display length per bone (cosmetic only - deformation is fully defined by the
    bind matrix). A bone with children extends toward its NEAREST child's head; a leaf inherits
    its parent's length; everything is clamped to MIN_BONE_LENGTH."""
    # Pass 1: bones with children - distance to the nearest child head.
    for bone in bones.values():
        if bone.children:
            head = bone.bind_arma_matrix.translation
            nearest = min((bones[child].bind_arma_matrix.translation - head).length for child in bone.children)
            bone.bone_length = max(nearest, MIN_BONE_LENGTH)

    # Pass 2: leaves - inherit the parent's length so finger tips / toe tips stay proportionate.
    for bone in bones.values():
        if not bone.children:
            if 0 <= bone.parent and bone.parent in bones:
                bone.bone_length = max(bones[bone.parent].bone_length * 0.5, MIN_BONE_LENGTH)
            else:
                bone.bone_length = DEFAULT_BONE_LENGTH

def compute_bones(borg: BORG) -> dict[int, BoneNode]:
    """End-to-end bone tree preparation: init, compose armature-space matrices, pick lengths."""
    bones = init_bones(borg)
    if not bones: return bones
    compose_armature_matrices(bones)
    pick_bone_lengths(bones)
    return bones

# ----------------------------------------
# ARMATURE CONSTRUCTION
# ----------------------------------------

def construct_blender_armature(borg: BORG, file_path: str) -> bpy.types.Object:
    """Build a Blender Armature object from parsed BORG data.

    Returns the created armature object so callers can reference it (e.g. to chain a model
    import that wants to parent meshes to the skeleton)."""
    print("\nConstructing Blender armature...")

    # --- COLLECTION MANAGEMENT ---
    collection_name = Path(file_path).stem
    if collection_name not in bpy.data.collections:
        target_collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(target_collection)
    else:
        target_collection = bpy.data.collections[collection_name]

    # --- ARMATURE CREATION ---
    armature_data = bpy.data.armatures.new(name="Armature")
    armature_obj  = bpy.data.objects.new(name=f"{collection_name}_skeleton", object_data=armature_data)

    armature_obj.show_in_front = True
    target_collection.objects.link(armature_obj)

    # --- BONE TREE PREP (swizzle, compose, lengths) ---
    bones = compute_bones(borg)
    if not bones:
        print("   WARNING: BORG has no bones - returning empty armature.")
        return armature_obj

    # Walk the tree depth-first for a deterministic creation order. Blender's edit_bones requires
    # the parent to exist before any child can be wired, so DFS-from-root guarantees that contract
    # regardless of how the source file ordered the bone definitions.
    bone_ids: list[int] = []

    def collect_dfs(bone_id: int) -> None:
        bone_ids.append(bone_id)
        for child in bones[bone_id].children:
            collect_dfs(child)

    bone_count = len(bones)
    for root_id, root_bone in bones.items():
        if root_bone.parent < 0 or root_bone.parent >= bone_count:
            collect_dfs(root_id)

    # --- ENTER EDIT MODE ---
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    edit_bones = armature_obj.data.edit_bones

    # --- PASS 1: create all edit bones AT THE BIND POSE ---
    # EditBone.matrix assignment sets head + roll from the 4x4 in one operation; setting length
    # afterwards extends the tail along the matrix's +Y. Because the matrix IS the bind pose, the
    # armature's rest pose equals the on-disk skeleton exactly - no pose-channel compensation.
    print(f"\nGenerating {len(bone_ids)} armature bones...")
    for bone_id in bone_ids:
        bone     = bones[bone_id]
        editbone = edit_bones.new(bone.name)
        bone.bl_bone_name = editbone.name           # Blender may suffix on name collision; capture the assigned name.
        editbone.use_connect = False                # See parenting note below.

        # Give the fresh bone a nonzero extent BEFORE assigning the matrix: a new edit bone is
        # zero-length, and the .length setter scales the (zero) head->tail vector, so setting an
        # explicit tail is the reliable way to establish extent. The matrix assignment then moves
        # head + orients the bone (roll included); the tail follows along the matrix's +Y at the
        # length we just gave it.
        editbone.tail   = (0.0, bone.bone_length, 0.0)
        editbone.matrix = bone.bind_arma_matrix

    # --- PASS 2: parenting (use_connect deliberately stays False) ---
    # Connect-mode would snap each child's head to its parent's tail, which destroys the bind
    # positions for any bone whose head doesn't coincide with the parent's tail (most of them).
    # Bones stay disconnected; the hierarchy is established purely via .parent.
    print("\nHandling bone parenting...")
    for bone_id in bone_ids:
        bone = bones[bone_id]
        if bone.parent < 0 or bone.parent >= len(bones): continue
        parent_bone = bones[bone.parent]
        edit_bones[bone.bl_bone_name].parent = edit_bones[parent_bone.bl_bone_name]

    # --- BACK TO OBJECT MODE ---
    bpy.ops.object.mode_set(mode='OBJECT')

    # --- BONE CUSTOM PROPERTIES (set on data.bones, NOT edit_bones, for cross-version safety) ---
    # The rename utilities iterate `arm.data.bones` looking for the `id` custom property to build
    # the bone-id-to-name map that powers the "Bone Indices to Names" / "Bone Names to Indices"
    # menu operators. Setting properties here on data.bones[] is guaranteed to work in object mode
    # regardless of Blender version.
    print("\nWriting bone custom properties...")
    data_bones = armature_obj.data.bones
    for bone_id in bone_ids:
        bone      = bones[bone_id]
        bone_def  = borg.bone_definitions[bone_id]
        data_bone = data_bones.get(bone.bl_bone_name)
        if data_bone is None:
            print(f"   WARNING: could not locate data.bone '{bone.bl_bone_name}' for custom property assignment.")
            continue
        data_bone["id"]           = bone_id
        data_bone["parent_index"] = bone_def["parent_index"]
        data_bone["body_part"]    = bone_def["body_part"]
        data_bone["bone_center"]  = list(bone_def["center"])
        data_bone["bone_size"]    = list(bone_def["size"])

    # --- X-AXIS MIRRORING (default OFF) ---
    # Symmetric editing CAN work for L/R bone pairs whose names follow a recognized convention
    # (`.L`/`.R`, `_l`/`_r`, etc.), but we leave the toggle to the user: Armature properties →
    # "Viewport Display" → "Skeleton" → "X-Axis Mirror". Since bones now sit at the raw bind pose,
    # L/R rolls mirror exactly as authored - any asymmetry you see is in the source skeleton.
    armature_obj.data.use_mirror_x = False
    try: armature_obj.pose.use_mirror_x = False
    except AttributeError: pass   # Older Blender versions may not expose pose.use_mirror_x.

    # --- METADATA ---
    armature_obj["game"]             = borg.game
    armature_obj["bone_count"]       = len(borg.bone_definitions)
    armature_obj["constraint_count"] = len(borg.bone_constraints)
    armature_obj["pose_bone_count"]  = len(borg.pose_bones)
    armature_obj["face_bone_count"]  = len(borg.face_bone_indices)

    print(f"\nARMATURE BUILT: {len(borg.bone_definitions)} bones, {len(borg.bone_constraints)} constraints, "
          f"{len(borg.pose_bones)} pose bones, {len(borg.face_bone_indices)} face bones")
    return armature_obj

# ----------------------------------------
# PUBLIC ENTRY POINT
# ----------------------------------------

def import_skeleton(self, context: bpy.types.Context, file_path: str, game: str) -> set[str]:
    """Import a Glacier 2 BORG (BoneRig) into Blender as an Armature object.

    Args:
        self:      The calling operator (used for self.report()).
        context:   Blender context.
        file_path: Absolute path to the .BORG file on disk.
        game:      One of GLACIER2_ABSOLUTION, GLACIER2_TRILOGY or GLACIER2_BOND. Determines
                   header layout and constraint encoding (Trilogy/Absolution = u8 type
                   discriminator, Bond = u16).
    """
    start_time = time.time()
    print(f"\nIMPORTING BORG SKELETON: {file_path}...\n")

    if not os.path.exists(file_path):
        self.report({'ERROR'}, f"BORG file not found: {file_path}")
        return {'CANCELLED'}

    if game not in (GLACIER2_ABSOLUTION, GLACIER2_TRILOGY, GLACIER2_BOND):
        self.report({'ERROR'}, f"Unsupported game type for BORG import: {game}")
        return {'CANCELLED'}

    try:
        borg = BORG(file_path, game)
    except Exception as exc:
        self.report({'ERROR'}, f"Failed to parse BORG: {exc}")
        return {'CANCELLED'}

    construct_blender_armature(borg, file_path)

    elapsed_time = time.time() - start_time
    minutes      = int(elapsed_time // 60)
    seconds      = elapsed_time % 60
    if minutes > 0: elapsed_str = f"{minutes} minute{'s' if minutes > 1 else ''} and {seconds:.2f} seconds"
    else: elapsed_str = f"{seconds:.2f} seconds"

    success_message = f'Successfully imported skeleton: "{Path(file_path).name}" in {elapsed_str}.'
    self.report({'INFO'}, success_message)
    print(f"\n{success_message}\n")

    return {'FINISHED'}
