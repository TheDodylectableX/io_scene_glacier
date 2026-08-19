# ------------------------------------------------
# MODEL HANDLER
#       Takes the parsed PRIM data and builds it
#       into Blender's scene as Mesh objects.
#       Handles every PRIM_SUBTYPE through one
#       general entry point so adding subtype-
#       specific behavior later is purely additive.
# ------------------------------------------------

from ..utilities import *

from ..data_format_parsers.prim import *
from ..data_format_parsers.clos import *

from ..data_format_parsers.prm import *
from ..data_format_parsers.gms import *

# ------------------------------------------------
# COLLECTION / NAMING HELPERS
# ------------------------------------------------

def ensure_collection(name: str, parent: Optional[bpy.types.Collection] = None) -> bpy.types.Collection:
    """Return an existing collection by name, or create + link a new one under `parent`.
    The lookup is SCOPED to `parent`'s children (not the global bpy.data.collections table) so
    generic names like "Mesh" / "Collision" can be reused under different file collections without
    colliding. A top-level collection (parent=None) is matched against the scene root's children."""
    siblings = parent.children if parent is not None else bpy.context.scene.collection.children
    if name in siblings: return siblings[name]
    new_coll = bpy.data.collections.new(name)
    siblings.link(new_coll)
    return new_coll

# ------------------------------------------------
# MATERIAL HANDLING
# ------------------------------------------------

def add_prim_materials(prim_record: dict, blender_obj: bpy.types.Object, file_stem: str, assign_material_colors: bool) -> None:
    """Attach a placeholder material to a single PRIM object based on its material_id.

    PRIM doesn't carry a name table for materials directly (material strings live in the parent
    resource entity, not in the .PRIM itself), so all we can mint at import time is a placeholder
    keyed by material_id for the user to rename or hook textures onto later.

    The file stem is part of the name because material_id is only unique WITHIN a file - id 2 in
    a hair PRIM has nothing to do with id 2 in a head PRIM. Since create_material reuses any
    existing datablock of the same name, a file-agnostic name would silently hand the second
    import the first one's material, dragging its colour and any wired-up textures along with it."""
    material_id   = prim_record["object_metadata"]["material_id"]
    material_name = f"{file_stem}_material_{material_id}"

    # create_material lives in utilities.py and handles the "exists vs create new" branching plus
    # the optional random color assignment for visual distinguishability.
    new_material = create_material(material_name, assign_material_colors)
    new_material["material_id"] = material_id
    new_material["source_file"] = file_stem
    add_material(new_material, blender_obj)

# ------------------------------------------------
# VERTEX COLOR HANDLING
# ------------------------------------------------

def apply_vertex_colors(blender_obj: bpy.types.Object, vertex_colors: list[tuple]) -> None:
    """Apply vertex colors from the model."""
    if not vertex_colors: return

    mesh = blender_obj.data
    if not mesh.vertex_colors: mesh.vertex_colors.new(name="Color")
    color_layer = mesh.vertex_colors.active

    for loop in mesh.loops: color_layer.data[loop.index].color = vertex_colors[loop.vertex_index]

# ------------------------------------------------
# WEIGHT HANDLING
# ------------------------------------------------

def apply_bone_weights(blender_obj: bpy.types.Object, bone_weights: list[list[int]], bone_local_indices: list[list[int]], bone_info: Optional[dict], bone_palette: Optional[list[int]], game: str = "") -> None:
    """Credit: REDxEYE, Modified by Dodylectable | Insert the skeletal model's weights.

      WoA:        6 weights per vertex; joint values are DIRECT BORG bone indices.
      007FL:      4 weights per vertex; parser unpacks three 10-bit indices from the packed index
                  variable and pulls the 4th from the vertex position's W component (already decoded).
      Absolution: 4 weights per vertex; joint values are boneRemapValues. For each influence with a
                  NONZERO weight, the global bone index is GetBoneIndex(boneRemapValue / 3) - i.e.
                  the slot in BoneInfo.bone_remap whose value equals boneRemapValue/3 (mirrors IOI's
                  deserializer). Influences with zero weight keep the raw value (never referenced).

    bone_palette is accepted for signature stability and round-trip/export use, but is NOT consulted
    during import (it's the engine's runtime per-primary-bone vertex connectivity, not a joint LUT)."""
    if not bone_weights or not bone_local_indices:
        print("  No weights/joints in record; skipping weight import.")
        return

    # Absolution resolves boneRemapValues -> global bone indices via BoneInfo.bone_remap.
    is_absolution = (game == GLACIER2_ABSOLUTION)
    bone_remap_table: Optional[list[int]] = bone_info.get("bone_remap") if (is_absolution and bone_info) else None

    def resolve(remap_value: int) -> int:
        if bone_remap_table is None:
            return remap_value
        target = remap_value // 3
        try:
            return bone_remap_table.index(target)
        except ValueError:
            return remap_value  # unmapped; keep raw rather than silently drop the influence

    # Map of vertex groups, keyed by group name
    final_weight_map: dict[str, bpy.types.VertexGroup] = {}
    max_bone_seen = -1

    # For every vertex (loop through bone indices and bone weights in lockstep)
    for vertex_index, (id_group, weight_group) in enumerate(zip(bone_local_indices, bone_weights)):
        for idx, wgt in zip(id_group, weight_group):
            if wgt <= 0: continue

            bone_index = resolve(idx) if is_absolution else idx
            group_name = f"bone_{bone_index}"
            if group_name not in final_weight_map: final_weight_map[group_name] = blender_obj.vertex_groups.new(name=group_name)
            final_weight_map[group_name].add([vertex_index], wgt / 255.0, 'REPLACE')
            if bone_index > max_bone_seen: max_bone_seen = bone_index

    print(f"  Vertex groups created: {len(final_weight_map)}  |  max bone index: {max_bone_seen}")

# ------------------------------------------------
# COLLISION HANDLING
# ------------------------------------------------

def build_collision_object(prim_record: dict, parent_collection: bpy.types.Collection, source_mesh_name: str) -> Optional[bpy.types.Object]:
    """Build a wireframe object representing the BoxColi (bullet/projectile collision boxes).

    BoxColi entries are quantized into the mesh's bounding box: Each min/max is 0/255 encoding a fraction of the mesh AABB.
    We dequantize back to world space using the object's bbox_min/bbox_max stored in the PRIM_OBJECT header.

    Returns the created object (or None if there's nothing to build) so callers can decide whether to parent it to anything."""
    collision = prim_record.get("collision")
    if not collision or not collision.get("entries"): return None

    object_meta = prim_record["object_metadata"]
    bbox_min = Vector(object_meta["bbox_min"])
    bbox_max = Vector(object_meta["bbox_max"])
    bbox_extent = bbox_max - bbox_min

    # Each BoxColi entry becomes 8 verts + 12 edges (cube wireframe). We accumulate all of them into a single mesh per source mesh to keep the outliner clean.
    all_verts: list[tuple[float, float, float]] = []
    all_edges: list[tuple[int, int]] = []

    for entry in collision["entries"]:
        local_min = entry["box_min"]
        local_max = entry["box_max"]

        # Dequantize: local fraction × extent + offset = world position.
        world_min = Vector((
            bbox_min.x + (local_min[0] / 255.0) * bbox_extent.x,
            bbox_min.y + (local_min[1] / 255.0) * bbox_extent.y,
            bbox_min.z + (local_min[2] / 255.0) * bbox_extent.z,
        ))
        world_max = Vector((
            bbox_min.x + (local_max[0] / 255.0) * bbox_extent.x,
            bbox_min.y + (local_max[1] / 255.0) * bbox_extent.y,
            bbox_min.z + (local_max[2] / 255.0) * bbox_extent.z,
        ))

        # Emit 8 corners of the AABB in a canonical order.
        base = len(all_verts)
        all_verts.extend([
            (world_min.x, world_min.y, world_min.z),  # 0  - - -
            (world_max.x, world_min.y, world_min.z),  # 1  + - -
            (world_max.x, world_max.y, world_min.z),  # 2  + + -
            (world_min.x, world_max.y, world_min.z),  # 3  - + -
            (world_min.x, world_min.y, world_max.z),  # 4  - - +
            (world_max.x, world_min.y, world_max.z),  # 5  + - +
            (world_max.x, world_max.y, world_max.z),  # 6  + + +
            (world_min.x, world_max.y, world_max.z),  # 7  - + +
        ])
        # 12 edges of a cube (bottom 4, top 4, vertical 4)
        all_edges.extend([
            (base+0, base+1), (base+1, base+2), (base+2, base+3), (base+3, base+0),
            (base+4, base+5), (base+5, base+6), (base+6, base+7), (base+7, base+4),
            (base+0, base+4), (base+1, base+5), (base+2, base+6), (base+3, base+7),
        ])

    if not all_verts: return None

    coli_name = f"{source_mesh_name}_Collision"
    coli_mesh = bpy.data.meshes.new(name=coli_name)
    coli_obj  = bpy.data.objects.new(coli_name, coli_mesh)
    parent_collection.objects.link(coli_obj)

    coli_mesh.from_pydata(all_verts, all_edges, [])
    coli_mesh.update()

    # Display as wire so it doesn't obscure the mesh underneath.
    coli_obj.display_type = 'WIRE'
    coli_obj.show_in_front = True
    coli_obj["chunk_count"]         = collision["chunk_count"]
    coli_obj["triangles_per_chunk"] = collision["triangles_per_chunk"]

    return coli_obj

# ------------------------------------------------
# CORE MESH CONSTRUCTION
# ------------------------------------------------

def construct_blender_mesh(prim_record: dict, mesh_index: int, file_stem: str, target_collection: bpy.types.Collection, use_custom_normals: bool, game: str = "") -> bpy.types.Object:
    """Build a single PRIM_OBJECT into a Blender Mesh object.

    This is subtype-agnostic by design: build_mesh_record in the parser packs every subtype's output into the same shape
    So this function only has to ask "do I have normals? UVs? vertex colors? bone weights?" and act accordingly.

    Adding a new subtype later means:
      1. Add a parse_bond_attrs_<subtype>() (or trilogy equivalent) method to PRIM.
      2. Register it in the subtype dispatch in PRIM.parse_vertex_attributes_bond / _trilogy.
      3. Done - This function picks up the new data automatically as long as the record shape is preserved."""
    sub_type      = prim_record["sub_type"]
    sub_type_name = prim_record["sub_type_name"]
    vertex_count  = prim_record["vertex_count"]
    index_count   = prim_record["index_count"]
    lod_index     = prim_record.get("lod_index", 0)

    # Naming scheme: <PRIMFileName>_<MeshIndex>_LOD_<LODIndex>_<TYPE>
    mesh_name = f"{file_stem}_{mesh_index:02d}_LOD_{lod_index}_{sub_type_name}"
    print(f"\nBuilding Mesh {mesh_index + 1} ({sub_type_name}, LOD {lod_index}, {vertex_count} verts, {index_count} indices)...")

    # --- BLENDER MESH + OBJECT ---
    mesh = bpy.data.meshes.new(name=mesh_name)
    obj  = bpy.data.objects.new(mesh_name, mesh)
    target_collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    # --- GEOMETRY ---
    positions = prim_record["positions"]
    triangles = prim_record["triangles"]
    normals   = prim_record["normals"]

    # NOTE on triangle winding: Dody's testing confirmed that BOTH games' on-disk triangle data
    # already lines up with Blender's CCW convention - no per-triangle reversal needed. The parser
    # passes triangles through unchanged.
    mesh.from_pydata(positions, [], triangles)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if not IS_BLENDER_4_1: mesh.use_auto_smooth = True  # 4.1+ removed this flag; smooth is per-poly now.

    # --- NORMALS ---
    # The `use_custom_normals` checkbox mirrors Blender's manual "Add Custom Split Normals Data"
    # mesh operator: when CHECKED we lock in the file's per-vertex NTB as custom split normals so
    # Blender renders exactly what Glacier authored; when UNCHECKED we leave normals alone and let
    # Blender compute them from geometry (so the mesh ends up with smooth-shaded auto normals only).
    # No mid-state: either we apply the file normals end-to-end, or we don't apply them at all.
    if use_custom_normals is False and normals:
        mesh.normals_split_custom_set_from_vertices(normals)
        print("  Built mesh and applied original Glacier normals as custom split normals.")
    else: print("  Built mesh with Blender-computed normals (custom normals disabled or unavailable).")

    # --- UV CHANNELS ---
    uv_channels = prim_record["uv_channels"] or []
    for channel_index, channel_data in enumerate(uv_channels):
        layer_name = f"UV_{channel_index + 1:02d}"
        uv_layer = mesh.uv_layers.new(name=layer_name)
        print(f"  Inserting UV channel '{layer_name}' ({len(channel_data)} entries)...")
        for loop in mesh.loops: uv_layer.data[loop.index].uv = channel_data[loop.vertex_index]

    # --- TANGENTS ---
    # Blender computes tangents from UVs + normals (mesh.calc_tangents) so we don't push the
    # decoded Glacier tangents directly. The decoded tangents/bitangents are preserved on the
    # parser record though, so an exporter can round-trip them later without re-deriving.
    if uv_channels and normals:
        try:
            mesh.calc_tangents()
            print("  Calculated tangents from UV+normal data.")
        except RuntimeError as exc: print(f"  WARNING: calc_tangents failed ({exc}); skipping.")

    # --- VERTEX COLORS ---
    vertex_colors = prim_record.get("vertex_colors")
    if vertex_colors:
        print("  Inserting vertex color attribute data...")
        apply_vertex_colors(obj, vertex_colors)

    # --- BONE WEIGHTS (weighted submeshes only) ---
    if sub_type == PRIM_SUBTYPE_WEIGHTED and prim_record["bone_weights"]:
        print("  Inserting bone weights and vertex groups...")
        apply_bone_weights(obj, prim_record["bone_weights"], prim_record["bone_local_indices"], prim_record.get("bone_info"), prim_record.get("bone_palette"), game)

    # --- METADATA on the object (useful for round-trip + debugging) ---
    obj["sub_type"]      = sub_type
    obj["sub_type_name"] = sub_type_name
    obj["material_id"]   = prim_record["object_metadata"]["material_id"]
    obj["lod_mask"]      = prim_record["object_metadata"]["lod_mask"]
    obj["variant_id"]    = prim_record["object_metadata"]["variant_id"]
    obj["wire_color"]    = prim_record["object_metadata"]["wire_color"]
    obj["bbox_min"]      = list(prim_record["object_metadata"]["bbox_min"])
    obj["bbox_max"]      = list(prim_record["object_metadata"]["bbox_max"])
    obj["cloth_id"]      = prim_record["cloth_id_raw"]

    # --- GLACIER MESH PROPERTIES (editable export metadata; preset from the file) ---
    # Mirrors the PRIM header/subtype/flags into the Object-Properties panel defined in __init__.
    # Wrapped defensively so a headless / unit-test context without the add-on registered still
    # builds meshes (the panel pointer only exists once register_plugin() has run).
    meta = prim_record["object_metadata"]
    glacier = getattr(obj, "glacier_mesh", None)
    if glacier is not None:
        # EnumProperty backed by integer-strings - assign defensively (a value outside the
        # declared items would raise; fall back to the default rather than abort the import).
        try: glacier.record_type = str(meta.get("record_type", 2))
        except TypeError: pass
        try: glacier.sub_type = str(sub_type)
        except TypeError: pass
        flags = meta.get("properties", {})
        glacier.is_x_axis_locked    = flags.get("is_x_axis_locked",    False)
        glacier.is_y_axis_locked    = flags.get("is_y_axis_locked",    False)
        glacier.is_z_axis_locked    = flags.get("is_z_axis_locked",    False)
        glacier.is_high_resolution  = flags.get("is_high_resolution",  False)
        glacier.has_ps3_edge        = flags.get("has_ps3_edge",        False)
        glacier.use_color_1         = flags.get("use_color_1",         False)
        glacier.has_no_physics_prop = flags.get("has_no_physics_prop", False)
        glacier.material_id         = meta.get("material_id", 0)
        glacier.lod_mask            = meta.get("lod_mask", 0)
        glacier.variant_id          = meta.get("variant_id", 0)

        # clothID split into its two halves. The high word is the mesh class (1 = standard,
        # 2 = hair card, which is what widens 007FL's vertex record to carry a second UV set);
        # the low word is the cloth blob index, nonzero only when the mesh has simulated cloth.
        # Absolution and WoA meshes generally leave the high word at 0, and that 0 has to survive
        # a round trip - hence the explicit "None" class rather than defaulting these to Standard.
        cloth_raw  = prim_record.get("cloth_id_raw", 0)
        mesh_class = (cloth_raw >> 16) & 0xFFFF
        try: glacier.mesh_class = str(mesh_class if mesh_class in (0, 1, 2) else 0)
        except TypeError: pass
        glacier.cloth_blob_index = cloth_raw & 0xFFFF

        # Color1 tint widget: sourced from the object's wire_color (the field that reads
        # 0xFFFFFFFF / white on most meshes; color_1 is usually 0). Stored as an int32 whose
        # low bytes are RGBA; decode via convert_vertex_color and drop alpha for the RGB widget.
        wire = meta.get("wire_color", 0xFFFFFFFF) & 0xFFFFFFFF
        cR =  wire        & 0xFF
        cG = (wire >> 8)  & 0xFF
        cB = (wire >> 16) & 0xFF
        cA = (wire >> 24) & 0xFF
        R, G, B, A = convert_vertex_color(cR, cG, cB, cA)
        glacier.color_1 = (R, G, B)

    mesh.update()
    return obj

# ------------------------------------------------
# PUBLIC ENTRY POINT
# ------------------------------------------------

def import_prim_model(self, context: bpy.types.Context, file_path: str, game: str, use_custom_normals: bool = False, assign_material_colors: bool = True, import_collisions: bool = False, import_lods: bool = True) -> set[str]:
    """Import a Glacier 2 PRIM (RenderPrimitive) file into Blender.

    Args:
        self:                      Calling operator (for self.report()).
        context:                   Blender context.
        file_path:                 Absolute path to the .PRIM file.
        game:                      GLACIER2_ABSOLUTION, GLACIER2_TRILOGY or GLACIER2_BOND.
        use_custom_normals:        If True, let Blender compute normals from geometry rather
                                   than applying the original ones. Default False (preserve
                                   on-disk normals for fidelity).
        assign_material_colors:    If True, give each placeholder material a distinct random
                                   color to visually separate submeshes in the viewport.
        import_collisions:         If True, additionally build wireframe BoxColi collision
                                   objects in a child 'Collision' sub-collection.
        import_lods:               If True, import every LOD grouped into 'LOD #' sub-collections
                                   under 'Mesh'. If False, build only LOD0 meshes (LOD mask 0x01)
                                   and cloth/physics meshes (LOD mask 0xFF).

    Returns:
        Blender operator result set.
    """
    start_time = time.time()
    print(f"\nIMPORTING PRIM MODEL: {file_path}...\n")

    if not os.path.exists(file_path):
        self.report({'ERROR'}, f"PRIM file not found: {file_path}")
        print(f"Cannot import model; file was not found at: {file_path}")
        return {'CANCELLED'}

    if game not in (GLACIER2_ABSOLUTION, GLACIER2_TRILOGY, GLACIER2_BOND):
        self.report({'ERROR'}, f"Unsupported game type for PRIM import: {game}")
        print(f"Invalid game type constant for PRIM: {game}")
        return {'CANCELLED'}

    # --- PARSE ---
    try:
        prim = PRIM(file_path, game)
    except Exception as exc:
        self.report({'ERROR'}, f"Failed to parse PRIM: {exc}")
        print(f"PRIM parse failure: {exc}")
        return {'CANCELLED'}

    if not prim.objects:
        self.report({'WARNING'}, "PRIM contains no objects.")
        print("PRIM has zero parsed objects; nothing to build.")
        return {'CANCELLED'}

    # --- COLLECTION SETUP ---
    # Hierarchy: <PRIMFileName> -> "Mesh"      -> "LOD #(i)" (Per-LOD mesh sub-collections)
    #                           -> "Collision" -> "LOD #(i)" (Per-LOD collision sub-collections)
    # Meshes always live under "Mesh"; Each mesh drops into the "LOD #(i)" child matching its LOD index
    # Collisions (when enabled) drop into the matching "LOD #(i)" under "Collision". LOD sub-collections are created lazily as indices appear.
    file_stem = Path(file_path).stem
    file_collection = ensure_collection(file_stem)
    mesh_collection = ensure_collection("Mesh", parent=file_collection)
    collision_collection = ensure_collection("Collision", parent=file_collection) if import_collisions else None
    mesh_lod_collections: dict[int, bpy.types.Collection] = {}
    collision_lod_collections: dict[int, bpy.types.Collection] = {}

    def mesh_lod_collection_for(lod_index: int) -> bpy.types.Collection:
        if lod_index not in mesh_lod_collections: mesh_lod_collections[lod_index] = ensure_collection(f"LOD #{lod_index}", parent=mesh_collection)
        return mesh_lod_collections[lod_index]

    def collision_lod_collection_for(lod_index: int) -> bpy.types.Collection:
        if lod_index not in collision_lod_collections: collision_lod_collections[lod_index] = ensure_collection(f"LOD #{lod_index}", parent=collision_collection)
        return collision_lod_collections[lod_index]

    # --- BUILD EACH PRIM_OBJECT ---
    # LOD filtering (when import_lods is False): keep only LOD0 meshes (mask 0x01) and
    # cloth/physics meshes (mask 0xFF); skip everything else. Masks are validated against the
    # parsed record so we never build the lower-detail LOD chain.
    built_objects: list[bpy.types.Object] = []
    for mesh_index, prim_record in enumerate(prim.objects):
        lod_mask  = prim_record["object_metadata"].get("lod_mask", 0)
        lod_index = prim_record.get("lod_index", 0)

        if not import_lods and lod_mask not in (0x01, 0xFF):
            print(f"  Skipping Mesh {mesh_index} (LOD mask 0x{lod_mask:02X}); 'Import LODs' is off.")
            continue

        target_collection = mesh_lod_collection_for(lod_index)
        blender_obj = construct_blender_mesh(prim_record, mesh_index, file_stem, target_collection, use_custom_normals, game)
        built_objects.append(blender_obj)

        # Attach a placeholder material per submesh.
        add_prim_materials(prim_record, blender_obj, file_stem, assign_material_colors)

        # Optional collision import - one wireframe object per source mesh, nested under the
        # matching "Collision -> LOD #<i>" sub-collection so meshes and their collisions filter
        # in parallel. The collision object inherits the mesh's LOD-aware name with a _Collision
        # suffix.
        if import_collisions and collision_collection is not None:
            coli_target = collision_lod_collection_for(lod_index)
            coli_obj = build_collision_object(prim_record, coli_target, blender_obj.name)
            if coli_obj is not None:
                coli_obj.parent = blender_obj
                # Clear the parent inverse so the collision wireframe sits in the same local space as the mesh (no double-transform).
                coli_obj.matrix_parent_inverse = blender_obj.matrix_world.inverted()

    prim.blender_objects = built_objects

    # --- PER-FILE METADATA on the scene ---
    # Capture the BORG resource index so a follow-up "import skeleton" workflow can resolve
    # the reference. We store it scene-wide because PRIMs reference their rig by hash, and the
    # user's existing skeleton_path panel is the natural place to surface this.
    bone_rig_resource_index = prim.object_header["bone_rig_resource_index"]
    if bone_rig_resource_index != BORG_RESOURCE_NONE: bpy.context.scene["last_prim_bone_rig_index"] = bone_rig_resource_index

    print(f"\nMODEL BUILDING COMPLETE! ({len(built_objects)} objects)")

    # --- TIMING REPORT ---
    elapsed_time = time.time() - start_time
    minutes      = int(elapsed_time // 60)
    seconds      = elapsed_time % 60
    if minutes > 0: elapsed_str = f"{minutes} minute{'s' if minutes > 1 else ''} and {seconds:.2f} seconds"
    else: elapsed_str = f"{seconds:.2f} seconds"

    success_message = f'Successfully imported model: "{Path(file_path).name}" in {elapsed_str}.'
    self.report({'INFO'}, success_message)
    print(success_message)

    return {'FINISHED'}

# =========================================================================================================
# CLOAKWORKS SHROUD CLOTH (.CLOS) HANDLER
#       Hitman: Absolution cloth resources. The parser (data_format_parsers/clos.py) walks the
#       CloakWorks reflection tree and hands back, per cloth piece, the high-res render surface:
#       vertices reconstructed from the simulation grid through the binding math, the render index
#       buffer, on-disk UVs, per-vertex normals and bone weights. This handler just builds those
#       into Blender - one mesh object per cloth piece, grouped under a file-named collection,
#       mirroring the PRIM importer's structure.
# =========================================================================================================

def construct_cloth_mesh(piece, piece_index: int, file_stem: str, target_collection: bpy.types.Collection, assign_material_colors: bool) -> bpy.types.Object:
    """Build a single CloakWorks cloth piece into a Blender Mesh object.

    A cloth piece is the high-res RENDER surface: render-vertex rest positions reconstructed from the
    simulation grid via barycentric binding, the render index buffer (index32s), and the on-disk
    per-vertex texcoords. There are no baked normals in the file - cloth normals are simulated at
    runtime via CloakWorks' ClothNormalsUpdater - so we let Blender compute smooth normals from
    geometry, which is the faithful rest-pose result."""
    # Naming scheme: <CLOSFileName>_<PieceIndex>_<ShapeKind>
    shape_kind = piece.shape_kind or "Cloth"
    mesh_name = f"{file_stem}_{piece_index:02d}_{shape_kind}"
    print(f"\nBuilding cloth piece {piece_index + 1} ({shape_kind}, sim grid {piece.num_rows}x{piece.num_columns}, render mesh {len(piece.positions)} vertices, {len(piece.triangles)} triangles)...")

    # --- BLENDER MESH + OBJECT ---
    mesh = bpy.data.meshes.new(name=mesh_name)
    obj  = bpy.data.objects.new(mesh_name, mesh)
    target_collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    # --- GEOMETRY ---
    # The render index buffer is a triangle list; positions are the reconstructed render vertices.
    mesh.from_pydata(piece.positions, [], piece.triangles)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if not IS_BLENDER_4_1: mesh.use_auto_smooth = True  # 4.1+ removed this flag; smooth is per-poly now.

    # Validate BEFORE the custom normals go on and never after: validate() prunes data it considers invalid and will happily throw away the custom split-normal layer along with it leaving the mesh silently shaded from geometry instead of from the file.
    mesh.validate()

    # --- CUSTOM NORMALS ---
    # The binding math yields true per-vertex normals (the ones the engine shades with at rest) so apply them as custom split normals instead of letting Blender guess from geometry.
    if piece.normals and len(piece.normals) == len(piece.positions): mesh.normals_split_custom_set_from_vertices([Vector(normal) for normal in piece.normals])

    # --- UVS ---
    # Per-render-vertex texcoords straight from the file (texCoords array).
    if piece.uvs:
        uv_layer = mesh.uv_layers.new(name="UVMap")
        # UVs are per-vertex; map them onto per-loop via each loop's vertex index.
        for loop in mesh.loops:
            u, v = piece.uvs[loop.vertex_index] if loop.vertex_index < len(piece.uvs) else (0.0, 0.0)
            # Glacier/CloakWorks texture space is top-left origin; flip V for Blender's bottom-left.
            uv_layer.data[loop.index].uv = (u, 1.0 - v)

    # --- BONE WEIGHTS ---
    # SkinningTransform streams, blended onto the render vertices by the parser. Vertex groups are
    # named by the skeleton bone so parenting to an imported BORG armature binds directly.
    for bone_name, vertex_weights in piece.bone_weights.items():
        group = obj.vertex_groups.new(name=bone_name)
        for vertex_index, weight in enumerate(vertex_weights):
            if weight > 0.0: group.add([vertex_index], weight, 'REPLACE')

    # --- MATERIAL ---
    material = create_material(f"{mesh_name}_Material", assign_material_colors)
    add_material(material, obj)

    mesh.update()
    print(f"  Built cloth mesh '{mesh_name}'.")
    return obj

def import_cloth_model(self, context: bpy.types.Context, file_path: str, assign_material_colors: bool = True) -> set[str]:
    """Import a CloakWorks Shroud cloth (.CLOS) file into the scene, format is Absolution-only so there is no per-file game selection - The format is fixed."""
    start_time = time.time()
    file_stem = Path(file_path).stem

    # --- PARSE ---
    try:
        cloth = CLOS(file_path)
        cloth.parse_cloth_file()
    except Exception as error:
        message = f'Failed to parse CloakWorks cloth "{Path(file_path).name}": {error}'
        self.report({'ERROR'}, message)
        print(message)
        return {'CANCELLED'}

    if not cloth.pieces:
        message = f'No cloth pieces found in "{Path(file_path).name}".'
        self.report({'WARNING'}, message)
        print(message)
        return {'CANCELLED'}

    # --- COLLECTION ---
    # Top collection is the .CLOS file name (matching the other importers). When a file holds more
    # than one cloth piece, each piece gets its own sub-collection named after its node so the
    # outliner stays organised; a single-piece file drops straight into the file collection.
    file_collection = ensure_collection(file_stem)
    multiple_pieces = len(cloth.pieces) > 1

    # --- BUILD ---
    built_objects: list[bpy.types.Object] = []
    for piece_index, piece in enumerate(cloth.pieces):
        if multiple_pieces: piece_collection = ensure_collection(piece.name, parent=file_collection)
        else: piece_collection = file_collection
        obj = construct_cloth_mesh(piece, piece_index, file_stem, piece_collection, assign_material_colors)
        built_objects.append(obj)

    print(f"\nCLOTH BUILDING COMPLETE! ({len(cloth.pieces)} cloth piece(s))")

    # --- TIMING REPORT ---
    elapsed_time = time.time() - start_time
    minutes      = int(elapsed_time // 60)
    seconds      = elapsed_time % 60
    if minutes > 0: elapsed_str = f"{minutes} minute{'s' if minutes > 1 else ''} and {seconds:.2f} seconds"
    else: elapsed_str = f"{seconds:.2f} seconds"

    success_message = f'Successfully imported cloth: "{Path(file_path).name}" in {elapsed_str}.'
    self.report({'INFO'}, success_message)
    print(success_message)

    return {'FINISHED'}

# =====================================================================================================================================================
# =====================================================================================================================================================
#
#   GLACIER 1
#       Everything below handles IO Interactive's FIRST engine generation - Hitman 2: Silent
#       Assassin, Hitman: Contracts, Hitman: Blood Money (and Mini Ninjas) and Freedom Fighters.
#
#       The Glacier 1 pipeline differs from Glacier 2 in one structural way that shapes all of
#       this code: a .PRM holds geometry ONLY, with every primitive sitting at the world origin.
#       Placement lives in the companion .GMS mission script. Import a .PRM on its own and the
#       whole level stacks up on top of itself; pair it with its .GMS and everything lands where
#       the designers put it - now with rotation, not just translation.
#
# =====================================================================================================================================================
# =====================================================================================================================================================

# ------------------------------------------------
# MATERIAL HANDLING
# ------------------------------------------------

def add_prm_material(blender_obj: bpy.types.Object, mesh_index: int, file_stem: str, assign_material_colors: bool) -> None:
    """Attach a UNIQUE placeholder material to a single Glacier 1 submesh.

    The .PRM carries no material name table (material strings live in the GMS resource tables,
    not the geometry container), so we mint a placeholder the user can rename / wire textures
    onto later.

    Uniqueness matters: create_material does a bpy.data.materials.get(name) first, so a shared
    name would hand every submesh the SAME material datablock - one slot, one colour, shared
    across the whole import. Keying on mesh_index gives each primitive its own datablock and,
    when assign_material_colors is on, its own distinguishing random colour. The file stem is in
    the name for the same reason it is on the PRIM side: mesh_index only means anything within
    one file, so without it a second import would inherit the first file's materials wholesale."""
    material = create_material(f"PRM_{file_stem}_material_{mesh_index:03d}", assign_material_colors)
    material["mesh_index"]  = mesh_index
    material["source_file"] = file_stem
    add_material(material, blender_obj)

# ------------------------------------------------
# WEIGHT HANDLING
# ------------------------------------------------

def apply_glacier1_bone_weights(blender_obj: bpy.types.Object, bone_weights: list[list[float]], bone_indices: list[list[int]], bone_names: Optional[list[str]] = None) -> None:
    """Insert vertex groups for a skinned Glacier 1 submesh.

    Only Blood Money's 52-byte vertex carries skinning in the level containers, and it stores
    THREE explicit float weights with a fourth implied as (1 - sum). We do not synthesise that
    fourth influence: its bone index is not stored alongside the other three, so inventing one
    would be a guess rather than a decode.

    Unlike the Glacier 2 path these weights arrive as floats already in the [0, 1] range, so
    there is no /255 normalisation step. Bone indices have already been divided back down by the
    parser (Glacier stores them pre-multiplied by 3).

    When `bone_names` is supplied (Blood Money, where the skeleton IS recovered) the vertex groups
    are named after the actual bones, so they bind straight onto the armature built from the same
    skeleton. Without it (classic games, skeleton not yet recovered) they fall back to `bone_<n>`
    numeric names and are created unparented - still correct, just not yet armature-linked."""
    if not bone_weights or not bone_indices:
        print("  No weights/joints in record; skipping weight import.")
        return

    weight_groups: dict[str, bpy.types.VertexGroup] = {}
    highest_bone_seen = -1

    def group_name_for(bone_index: int) -> str:
        """Bone's real name when the skeleton is known, else a stable numeric fallback."""
        if bone_names is not None and 0 <= bone_index < len(bone_names): return bone_names[bone_index]
        return f"bone_{bone_index}"

    for vertex_index, (index_group, weight_group) in enumerate(zip(bone_indices, bone_weights)):
        for bone_index, weight in zip(index_group, weight_group):
            if weight <= 0.0: continue

            group_name = group_name_for(bone_index)
            if group_name not in weight_groups: weight_groups[group_name] = blender_obj.vertex_groups.new(name=group_name)
            weight_groups[group_name].add([vertex_index], weight, 'REPLACE')
            if bone_index > highest_bone_seen: highest_bone_seen = bone_index

    print(f"  Vertex groups created: {len(weight_groups)}  |  max bone index: {highest_bone_seen}")

# ------------------------------------------------
# SKELETON / ARMATURE CONSTRUCTION (Blood Money)
# ------------------------------------------------

def construct_glacier1_armature(skeleton: dict, model_chunk: int, file_stem: str, target_collection: bpy.types.Collection) -> Optional[bpy.types.Object]:
    """Build a Blender armature from one decoded Blood Money skeleton.

    The parser hands us, per bone: a name, a parent index, and a LOCAL transform (quaternion +
    position). We compose each bone's local transform down its parent chain into a world (bind)
    matrix and drop the bone in at that pose, exactly as the Glacier 2 BORG path does - the only
    difference is that Glacier 1 gives us local quats to compose rather than pre-baked globals.

    Returns the armature object (so the caller can parent weighted meshes to it), or None if the
    skeleton is degenerate."""
    bone_count = skeleton["bone_count"]
    names      = skeleton["names"]
    parents    = skeleton["parents"]
    rotations  = skeleton["local_rotations"]   # (x, y, z, w) per bone
    positions  = skeleton["local_positions"]   # (x, y, z) per bone
    if bone_count <= 0: return None

    print(f"\nConstructing armature for model chunk #{model_chunk} ({bone_count} bones)...")

    # --- COMPOSE LOCAL -> WORLD BIND MATRICES ---
    # global(bone) = global(parent) @ local(bone). Parents always precede children in these files,
    # but we resolve recursively with memoisation so any ordering is safe.
    local_matrices: list[Matrix] = []
    for bone in range(bone_count):
        qx, qy, qz, qw = rotations[bone]
        translation = Vector(positions[bone])
        # mathutils.Quaternion takes (w, x, y, z).
        rotation_matrix = Quaternion((qw, qx, qy, qz)).to_matrix().to_4x4()
        rotation_matrix.translation = translation
        local_matrices.append(rotation_matrix)

    world_matrices: list[Optional[Matrix]] = [None] * bone_count

    def compose(bone: int) -> Matrix:
        cached = world_matrices[bone]
        if cached is not None: return cached
        parent = parents[bone]
        if parent < 0 or parent >= bone_count or parent == bone:
            world_matrices[bone] = local_matrices[bone].copy()
        else:
            world_matrices[bone] = compose(parent) @ local_matrices[bone]
        return world_matrices[bone]

    for bone in range(bone_count): compose(bone)

    # --- ARMATURE OBJECT ---
    armature_data = bpy.data.armatures.new(name=f"{file_stem}_skeleton_{model_chunk:04d}")
    armature_obj  = bpy.data.objects.new(name=f"{file_stem}_skeleton_{model_chunk:04d}", object_data=armature_data)
    armature_obj.show_in_front = True
    target_collection.objects.link(armature_obj)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_obj.data.edit_bones

    # --- PASS 1: create every bone at its world bind pose ---
    # Blender may suffix duplicate names, so capture the assigned name per bone for the parenting
    # pass and for vertex-group binding. A short default length keeps the bone visible; the matrix
    # assignment orients it.
    assigned_names: list[str] = []
    for bone in range(bone_count):
        edit_bone = edit_bones.new(names[bone])
        assigned_names.append(edit_bone.name)
        edit_bone.use_connect = False
        edit_bone.tail = (0.0, GLACIER1_ARMATURE_BONE_LENGTH, 0.0)
        edit_bone.matrix = world_matrices[bone]

    # --- PASS 2: parenting (disconnected, so bind heads are preserved) ---
    for bone in range(bone_count):
        parent = parents[bone]
        if parent < 0 or parent >= bone_count or parent == bone: continue
        edit_bones[assigned_names[bone]].parent = edit_bones[assigned_names[parent]]

    bpy.ops.object.mode_set(mode='OBJECT')

    # --- BONE CUSTOM PROPERTIES (for round-trip + rename utilities) ---
    data_bones = armature_obj.data.bones
    for bone in range(bone_count):
        data_bone = data_bones.get(assigned_names[bone])
        if data_bone is None: continue
        data_bone["id"]           = bone
        data_bone["parent_index"] = parents[bone]

    armature_obj["model_chunk"] = model_chunk
    armature_obj["bone_count"]  = bone_count

    # Expose the exact (possibly-suffixed) bone names so weighted meshes bind to matching groups.
    skeleton["assigned_names"] = assigned_names
    return armature_obj

# ------------------------------------------------
# CORE MESH CONSTRUCTION
# ------------------------------------------------

def construct_glacier1_mesh(prm_record: dict, mesh_index: int, file_stem: str, target_collection: bpy.types.Collection, use_custom_normals: bool) -> bpy.types.Object:
    """Build a single Glacier 1 submesh into a Blender Mesh object.

    Container-agnostic: the classic anchor games and Blood Money's heap both funnel through the
    same record shape, so this only asks "do I have normals? UVs? colours? weights?" and acts. It
    mirrors the Glacier 2 `construct_blender_mesh` beat-for-beat (geometry -> normals -> UVs ->
    tangents -> colours -> weights -> metadata) so the two engines behave identically in Blender;
    the differences are only in what the record happens to carry."""
    vertex_count  = prm_record["vertex_count"]
    index_count   = prm_record["index_count"]
    vertex_stride = prm_record["vertex_stride"]
    is_weighted   = prm_record.get("is_weighted", False)

    # Naming: <PRMFileName>_<MeshIndex>_<Stride>B, with a WEIGHTED tag on skinned meshes.
    mesh_name = f"{file_stem}_{mesh_index:03d}_{vertex_stride}B"
    if is_weighted: mesh_name += "_WEIGHTED"
    print(f"\nBuilding submesh {mesh_index} ({vertex_stride}B stride, {vertex_count} verts, {index_count} indices)...")

    # --- BLENDER MESH + OBJECT ---
    mesh = bpy.data.meshes.new(name=mesh_name)
    obj  = bpy.data.objects.new(mesh_name, mesh)
    target_collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    # --- GEOMETRY ---
    positions = prm_record["positions"]
    triangles = prm_record["triangles"]
    normals   = prm_record["normals"]

    # Triangulation, de-stitching and winding are all resolved in the parser (strips for the
    # classic games, triangle lists for Blood Money), so triangles pass through unchanged.
    mesh.from_pydata(positions, [], triangles)
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if not IS_BLENDER_4_1: mesh.use_auto_smooth = True  # 4.1+ removed this flag; smooth is per-poly now.

    # --- NORMALS ---
    # Applied when the file actually carries a trusted normal: all classic meshes, Blood Money's
    # weighted vertices, and Blood Money's 36B vertices (whose +12 float3 normal reads unit-length
    # 500/500). Blood Money's STATIC 40B vertices carry no usable normal, so `normals` is empty
    # there and Blender computes smooth normals itself.
    if use_custom_normals is False and normals:
        mesh.normals_split_custom_set_from_vertices(normals)
        print("  Applied original Glacier normals as custom split normals.")
    else: print("  Using Blender-computed normals (custom normals disabled or unavailable).")

    # --- UV CHANNELS ---
    uv_channels = prm_record["uv_channels"] or []
    for channel_index, channel_data in enumerate(uv_channels):
        if not channel_data: continue
        layer_name = f"UV_{channel_index + 1:02d}"
        uv_layer = mesh.uv_layers.new(name=layer_name)
        print(f"  Inserting UV channel '{layer_name}' ({len(channel_data)} entries)...")
        for loop in mesh.loops: uv_layer.data[loop.index].uv = channel_data[loop.vertex_index]

    # --- TANGENTS ---
    # Blender derives tangents from UVs + normals; only attempt it when both exist.
    if uv_channels and uv_channels[0] and normals:
        try:
            mesh.calc_tangents()
            print("  Calculated tangents from UV + normal data.")
        except RuntimeError as exc: print(f"  WARNING: calc_tangents failed ({exc}); skipping.")

    # --- VERTEX COLORS ---
    vertex_colors = prm_record.get("vertex_colors")
    if vertex_colors:
        print("  Inserting vertex color attribute data...")
        apply_vertex_colors(obj, vertex_colors)

    # --- BONE WEIGHTS (skinned submeshes only) ---
    if is_weighted:
        print("  Inserting bone weights and vertex groups...")
        bone_names = prm_record.get("bone_names")  # real names when the skeleton is recovered
        apply_glacier1_bone_weights(obj, prm_record["bone_weights"], prm_record["bone_indices"], bone_names)

    # --- METADATA (round-trip + debugging) ---
    obj["game"]          = prm_record["game"]
    obj["vertex_stride"] = vertex_stride
    obj["vertex_count"]  = vertex_count
    obj["index_count"]   = index_count
    obj["strip_count"]   = prm_record["strip_count"]
    obj["anchor"]        = prm_record["anchor"]
    obj["is_weighted"]   = is_weighted
    if prm_record["block_index"] is not None: obj["block_index"] = prm_record["block_index"]

    mesh.update()
    return obj

# ------------------------------------------------
# TRANSFORM APPLICATION
# ------------------------------------------------

def build_transform_map(prm: PRM, gms: GMS) -> dict[int, list[dict]]:
    """Map each mesh index in `prm.objects` to the transforms it should be placed at.

    The two container families reference geometry differently, so the resolution differs:

      CLASSIC   the GMS prop's model reference IS the .PRM byte offset of the primitive, which
                is the same anchor the parser keyed its mesh map on. One hop.

      BLOOD MONEY  the prop points at a MODEL chunk, which owns a part list, which owns the
                   buffer chunks. Three hops through `PRM.resolve_model_parts`. A model commonly
                   owns several parts, so one prop can place several meshes at the same spot.

    Returns mesh index -> list of transform dicts ({position, rotation}). A mesh referenced by
    several props gets several entries and is instanced once per entry."""
    transform_map: dict[int, list[dict]] = {}
    unresolved = 0

    for model_reference, transforms in gms.transforms_by_model.items():
        if prm.is_blood_money: targets = prm.resolve_model_parts(model_reference)
        else: targets = [model_reference]

        matched = False
        for target in targets:
            mesh_index = prm.mesh_index_by_reference.get(target)
            if mesh_index is None: continue
            matched = True
            transform_map.setdefault(mesh_index, []).extend(transforms)

        if not matched: unresolved += 1

    print(f"Transform map: {len(transform_map)} meshes placed, {unresolved} model references unresolved.")
    return transform_map

def apply_prop_transform(blender_obj: bpy.types.Object, transform: dict) -> None:
    """Place a built mesh at a prop's world transform.

    `transform` carries a `position` (vec3) and a `rotation` (a flat row-major 3x3 tuple, or None
    when the record's matrix pointer did not resolve to an orthonormal matrix). When a rotation is
    present we build a full 4x4 basis so the object is both located AND oriented; when it is absent
    we place at the correct location with identity orientation rather than guessing a rotation."""
    position = Vector(transform["position"])
    rotation = transform.get("rotation")

    if rotation is None:
        blender_obj.matrix_world = Matrix.Translation(position)
        return

    # Flat row-major 3x3 -> mathutils 3x3 -> 4x4, then drop the translation in. mathutils.Matrix
    # takes a sequence of ROWS, which is exactly how the matrix is stored on disk.
    basis = Matrix((rotation[0:3], rotation[3:6], rotation[6:9])).to_4x4()
    basis.translation = position
    blender_obj.matrix_world = basis

# ------------------------------------------------
# PUBLIC ENTRY POINT
# ------------------------------------------------

def import_prm_model_g1(self, context: bpy.types.Context, file_path: str, game: str, use_custom_normals: bool = False, assign_material_colors: bool = True, import_gms: bool = True) -> set[str]:
    """Import a Glacier 1 PRM (RenderPrimitive) file into Blender.

    Args:
        self:                    Calling operator (for self.report()).
        context:                 Blender context.
        file_path:               Absolute path to the .PRM file.
        game:                    GLACIER1_H2SA, GLACIER1_HMC, GLACIER1_HBM or GLACIER1_FIGHTERS.
        use_custom_normals:      If True, let Blender compute normals from geometry rather than
                                 applying the on-disk ones. Default False (preserve file normals).
        assign_material_colors:  If True, give each placeholder material a random colour.
        import_gms:              If True, look for the companion .GMS beside the .PRM, parse it
                                 (inflating it first if it is still compressed) and use its prop
                                 table to place AND orient every mesh in the world. With this off,
                                 every primitive is built at the origin and the level stacks.

    Hierarchy built:
        <PRMFileName>                     (top-level collection)
          <PRMFileName>_Primitive_000     (sub-collection, holds the submesh and its instances)
          <PRMFileName>_Primitive_001
          ...
          <PRMFileName>_Skeletons         (Blood Money only: one armature per model skeleton)

    Returns:
        Blender operator result set.
    """
    start_time = time.time()
    print(f"\nIMPORTING GLACIER 1 PRM MODEL: {file_path}...\n")

    if not os.path.exists(file_path):
        self.report({'ERROR'}, f"PRM file not found: {file_path}")
        print(f"Cannot import model; file was not found at: {file_path}")
        return {'CANCELLED'}

    if game not in GLACIER1_SUPPORTED:
        self.report({'ERROR'}, f"Unsupported game type for Glacier 1 PRM import: {game}")
        print(f"Invalid game type constant for PRM: {game}")
        return {'CANCELLED'}

    # --- PARSE GEOMETRY ---
    try:
        prm = PRM(file_path, game)
    except Exception as exc:
        self.report({'ERROR'}, f"Failed to parse PRM: {exc}")
        print(f"PRM parse failure: {exc}")
        return {'CANCELLED'}

    if not prm.objects:
        self.report({'WARNING'}, "PRM contains no submeshes.")
        print("PRM has zero parsed submeshes; nothing to build.")
        return {'CANCELLED'}

    # --- ATTACH SKELETON BONE NAMES TO WEIGHTED RECORDS (Blood Money) ---
    # A weighted record knows which model chunk owns it (skeleton_model). Copy that skeleton's
    # bone names onto the record so the mesh builder can name vertex groups after real bones.
    for prm_record in prm.objects:
        model_chunk = prm_record.get("skeleton_model")
        if model_chunk is not None and model_chunk in prm.skeletons:
            prm_record["bone_names"] = prm.skeletons[model_chunk]["names"]

    # --- PARSE PLACEMENT (companion GMS) ---
    # Non-fatal by design: a missing or malformed GMS costs us placement, not the import. The
    # geometry is still perfectly usable at the origin, so we warn and carry on.
    transform_map: dict[int, list[dict]] = {}
    if import_gms:
        gms_path = find_companion_gms(file_path)
        if gms_path is None:
            print("No companion .GMS found beside the .PRM; every mesh will be built at the origin.")
            self.report({'WARNING'}, "No companion .GMS found; meshes placed at the origin.")
        else:
            try:
                gms = GMS(gms_path, game)
                transform_map = build_transform_map(prm, gms)
            except Exception as exc:
                print(f"GMS parse failure ({exc}); falling back to origin placement.")
                self.report({'WARNING'}, f"Could not read the companion .GMS: {exc}")

    # --- COLLECTION SETUP ---
    # <PRMFileName> holds one sub-collection per primitive so the outliner mirrors the file's
    # primitive order. Instanced primitives keep all their copies inside their own sub-collection.
    file_stem = Path(file_path).stem
    file_collection = ensure_collection(file_stem)

    # --- BUILD ARMATURES (Blood Money skeletons) ---
    # One armature per decoded model skeleton, all under a <PRMFileName>_Skeletons sub-collection.
    # Weighted meshes parent to the armature that owns them. Built before meshes so the parent
    # exists when a weighted submesh is linked.
    armature_by_model: dict[int, bpy.types.Object] = {}
    if prm.skeletons:
        skeleton_collection = ensure_collection(f"{file_stem}_Skeletons", parent=file_collection)
        for model_chunk, skeleton in prm.skeletons.items():
            try:
                armature_obj = construct_glacier1_armature(skeleton, model_chunk, file_stem, skeleton_collection)
                if armature_obj is not None: armature_by_model[model_chunk] = armature_obj
            except Exception as exc:
                print(f"  Skeleton for model chunk #{model_chunk} failed to build ({exc}); skipping.")

    # --- BUILD EACH SUBMESH ---
    built_objects: list[bpy.types.Object] = []
    instance_count = 0
    for mesh_index, prm_record in enumerate(prm.objects):
        primitive_collection = ensure_collection(f"{file_stem}_Primitive_{mesh_index:03d}", parent=file_collection)
        blender_obj = construct_glacier1_mesh(prm_record, mesh_index, file_stem, primitive_collection, use_custom_normals)
        add_prm_material(blender_obj, mesh_index, file_stem, assign_material_colors)

        # Parent a weighted mesh to the armature that owns it and add an Armature modifier, so the
        # vertex groups (named after real bones) drive deformation immediately.
        model_chunk = prm_record.get("skeleton_model")
        if prm_record.get("is_weighted") and model_chunk in armature_by_model: bind_mesh_to_armature(blender_obj, armature_by_model[model_chunk])

        built_objects.append(blender_obj)

        placements = transform_map.get(mesh_index, [])
        if not placements: continue

        # First placement moves the original; any further placements are linked duplicates that
        # share the same mesh datablock, so a primitive used 400 times costs one mesh, not 400.
        apply_prop_transform(blender_obj, placements[0])
        instance_count += 1
        for extra_transform in placements[1:]:
            copy_obj = blender_obj.copy()  # linked: shares obj.data
            primitive_collection.objects.link(copy_obj)
            apply_prop_transform(copy_obj, extra_transform)
            built_objects.append(copy_obj)
            instance_count += 1

    prm.blender_objects = built_objects

    print(f"\nMODEL BUILDING COMPLETE! ({len(prm.objects)} submeshes, {instance_count} placed instances, {len(armature_by_model)} armatures)")

    # --- TIMING REPORT ---
    elapsed_time = time.time() - start_time
    minutes      = int(elapsed_time // 60)
    seconds      = elapsed_time % 60
    if minutes > 0: elapsed_str = f"{minutes} minute{'s' if minutes > 1 else ''} and {seconds:.2f} seconds"
    else: elapsed_str = f"{seconds:.2f} seconds"

    success_message = f'Successfully imported model: "{Path(file_path).name}" in {elapsed_str}.'
    self.report({'INFO'}, success_message)
    print(success_message)

    return {'FINISHED'}

def bind_mesh_to_armature(blender_obj: bpy.types.Object, armature_obj: bpy.types.Object) -> None:
    """Parent a weighted mesh to its armature and add an Armature modifier.

    The mesh's vertex groups already carry the real bone names (see apply_glacier1_bone_weights),
    so an Armature modifier pointed at the same skeleton binds the deformation with no remapping.
    Object parenting keeps the mesh travelling with the armature."""
    blender_obj.parent = armature_obj
    modifier = blender_obj.modifiers.new(name="Armature", type='ARMATURE')
    modifier.object = armature_obj
    modifier.use_vertex_groups = True
