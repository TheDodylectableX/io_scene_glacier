import bpy.utils.previews
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, PointerProperty, FloatVectorProperty, CollectionProperty
from bpy.types import Operator, Panel, PropertyGroup

from .data_handlers.model_handler    import import_prim_model, import_prm_model_g1, import_cloth_model
from .data_handlers.skeleton_handler import import_skeleton
from .data_handlers.texture_handler  import import_texture, resolve_orphan_textures
from .data_format_parsers.tex        import import_tex_archive

from .utilities import *

custom_icons = None

# ========================================================================================================================================

bl_info = {
    "name": "IO Interactive Glacier1/2 Engine Modding Toolkit",
    "description": "Import and export various asset types for IO Interactive's Glacier 1/2 games.",
    "author": "Dodylectable",
    "blender": (4, 0, 0),
    "version": (1, 0, 0),
    "location": "File > Import",
    "doc_url": "https://github.com/TheDodylectableX/io_scene_glacier/wiki",
    "support": "COMMUNITY",
    "category": "Import-Export"
}

def register():
    """Entry point for enabling the addon."""
    print("io_scene_glacier: Initializing the plugin...")
    register_plugin()

def unregister():
    """Entry point for disabling the addon."""
    print("io_scene_glacier: Tearing down the plugin...")
    unregister_plugin()

if __name__ == "__main__": register()

# ===================================================================================================================================================

# ==============================================================================
# GAME DROPDOWN CALLBACK
# ==============================================================================
#
# Blender's EnumProperty supports custom icon previews ONLY via the dynamic callback form, where
# the callback returns a list of (identifier, name, description, icon_value, number) tuples.
# Static `items=(...)` arrays can only reference Blender's built-in string icons.
#
# We MUST keep a strong Python reference to the items list while the dropdown is alive, or Blender
# will garbage-collect the strings and corrupt the UI (a well-documented Blender Python footgun).
# `GAME_ITEMS_CACHE` holds it for us.
# ==============================================================================

GAME_ITEMS_CACHE: list[tuple] = []
GAME_ITEMS_CACHE_G1: list[tuple] = []

def get_game_enum_items(self, context) -> list[tuple]:
    """Build the Glacier 2 Game dropdown items dynamically so we can hand Blender our custom icon previews.

    Returning identifiers that match the global game constants in utilities.py keeps the
    operator's execute() body straight-through: `import_*(self, context, self.filepath, self.game, ...)`.
    """
    global GAME_ITEMS_CACHE
    GAME_ITEMS_CACHE = [
        (GLACIER2_ABSOLUTION, "Hitman: Absolution",             "First IOI game to use the Glacier 2 engine.", get_icon_by_id("absolution"), 0),
        (GLACIER2_TRILOGY,    "Hitman: World of Assassination", "The 2016, 2018 and 2021 Trilogy together",    get_icon_by_id("trilogy"),    1),
        (GLACIER2_BOND,       "007: First Light",               "Project 007",                                 get_icon_by_id("bond"),       2),
    ]
    return GAME_ITEMS_CACHE

def get_game_enum_items_g1(self, context) -> list[tuple]:
    """Build the Glacier 1 Game dropdown items. Same custom-icon pattern as the Glacier 2 list,
    scoped to the four titles the Glacier 1 PRM parser supports. Identifiers map 1:1 to the
    GLACIER1_* constants in utilities.py so the operator body stays straight-through."""
    global GAME_ITEMS_CACHE_G1
    GAME_ITEMS_CACHE_G1 = [
        (GLACIER1_H2SA,     "Hitman 2: Silent Assassin", "Classic anchor / reference-table PRM container.", get_icon_by_id("silent_assassin"), 0),
        (GLACIER1_HMC,      "Hitman: Contracts",         "Classic anchor / reference-table PRM container.", get_icon_by_id("contracts"),       1),
        (GLACIER1_HBM,      "Hitman: Blood Money",       "Heap / block-table PRM container.",               get_icon_by_id("blood_money"),     2),
        (GLACIER1_FIGHTERS, "Freedom Fighters",          "Classic anchor / reference-table PRM container.", get_icon_by_id("freedom_fighters"),3),
    ]
    return GAME_ITEMS_CACHE_G1

# ==============================================================================
# SHARED CLASS MIXINS
# ==============================================================================

class IOIGameSelectMixin:
    """Provides the Game dropdown shared by every importer and future exporter. Identifiers map 1:1 to the enumerations in utilities.py so the operator body can pass `self.game` straight into the handler functions."""
    game: EnumProperty(
        name="Game",
        description="Which Glacier 2 title this file is from. Determines header layout, vertex stream format and constraint encoding",
        items=get_game_enum_items,
    ) # type: ignore

class IOIGameSelectMixinG1:
    """Provides the Glacier 1 Game dropdown. Identifiers map 1:1 to the GLACIER1_* enumerations in utilities.py so the operator body can pass `self.game` straight into the handler."""
    game: EnumProperty(
        name="Game",
        description="Which Glacier 1 title this file is from. Determines the PRM container (classic anchors vs Blood Money heap) and per-game field layout",
        items=get_game_enum_items_g1,
    ) # type: ignore

class IOIPrmImportMixinG1(IOIGameSelectMixinG1):
    """Shared properties for the Glacier 1 PRM importer."""
    custom_normals: BoolProperty(
        name="Custom Normals",
        description="If enabled, let Blender recompute normals from geometry instead of using the original Glacier-stored normals. (Blood Money always uses Blender-computed normals as its encoding is unconfirmed)",
        default=False,
    ) # type: ignore

    assign_material_colors: BoolProperty(
        name="Assign Material Colors",
        description="Assign a random color to the placeholder material to help with distinguishing primitives",
        default=True,
    ) # type: ignore

    import_gms: BoolProperty(
        name="Import GMS",
        description="Look for the companion .GMS mission script beside the .PRM and use its prop table to place every mesh in the world. Compressed scripts are inflated automatically. With this disabled, every primitive is built at the origin and the level stacks on top of itself",
        default=True,
    ) # type: ignore

class IOIPrimImportMixin(IOIGameSelectMixin):
    """Shared properties for the PRIM importer."""
    custom_normals: BoolProperty(
        name="Custom Normals",
        description="If enabled, let Blender recompute normals from geometry instead of using the original Glacier-stored normals. (Smoother look but loses fidelity)",
        default=False,
    ) # type: ignore

    assign_material_colors: BoolProperty(
        name="Assign Material Colors",
        description="Assign random colors to the model's placeholder materials to help with distinguishing submeshes",
        default=True,
    ) # type: ignore

    import_collisions: BoolProperty(
        name="Import Collisions",
        description="Build wireframe objects for the BoxColi (bullet/projectile collision boxes) under a 'Collision' sub-collection",
        default=False,
    ) # type: ignore

    import_lods: BoolProperty(
        name="Import LODs",
        description="Import every LOD grouped into 'LOD #' sub-collections. When disabled, Only LOD0 meshes (LOD mask 0x01) and cloth/physics meshes (LOD mask 0xFF) are built",
        default=True,
    ) # type: ignore

# ==============================================================================
# IMPORTER CLASSES
# ==============================================================================

class ImportG2RenderPrimitive(Operator, ImportHelper, IOIPrimImportMixin):
    """Import a Glacier 2 RenderPrimitive (.PRIM) model"""
    bl_idname = "import_glacier2.renderprimitive"
    bl_label = "Import Glacier 2 RenderPrimitive (.PRIM)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".PRIM"
    filter_glob: StringProperty(default="*.PRIM", options={'HIDDEN'}, maxlen=1024) # type: ignore

    def execute(self, context): return import_prim_model(self, context, self.filepath, self.game, use_custom_normals = self.custom_normals, assign_material_colors = self.assign_material_colors, import_collisions = self.import_collisions, import_lods = self.import_lods)

class ImportG1RenderPrimitive(Operator, ImportHelper, IOIPrmImportMixinG1):
    """Import a Glacier 1 RenderPrimitive (.PRM) level model"""
    bl_idname = "import_glacier1.renderprimitive"
    bl_label = "Import Glacier 1 RenderPrimitive (.PRM)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".PRM"
    filter_glob: StringProperty(default="*.PRM", options={'HIDDEN'}, maxlen=1024) # type: ignore

    def execute(self, context): return import_prm_model_g1(self, context, self.filepath, self.game, use_custom_normals = self.custom_normals, assign_material_colors = self.assign_material_colors, import_gms = self.import_gms)

class ImportG1RenderTexture(Operator, ImportHelper, IOIGameSelectMixinG1):
    """Import a Glacier 1 RenderTexture (.TEX) archive.

    Always extracts every texture into a folder beside the .TEX (same name, no suffix), rebuilding
    a standalone .DDS per texture with subfolders preserved and duplicate names disambiguated by
    id. Optionally loads the rebuilt DDS files into Blender as well."""
    bl_idname = "import_glacier1.rendertexture"
    bl_label = "Import Glacier 1 RenderTexture (.TEX)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".TEX"
    filter_glob: StringProperty(default="*.TEX", options={'HIDDEN'}, maxlen=1024) # type: ignore

    import_to_blender: BoolProperty(
        name="Import Textures to Blender",
        description="When enabled, load every extracted texture into Blender after rebuilding it. When disabled, the archive is only extracted and its DDS files rebuilt on disk - nothing is loaded into the scene",
        default=False,
    ) # type: ignore

    def execute(self, context): return import_tex_archive(self, context, self.filepath, self.game, import_to_blender = self.import_to_blender)

class ImportG2BoneRig(Operator, ImportHelper, IOIGameSelectMixin):
    """Import a Glacier 2 BoneRig (.BORG) skeleton"""
    bl_idname = "import_glacier2.bonerig"
    bl_label = "Import Glacier 2 BoneRig (.BORG)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".BORG"
    filter_glob: StringProperty(default="*.BORG", options={'HIDDEN'}, maxlen=1024) # type: ignore

    def execute(self, context): return import_skeleton(self, context, self.filepath, self.game)

class ImportG2RenderTexture(Operator, ImportHelper, IOIGameSelectMixin):
    """Import Glacier 2 RenderTexture (.TEXT) images - select as many as you like at once"""
    bl_idname = "import_glacier2.rendertexture"
    bl_label = "Import Glacier 2 RenderTexture (.TEXT)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".TEXT"
    # Allow .TEXD in the picker too, so the user can constrain the companion pool by hand if they
    # want; otherwise companions are auto-discovered next to each selected .TEXT.
    filter_glob: StringProperty(default="*.TEXT;*.TEXD", options={'HIDDEN'}, maxlen=1024) # type: ignore

    # Multi-file selection. Blender populates `files` + `directory` whenever more than one file is
    # chosen; for a single pick we fall back to `self.filepath` in execute().
    files: CollectionProperty(name="File Path", type=bpy.types.OperatorFileListElement) # type: ignore
    directory: StringProperty(subtype='DIR_PATH') # type: ignore

    def execute(self, context):
        paths = [os.path.join(self.directory, f.name) for f in self.files] if self.files else [self.filepath]
        orphans = import_texture(self, context, paths, self.game)
        # WoA/007FL textures whose .TEXD wasn't beside them are imported at proxy resolution; offer
        # to point at a folder holding the streams so we can upgrade them to full res. Deferred via a
        # timer so the file browser opens cleanly once this operator has already returned.
        if orphans:
            def open_locate_dialog():
                try: bpy.ops.import_glacier2.locate_texd('INVOKE_DEFAULT')
                except Exception as error: print(f"[Glacier TEXT] Could not open the .TEXD folder dialog: {error}")
                return None
            bpy.app.timers.register(open_locate_dialog, first_interval=0.01)
        return {'FINISHED'}


class ImportG2LocateTexd(Operator):
    """Point to a folder holding the .TEXD streams for textures whose companion wasn't found beside the .TEXT"""
    bl_idname = "import_glacier2.locate_texd"
    bl_label = "Locate .TEXD Folder"
    bl_options = {'REGISTER'}
    directory: StringProperty(subtype='DIR_PATH') # type: ignore
    filter_glob: StringProperty(default="*.TEXD", options={'HIDDEN'}, maxlen=1024) # type: ignore

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)  # pick the folder; execute() runs on confirm, nothing on cancel
        return {'RUNNING_MODAL'}

    def execute(self, context):
        resolve_orphan_textures(self, context, self.directory)
        return {'FINISHED'}

class ImportCloakWorksCloth(Operator, ImportHelper):
    """Import a CloakWorks Shroud Cloth (.CLOS) mesh from Hitman: Absolution.

    CLOS is Absolution-only, so there is no per-file Game selection - the format is fixed. Each cloth
    piece imports as its rest-pose grid mesh (real node positions, generated topology, planar UVs)."""
    bl_idname = "import_cloakworks.shroudcloth"
    bl_label = "Import CloakWorks Shroud Cloth (.CLOS)"
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = ".CLOS"
    filter_glob: StringProperty(default="*.CLOS", options={'HIDDEN'}, maxlen=1024) # type: ignore

    assign_material_colors: BoolProperty(
        name="Assign Material Colors",
        description="Assign a random color to each cloth piece's placeholder material to help with distinguishing pieces",
        default=True,
    ) # type: ignore

    def execute(self, context): return import_cloth_model(self, context, self.filepath, assign_material_colors = self.assign_material_colors)

# ===================================================================================================================================================

IMPORTER_CLASSES = (ImportG1RenderPrimitive, ImportG1RenderTexture, ImportG2RenderPrimitive, ImportG2BoneRig, ImportG2RenderTexture, ImportG2LocateTexd, ImportCloakWorksCloth)
# EXPORTER_CLASSES = (ExportG2RenderPrimitive, ExportG2BoneRig, ExportG2RenderTexture)

# ==============================================================================
# GLACIER MESH PROPERTIES  (Object Properties panel + backing PropertyGroup)
# ==============================================================================
#
# Every imported PRIM submesh carries header/subtype/flag metadata that the engine needs and a
# future exporter must reproduce. We store it in a PropertyGroup attached to bpy.types.Object so
# it is (a) editable in the UI, (b) saved with the .blend, and (c) trivially readable at export
# time via `obj.glacier_mesh`. Import presets every field from the parsed record (per submesh /
# per LOD); the user can then override any of them to control how the mesh exports.
#
# The enums mirror the format exactly:
#   record Type    -> PRIM_OBJECT_TYPE   (BT: UNKNOWN/OBJECT_HEADER/MESH/DECAL/SPRITES/SHAPE/UNUSED)
#   subType        -> PRIM_OBJECT_SUBTYPE(BT: STANDARD/LINKED/WEIGHTED/STANDARD_UV_2..4)
#   properties     -> PRIM_OBJECT_FLAGS  (u8 bitfield, LSB-first)
# Both games share these enums; only 007FL's stream encoding differs, which is invisible here.
# ==============================================================================

GLACIER_PRIM_TYPE_ITEMS = (
    ('0', "Unknown", "Unset / Unknown mesh type",   0),
    ('2', "Mesh",    "Standard renderable mesh",    2),
    ('3', "Decal",   "Decal primitive",             3),
    ('4', "Sprites", "Sprite primitive",            4),
    ('5', "Shape",   "Shape primitive",             5),
)

GLACIER_PRIM_SUBTYPE_ITEMS = (
    ('0', "Standard",     "Standard mesh with a single UV channel and an optional vertex color layer, (Used for props)",   0),
    ('1', "Linked",       "Linked mesh with 16 bytes per vertex and interleaved vertex data (Used for weapons and props)", 1),
    ('2', "Weighted",     "Skinned mesh with planar attributes and per vertex skinning (Used for characters)",             2),
    ('3', "Standard UV2", "Standard mesh but with 2 UV channels",                                                          3),
    ('4', "Standard UV3", "Standard mesh but with 3 UV channels",                                                          4),
    ('5', "Standard UV4", "Standard mesh but with 4 UV channels",                                                          5),
)

GLACIER_MESH_CLASS_ITEMS = (
    ('0', "None",      "Class not set - clothID's high half is 0. Normal for Absolution and WoA meshes",             0),
    ('1', "Standard",  "Regular mesh - single UV set in the vertex record (bodies, faces, props, hair scalp)",       1),
    ('2', "Hair Card", "Hair card mesh - carries a second UV set in the vertex record (007FL)",                      2),
)

class GlacierMeshProperties(PropertyGroup):
    """Per-object Glacier PRIM metadata. Available on every mesh so custom (non-imported) meshes
    can be authored for export; the importer presets these fields from the parsed record.

    Note: the target GAME is intentionally NOT stored here - the flags/enums are identical across
    all three Glacier 2 titles, so the game is chosen once on the Export operator (mirroring the
    Import operator's dropdown) rather than per object.
    """

    # ----- 1. Header -----
    record_type: EnumProperty(
        name="Type",
        description="PRIM_OBJECT record type from the header (Mesh for all renderable geometry)",
        items=GLACIER_PRIM_TYPE_ITEMS,
        default='2',
    ) # type: ignore

    # ----- 2. Subtype -----
    sub_type: EnumProperty(
        name="Subtype",
        description="Vertex layout family. Determines how attributes are written on export",
        items=GLACIER_PRIM_SUBTYPE_ITEMS,
        default='0',
    ) # type: ignore

    # ----- 3. Properties (PRIM_OBJECT_FLAGS, u8, LSB-first) -----
    is_x_axis_locked:    BoolProperty(name="X Axis Locked",   description="Bit 0: Lock the X axis.",                                            default=False) # type: ignore
    is_y_axis_locked:    BoolProperty(name="Y Axis Locked",   description="Bit 1: Lock the Y axis.",                                            default=False) # type: ignore
    is_z_axis_locked:    BoolProperty(name="Z Axis Locked",   description="Bit 2: Lock the Z axis.",                                            default=False) # type: ignore
    is_high_resolution:  BoolProperty(name="High Resolution", description="Bit 3: Use floating points for vertex positions instead of shorts.", default=False) # type: ignore
    has_ps3_edge:        BoolProperty(name="PS3 Edge",        description="Bit 4: PS3 EDGE geometry data present.",                             default=False) # type: ignore
    use_color_1:         BoolProperty(name="Use Color 1",     description="Bit 5: Use the object's solid Color1 tint.",                         default=False) # type: ignore
    has_no_physics_prop: BoolProperty(name="No Physics Prop", description="Bit 6: Mesh has no physics proxy.",                                  default=False) # type: ignore

    # Solid color 1 tint (RGB). Only meaningful when 'Use Color 1' is enabled. Stored 0-1 float per channel; Converted to/from the on-disk u8 RGBA at import/export.
    # White = No tint. (Set as the default)
    color_1: FloatVectorProperty(
        name="Color 1",
        description="Solid color tint applied when 'Use Color 1' is enabled",
        subtype='COLOR',
        size=3,
        min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0),
    ) # type: ignore

    # ----- Extra metadata (preset from file on import; editable for export) -----
    material_id: IntProperty(name="Material ID", description="Material slot index referenced by this submesh",         default=1, min=0)          # type: ignore
    lod_mask:    IntProperty(name="LOD Mask",    description="Per-LOD-level bitmask (Bit N = LOD N). 0xFF = All LODs", default=1, min=0, max=255) # type: ignore
    variant_id:  IntProperty(name="Variant ID",  description="Mesh variant index",                                     default=0, min=0)          # type: ignore

    # ----- Cloth / mesh class (the clothID u32, split into its two meaningful halves) -----
    #
    # clothID packs two unrelated things into one word, so the panel splits them rather than
    # making you do hex arithmetic:
    #
    #   high u16 = MESH CLASS. On 007FL this is what selects the vertex record width: class 1
    #              meshes write a 16-byte NTB+UV record (one UV set), class 2 - hair cards only -
    #              write 20 bytes with a second UV set. Get this wrong on export and every
    #              attribute after the normals lands at the wrong offset.
    #   low u16  = CLOTH BLOB INDEX. Nonzero means this mesh has simulated cloth attached and the
    #              mesh body carries a cloth-data offset alongside it; 0 means no cloth.
    #
    # Both halves are authorable because a custom mesh may legitimately need either (a new hair
    # card set, or a garment reusing an existing cloth blob).
    mesh_class: EnumProperty(
        name="Mesh Class",
        description="High half of clothID. Selects the vertex record layout on 007FL - Hair Card meshes carry a second UV set",
        items=GLACIER_MESH_CLASS_ITEMS,
        default='0',
    ) # type: ignore

    cloth_blob_index: IntProperty(
        name="Cloth Blob Index",
        description="Low half of clothID. Index of this mesh's cloth simulation blob; 0 means the mesh has no cloth data",
        default=0, min=0, max=0xFFFF,
    ) # type: ignore

PRIM_FLAG_FIELDS = ("is_x_axis_locked", "is_y_axis_locked", "is_z_axis_locked", "is_high_resolution", "has_ps3_edge", "use_color_1", "has_no_physics_prop")

def lod_index_from_mask(mask: int) -> int:
    """Lowest set bit position = the most-detailed LOD this mesh participates in (0 if none)."""
    return (mask & -mask).bit_length() - 1 if mask else 0

class GLACIER_PT_mesh_properties(Panel):
    """Object Properties subsection exposing the PRIM header/subtype/flags for edit + export."""
    bl_idname      = "GLACIER_PT_mesh_properties"
    bl_label       = "IO Interactive Glacier2 Engine: Mesh Properties"
    bl_space_type  = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context     = "object"
    bl_options     = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        if obj is None or obj.type != 'MESH':
            return False
        # Hide on collision wireframes: they're authored/derived geometry, not exportable PRIM
        # submeshes. Detect them by the "_Collision" naming the importer applies, or by having no
        # faces (BoxColi wireframes are edge-only). Everything else - including brand-new custom
        # meshes the user models themselves - shows the panel so it can be authored for export.
        if "collision" in obj.name.lower():
            return False
        mesh = obj.data
        if mesh is not None and len(mesh.polygons) == 0:
            return False
        return True

    def draw(self, context):
        layout = self.layout
        props  = context.object.glacier_mesh
        layout.use_property_split = True
        layout.use_property_decorate = False

        # 1. Header
        header_box = layout.box()
        header_box.label(text="Header", icon='OBJECT_DATA')
        header_box.prop(props, "record_type")

        # 2. Subtype
        subtype_box = layout.box()
        subtype_box.label(text="Subtype", icon='MESH_DATA')
        subtype_box.prop(props, "sub_type")

        # 3. Properties (flags)
        flags_box = layout.box()
        flags_box.label(text="Properties", icon='MODIFIER')
        flag_col = flags_box.column(align=True)
        for field in PRIM_FLAG_FIELDS: flag_col.prop(props, field)

        # Color1 tint picker - greyed out unless 'Use Color 1' is enabled.
        color_row = flags_box.row()
        color_row.enabled = props.use_color_1
        color_row.prop(props, "color_1")

        # Reference metadata
        meta_box = layout.box()
        meta_box.label(text="Submesh / LOD", icon='INFO')
        meta_box.prop(props, "material_id")
        meta_box.prop(props, "variant_id")
        row = meta_box.row()
        row.prop(props, "lod_mask")
        # Derived LOD index shown read-only next to the mask.
        meta_box.label(text=f"LOD Index: {lod_index_from_mask(props.lod_mask)}")

        # Cloth / mesh class - the two halves of clothID, plus the raw word they compose into so
        # it can be matched against a hex editor without doing the arithmetic by hand.
        cloth_box = layout.box()
        cloth_box.label(text="Cloth / Mesh Class", icon='MOD_CLOTH')
        cloth_box.prop(props, "mesh_class")
        cloth_box.prop(props, "cloth_blob_index")
        cloth_raw = (int(props.mesh_class) << 16) | (props.cloth_blob_index & 0xFFFF)
        cloth_box.label(text=f"Cloth ID: 0x{cloth_raw:08X}" + (" (Has Cloth Data)" if props.cloth_blob_index else " (No Cloth Data)"))

# ==============================================================================
# UI MENUS & PANELS
# ==============================================================================

class GLACIER_MT_import_menu(bpy.types.Menu):
    bl_label = "IO Interactive Modding"
    def draw(self, context): add_import_options(self.layout)

class GLACIER_MT_3dview_menu(bpy.types.Menu):
    bl_label = "IO Interactive Modding"
    def draw(self, context): add_3d_view_object_menu_options(self.layout)

class GLACIER_OT_skm_indices_to_names(bpy.types.Operator):
    """Renames all mesh vertex groups with bone IDs to their respective bone names."""
    bl_idname = "object.skm_indices_to_names"
    bl_label = "Skeletal Mesh: Indices to Bone Names"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'MESH' and obj.modifiers.get('Armature'):
            handle_vertex_group_rename_to_names()
            self.report({'INFO'}, "Successfully renamed vertex groups to bone names!")
            return {'FINISHED'}
        self.report({'WARNING'}, "Selected mesh does not have an armature modifier.")
        return {'CANCELLED'}

class GLACIER_OT_skm_names_to_indices(bpy.types.Operator):
    """Renames all mesh vertex groups with bone names to their respective bone indices."""
    bl_idname = "object.skm_names_to_indices"
    bl_label = "Skeletal Mesh: Bone Names to Indices"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        obj = context.active_object
        if obj and obj.type == 'MESH' and obj.modifiers.get('Armature'):
            handle_vertex_group_rename_to_indices()
            self.report({'INFO'}, "Successfully renamed vertex groups to bone indices!")
            return {'FINISHED'}
        self.report({'WARNING'}, "Selected mesh does not have an armature modifier.")
        return {'CANCELLED'}

# ==============================================================================
# ICON MANAGEMENT
# ==============================================================================

def register_icons():
    """Load every PNG in /icons/ into the custom_icons preview collection."""
    global custom_icons
    script_dir = os.path.dirname(__file__)
    icon_dir = os.path.join(script_dir, "icons")

    pcoll = bpy.utils.previews.new()
    if os.path.exists(icon_dir):
        for image in os.listdir(icon_dir):
            if image.endswith(".png"):
                shorthand = os.path.splitext(image)[0]
                pcoll.load(shorthand, os.path.join(icon_dir, image), 'IMAGE')
    custom_icons = pcoll

def unregister_icons():
    global custom_icons
    if custom_icons:
        bpy.utils.previews.remove(custom_icons)
        custom_icons = None

def get_icon_by_id(icon_name: str) -> int:
    """Return the icon_value integer for a custom icon, or 0 (no icon) if missing. Safe to call before register_icons() has run; it just returns 0 in that case which Blender treats as 'no custom icon' and falls back to whatever default the widget uses."""
    if custom_icons and icon_name in custom_icons: return custom_icons[icon_name].icon_id
    return 0

# ==============================================================================
# MENU APPEND FUNCTIONS
# ==============================================================================

def add_import_sub_menu(self, context):
    self.layout.menu("GLACIER_MT_import_menu", text="IO Interactive's Glacier Engine", icon_value=get_icon_by_id("icon"))

def add_3d_view_object_menu(self, context):
    self.layout.separator()
    self.layout.menu("GLACIER_MT_3dview_menu", text="IO Interactive's Glacier Engine", icon_value=get_icon_by_id("icon"))

def add_import_options(layout):
    """Populate the import sub-menu. Each operator entry shows the Glacier engine generic icon since the per-file game choice happens inside the import dialog via the Game dropdown."""
    glacier_icon = get_icon_by_id("icon")
    layout.operator("import_glacier1.renderprimitive", text="Glacier 1 RenderPrimitive (.PRM)",  icon_value=glacier_icon)
    layout.operator("import_glacier1.rendertexture",   text="Glacier 1 RenderTexture (.TEX)",    icon_value=glacier_icon)

    layout.separator()

    layout.operator("import_glacier2.renderprimitive", text="Glacier 2 RenderPrimitive (.PRIM)", icon_value=glacier_icon)
    layout.operator("import_glacier2.bonerig",         text="Glacier 2 BoneRig (.BORG)",         icon_value=glacier_icon)
    layout.operator("import_glacier2.rendertexture",   text="Glacier 2 RenderTexture (.TEXT)",   icon_value=glacier_icon)

    layout.separator() # CloakWorks Shroud cloth is only in Hitman: Absolution so it doesn't need the game selection dialogue.

    layout.operator("import_cloakworks.shroudcloth",   text="CloakWorks Shroud Cloth (.CLOS)",    icon_value=get_icon_by_id("absolution"))

def add_3d_view_object_menu_options(layout):
    can_rename = len(bpy.context.selected_objects) > 0
    row = layout.row()
    row.enabled = can_rename
    row.operator("object.skm_indices_to_names", text="Bone Indices to Names")

    row = layout.row()
    row.enabled = can_rename
    row.operator("object.skm_names_to_indices", text="Bone Names to Indices")

# ==============================================================================
# PLUGIN REGISTRATION
# ==============================================================================

CLASSES = (
    GLACIER_MT_import_menu,
    GLACIER_MT_3dview_menu,
    *IMPORTER_CLASSES,
    GLACIER_OT_skm_indices_to_names,
    GLACIER_OT_skm_names_to_indices,
    GlacierMeshProperties,
    GLACIER_PT_mesh_properties,
)

def register_plugin():
    """Register all components of the plugin."""
    # Icons FIRST - the EnumProperty callback for the Game dropdown calls get_icon_by_id(), and
    # that needs the preview collection populated before any operator class with the property
    # is registered (otherwise the items would show icon_value=0 and never refresh).
    register_icons()

    for cls in CLASSES: bpy.utils.register_class(cls)

    # Attach the per-object Glacier metadata. PointerProperty must be assigned AFTER its
    # PropertyGroup class is registered (above).
    bpy.types.Object.glacier_mesh = PointerProperty(type=GlacierMeshProperties)

    bpy.types.TOPBAR_MT_file_import.append(add_import_sub_menu)
    bpy.types.VIEW3D_MT_object.append(add_3d_view_object_menu)

def unregister_plugin():
    """Unregister all components of the plugin."""
    bpy.types.TOPBAR_MT_file_import.remove(add_import_sub_menu)
    bpy.types.VIEW3D_MT_object.remove(add_3d_view_object_menu)

    # Remove the per-object pointer BEFORE unregistering its class.
    if hasattr(bpy.types.Object, "glacier_mesh"):
        del bpy.types.Object.glacier_mesh

    for cls in reversed(CLASSES):
        try: bpy.utils.unregister_class(cls)
        except RuntimeError: pass

    unregister_icons()

# ===================================================================================================================================================
