# =====================================================
#   GLACIER 1 GMS (GLOBAL MISSION SCRIPT) PARSER
#       Parses IO Interactive's Glacier 1 mission
#       script containers used by:
#           - Hitman 2: Silent Assassin
#           - Hitman: Contracts
#           - Hitman: Blood Money (and Mini Ninjas, same tech)
#           - Freedom Fighters
#
#       Mirrors GMS.bt. If you change the binary
#       template, mirror the change here.
#
#       The GMS is the companion to a level's .PRM. The
#       .PRM holds raw geometry with everything sitting
#       at the origin; the GMS holds the PROP TABLE that
#       says where each piece is actually placed and
#       which primitive it draws. Import a .PRM alone and
#       every mesh lands on top of every other mesh at
#       the world origin - the GMS is what un-roots them.
#
#       Original research on the prop table by id-daemon;
#       verified byte-exact against real samples here.
# =====================================================

import struct, zlib
from ..io import Reader
from ..utilities import *

# ==========
# CONSTANTS
# ==========

# Shipped GMS files are raw DEFLATE behind a 9 byte header. There is no zlib wrapper, so the
# window-bits argument must be negative (-15 = raw inflate, no header, no checksum).
GMS_COMPRESSED_HEADER_SIZE = 9
GMS_RAW_DEFLATE_WINDOW     = -15

# Prop table entry: a u64 whose LOW 24 BITS are a WORD index. Multiply by 4 for the byte offset
# of the prop record. The UPPER bits are geometry flags: bit 24 marks the root of a group and
# bits 25+ are the scene-hierarchy relative depth level (BMEdit's GMSGeomEntity::getRelativeDepth
# uses `flags >> 25`). We surface the depth but do NOT need it for placement - the transform the
# pointers resolve to is already world-absolute.
GMS_PROP_OFFSET_MASK   = 0xFFFFFF
GMS_PROP_OFFSET_SHIFT  = 4
GMS_PROP_ROOT_FLAG     = 0x1000000
GMS_PROP_DEPTH_SHIFT   = 25

# Prop record field offsets, relative to the record start.
#
# The transform is NOT stored inline - the record carries POINTERS to it. This is the corrected
# decode: probing every record across all four games, the word at record+4 points at an
# orthonormal 3x3 rotation matrix and the word at record+8 points at a vec3 position. Validated
# at ~100%: H2SA 3979/3990 matrices + 3990/3990 positions, HMC 6247/6252, FF 6615/6617,
# HBM 7857/7864. The old decode read a raw vec3 at record-12 (translation only, no rotation) and
# silently read garbage for the ~half of records that don't own an inline position block.
#
#   record + 0  : descriptor pointer / prop id
#   record + 4  : POINTER to a 3x3 rotation matrix (9 floats, row-major)
#   record + 8  : POINTER to a vec3 world position
#   record + 12 : model reference - a raw .PRM byte offset on the classic games,
#                 a .PRM block-table INDEX on Blood Money
GMS_PROP_DESCRIPTOR_OFFSET = 0
GMS_PROP_MATRIX_PTR_OFFSET = 4
GMS_PROP_POSITION_PTR_OFFSET = 8
GMS_PROP_MODEL_REF_OFFSET  = 12

# Sizes of the pointed-at transform data.
GMS_MATRIX_FLOATS   = 9   # 3x3 row-major
GMS_POSITION_FLOATS = 3

# Sanity bound for a decoded position component. Level coordinates are in the low thousands;
# anything past this is a misparse, not a far-away prop.
GMS_POSITION_SANITY_LIMIT = 1.0e6

# A decoded 3x3 is accepted as a rotation only if its rows are unit-length and mutually
# orthogonal within these tolerances - the same test that validated the pointer at ~100%. A
# record whose matrix pointer fails the test still places by translation with identity rotation.
GMS_MATRIX_UNIT_TOLERANCE = 0.05
GMS_MATRIX_ORTHO_TOLERANCE = 0.05

# The games this parser accepts. Same set as the PRM parser (Mini Ninjas rides on GLACIER1_HBM).
GLACIER1_GMS_SUPPORTED = (GLACIER1_H2SA, GLACIER1_HMC, GLACIER1_HBM, GLACIER1_FIGHTERS)

# =====================================================
# MAIN PARSER CLASS
# =====================================================

class GMS():
    """Glacier 1 mission script parser. Produces the prop table: every placed object's world
    transform (position + rotation) and the primitive it references in the companion `.PRM`."""
    def __init__(self, file_path: str, game: str):
        """Construct the parser and run the full parse pass."""

        super().__init__()

        # ===============================
        # == CLASS MEMBERS ==============
        # ===============================

        # -- INPUT METADATA
        self.script_file: str = file_path
        """The path to the source `.GMS` file."""

        self.game: str = game
        """Which Glacier 1 title produced this file. Drives model-reference interpretation."""

        self.is_blood_money: bool = (game == GLACIER1_HBM)
        """Blood Money stores a block-table index where the classic games store a byte offset."""

        # -- CONTAINER STATE
        self.was_compressed: bool = False
        """True when the file needed inflating (a shipped GMS) rather than being already decompressed."""

        self.buffer_length: int = 0
        """Length of the (decompressed) buffer, for bounds checks."""

        self.header: dict = {}
        """The decoded 16-uint header."""

        # -- PROP TABLE
        self.props: list[dict] = []
        """Every decoded prop. See `build_prop_record` for the shape of each entry."""

        self.transforms_by_model: dict[int, list[dict]] = {}
        """Model reference -> list of transforms. Each transform is a dict with `position` and
        `rotation` (a 3x3 row-major tuple, or None when the record's matrix pointer failed the
        orthonormality test). A single primitive is commonly instanced many times across a level,
        so this is one-to-many by design."""

        # -- VALIDATION
        if game not in GLACIER1_GMS_SUPPORTED: raise ValueError(f"Unsupported game type for GMS parsing: '{game}'. Use GLACIER1_H2SA, GLACIER1_HMC, GLACIER1_HBM or GLACIER1_FIGHTERS.")

        # ===============================
        # == PARSE THE DATA =============
        # ===============================
        self.parse_script_file()

    # =====================================================
    # DECOMPRESSION
    # =====================================================

    def load_and_decompress(self) -> bytes:
        """Read the GMS off disk, inflating it if it is still in its shipped compressed form.

        Detection is structural rather than magic-based: in a decompressed GMS the first uint is
        the prop table's offset, so it must land inside the file with room for a count behind it.
        If that holds we take the buffer as-is (Dody's samples are pre-decompressed); if it does
        not, we skip the 9 byte header and raw-inflate."""
        raw = open(self.script_file, "rb").read()

        if len(raw) >= 8:
            table_offset = struct.unpack_from("<I", raw, 0)[0]
            if 0 < table_offset < len(raw) - 4:
                self.was_compressed = False
                return raw

        # Still packed: 9 byte header then a raw DEFLATE stream (no zlib wrapper, no checksum).
        try:
            inflated = zlib.decompressobj(GMS_RAW_DEFLATE_WINDOW).decompress(raw[GMS_COMPRESSED_HEADER_SIZE:])
        except zlib.error as error:
            raise ValueError(f"GMS is neither a valid decompressed container nor raw DEFLATE: {error}")

        if len(inflated) < 8: raise ValueError("GMS inflated to an implausibly small buffer.")
        self.was_compressed = True
        print(f"Inflated GMS: {len(raw)} -> {len(inflated)} bytes")
        return inflated

    # =====================================================
    # TOP-LEVEL DRIVER
    # =====================================================

    def parse_script_file(self) -> None:
        """Parse the mission script. Header, then the prop table."""
        print(f"\nParsing GMS script ({self.game}): {self.script_file}\n")

        buffer = self.load_and_decompress()
        self.buffer = buffer
        reader = Reader(buffer)
        self.buffer_length = len(buffer)

        # ---------------- HEADER ----------------
        # 16 uints. Only the first is needed for the prop table; the rest are the section offsets
        # documented in GMS.bt (entity table, byte code, resource tables and so on).
        header_fields = [reader.uint32() for _ in range(16)]
        self.header = {
            "prop_table_offset":          header_fields[0],
            "string_offset_table_offset": header_fields[1],
            "name_list_offset":           header_fields[2],
            "format_version":             header_fields[3],
            "entity_table_offset":        header_fields[5],
            "byte_code_offset":           header_fields[6],
            "resource_group_offset":      header_fields[8],
            "resource_entry_offset":      header_fields[9],
        }

        if self.header["format_version"] != 4: print(f"Note: GMS format version is {self.header['format_version']}, not the 4 seen in every sample.")

        self.parse_prop_table(reader)

        print(f"\nGMS PARSING COMPLETE!  {len(self.props)} props, {len(self.transforms_by_model)} distinct model references.\n")

    # =====================================================
    # PROP TABLE
    # =====================================================

    def parse_prop_table(self, reader: Reader) -> None:
        """Parse the prop table (header field +00).

        Layout: [uint propCount][propCount * uint64]. Each u64's low 24 bits are a WORD index;
        multiply by 4 to get the byte offset of that prop's record. Every entry in every sample
        resolves to an in-file record (3990/3990, 6252/6252, 6617/6617, 7864/7864). The record's
        transform is read through the record+4 (matrix) / record+8 (position) pointers."""
        table_offset = self.header["prop_table_offset"]
        if not (0 < table_offset < self.buffer_length - 4):
            print("GMS prop table offset is out of range; no transforms will be applied.")
            return

        reader.seek(table_offset)
        prop_count = reader.uint32()
        if table_offset + 4 + 8 * prop_count > self.buffer_length:
            print(f"GMS prop table ({prop_count} entries) overruns the file; skipping.")
            return

        print(f"Prop table: {prop_count} entries at 0x{table_offset:08X}")

        entries = [reader.uint64() for _ in range(prop_count)]

        out_of_range = 0
        rotated = 0
        for packed in entries:
            record_offset = (packed & GMS_PROP_OFFSET_MASK) * GMS_PROP_OFFSET_SHIFT
            depth_level   = (packed >> GMS_PROP_DEPTH_SHIFT)
            is_root       = bool(packed & GMS_PROP_ROOT_FLAG)

            # The record must have room for its descriptor, the two transform pointers and the
            # model reference (record .. record+16).
            if record_offset < 0 or record_offset + GMS_PROP_MODEL_REF_OFFSET + 4 > self.buffer_length:
                out_of_range += 1
                continue

            reader.seek(record_offset + GMS_PROP_DESCRIPTOR_OFFSET)
            descriptor = reader.uint32()

            reader.seek(record_offset + GMS_PROP_MATRIX_PTR_OFFSET)
            matrix_pointer = reader.uint32()
            reader.seek(record_offset + GMS_PROP_POSITION_PTR_OFFSET)
            position_pointer = reader.uint32()

            reader.seek(record_offset + GMS_PROP_MODEL_REF_OFFSET)
            model_reference = reader.uint32()

            position = self.read_position(position_pointer)
            if position is None: continue  # no sane position -> can't place this prop

            rotation = self.read_rotation(matrix_pointer)
            if rotation is not None: rotated += 1

            record = self.build_prop_record(record_offset, descriptor, model_reference, position, rotation, depth_level, is_root)
            self.props.append(record)

            if model_reference != 0:
                self.transforms_by_model.setdefault(model_reference, []).append({"position": position, "rotation": rotation})

        if out_of_range: print(f"  {out_of_range} prop entries pointed outside the file and were skipped.")
        print(f"  {len(self.props)} props decoded ({rotated} with a recovered rotation), {sum(len(v) for v in self.transforms_by_model.values())} carry a model reference.")

    def read_position(self, position_pointer: int) -> Optional[tuple[float, float, float]]:
        """Follow a record's position pointer to a vec3. Returns None when the pointer is out of
        range or the coordinates are NaN/inf/absurd (so a bad record can't drag a mesh to
        infinity)."""
        if not (0 < position_pointer <= self.buffer_length - 4 * GMS_POSITION_FLOATS): return None
        try:
            position = struct.unpack_from("<3f", self.buffer, position_pointer)
        except struct.error:
            return None
        if not all(abs(component) < GMS_POSITION_SANITY_LIMIT and component == component for component in position): return None
        return position

    def read_rotation(self, matrix_pointer: int) -> Optional[tuple[float, ...]]:
        """Follow a record's matrix pointer to a 3x3 row-major rotation and return it as a flat
        9-tuple, but ONLY if it passes an orthonormality test. Returns None otherwise, in which
        case the handler places the mesh with identity orientation rather than a garbage rotation.

        This is deliberately strict: it is the same test that pinned the pointer at ~100% across
        all four games, so a matrix that fails it is almost certainly a record without its own
        rotation block rather than a valid-but-unusual matrix."""
        if not (0 < matrix_pointer <= self.buffer_length - 4 * GMS_MATRIX_FLOATS): return None
        try:
            matrix = struct.unpack_from("<9f", self.buffer, matrix_pointer)
        except struct.error:
            return None
        if not all(value == value and abs(value) < 1.0e6 for value in matrix): return None
        if not self.is_orthonormal(matrix): return None
        return matrix

    def is_orthonormal(self, matrix: tuple[float, ...]) -> bool:
        """True when the flat 9-tuple's three rows are unit-length and mutually orthogonal within
        tolerance. Rejects scale/shear/garbage while accepting genuine rotations."""
        rows = (matrix[0:3], matrix[3:6], matrix[6:9])
        for row in rows:
            length_squared = row[0] * row[0] + row[1] * row[1] + row[2] * row[2]
            if abs(length_squared - 1.0) > GMS_MATRIX_UNIT_TOLERANCE: return False

        def dot(a: tuple, b: tuple) -> float: return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
        if abs(dot(rows[0], rows[1])) > GMS_MATRIX_ORTHO_TOLERANCE: return False
        if abs(dot(rows[0], rows[2])) > GMS_MATRIX_ORTHO_TOLERANCE: return False
        if abs(dot(rows[1], rows[2])) > GMS_MATRIX_ORTHO_TOLERANCE: return False
        return True

    # =====================================================
    # PROP RECORD BUILDER
    # =====================================================

    def build_prop_record(self, record_offset: int, descriptor: int, model_reference: int, position: tuple[float, float, float], rotation: Optional[tuple[float, ...]], depth_level: int, is_root: bool) -> dict:
        """Pack one prop's decoded data into a record dict.

        `model_reference` is deliberately left raw: the model handler resolves it against the PRM,
        and its meaning is game-dependent (byte offset on the classic games, block-table index on
        Blood Money). `rotation` is a flat 9-tuple (row-major 3x3) or None."""
        return {
            "record_offset":   record_offset,
            "descriptor":      descriptor,
            "model_reference": model_reference,
            "position":        position,
            "rotation":        rotation,
            "depth_level":     depth_level,
            "is_root":         is_root,
        }

    # =====================================================
    # LOOKUP HELPERS
    # =====================================================

    def transforms_for(self, model_reference: int) -> list[dict]:
        """Every transform (position + rotation) at which the given model reference is placed.
        Empty when the primitive is never referenced by a prop (common: ~35% of primitives in
        every sample)."""
        return self.transforms_by_model.get(model_reference, [])

    def positions_for(self, model_reference: int) -> list[tuple[float, float, float]]:
        """Backwards-compatible view: just the world positions for a model reference."""
        return [transform["position"] for transform in self.transforms_by_model.get(model_reference, [])]

# =====================================================================================================================================================
# COMPANION FILE DISCOVERY
# =====================================================================================================================================================

def find_companion_gms(prm_path: str) -> Optional[str]:
    """Locate the .GMS sitting beside a .PRM with the same stem.

    Case varies across releases and extraction tools (`.GMS`, `.gms`), and Dody's own working
    copies carry suffixes like `_dec` / `_decompressed`, so we try the exact stem first and then
    fall back to a prefix match in the same directory."""
    prm_file = Path(prm_path)
    directory = prm_file.parent
    stem = prm_file.stem

    for candidate in (directory / f"{stem}.GMS", directory / f"{stem}.gms"):
        if candidate.exists(): return str(candidate)

    # Fallback: any .GMS in the same folder whose stem starts with the PRM's stem. Catches the
    # "<name>_decompressed.gms" convention without hard-coding the suffix list.
    try:
        for entry in sorted(directory.iterdir()):
            if entry.suffix.lower() == ".gms" and entry.stem.lower().startswith(stem.lower()): return str(entry)
    except OSError: pass

    return None
