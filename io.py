# ===================================================
# I/O HELPER / LAYER
# For reading and writing binary data from/in files.
# ===================================================

import os, bpy, struct, math, mathutils

# =============
# READER CLASS
# =============

class Reader(object):
    """Data parser class. Used for reading binary data from a file!"""
    def __init__(self, buf: bytes, is_little_endian: bool = True):
        """Construct the Reader object. Input data and if we want to use Little Endian for our reading process. (Default is BE)"""
        super().__init__()

        # ==============
        # CLASS MEMBERS
        # ==============

        self.offset: int = 0
        """The offset or position we are currently at in reading the file."""

        self.data: bytes = buf
        """The bytes that the current instance of Reader is currently pulling from."""

        self.LE: bool = is_little_endian
        """Is this file being read in Little Endian?"""

        self.length: int = len(buf)
        """Total number of bytes in the data buffer provided."""

    # =============
    # CORE METHODS
    # =============

    def tell(self) -> int:
        """Where are we in our data buffer? Returns the offset."""
        return self.offset
    
    def skip(self, skip_by: int):
        """Move the offset forward by the given number of bytes."""
        self.offset += skip_by

    def seek(self, position: int):
        """Manually set the read position offset at the given position."""
        self.offset = position
    
    def read(self, fmt) -> tuple:
        """Using `struct.unpack_from()`, return the needed value and advance the position forward by the number of bytes the desired type occupies."""
        result = struct.unpack_from(("" if self.LE else ">") + fmt, self.data, self.offset)
        self.offset += struct.calcsize(fmt)
        return result
    
    def read_string(self, length: int, encoding: str = 'utf-8', errors: str = 'replace') -> str:
        """Credit: arcusmaximus, Modified by Dodylectable | Read a string from `x` amount of bytes."""
        if length <= 0: return ""
            
        result = self.read_bytes(length)

        if hasattr(result, 'tobytes'): raw_bytes = result.tobytes()
        else: raw_bytes = result
            
        return raw_bytes.decode(encoding, errors=errors)
    
    def read_null_terminated_string(self, encoding: str = 'utf-8', errors: str = 'replace') -> str:
        """Read a series of bytes until we're met with a null byte then return that as a string."""
        buffer = bytearray()
        while True:
            byte = self.read_bytes(1)
            if not byte or byte == b'\x00': break
            
            buffer.extend(byte)

        return buffer.decode(encoding, errors=errors)
    
    def read_prefixed_string_int(self, encoding: str = 'utf-8', errors: str = 'replace') -> str:
        """Read string length (based on an unsigned integer value) then read the actual string per said length."""
        length = self.uint32()
        return self.read_string(length, encoding, errors)

    def read_prefixed_string_short(self, encoding: str = 'utf-8', errors: str = 'replace') -> str:
        """Read string length (based on an unsigned short value) then read the actual string per said length."""
        length = self.ushort()
        return self.read_string(length, encoding, errors)
    
    def read_bytes(self, length: int) -> memoryview:
        """Credit: arcusmaximus | Read a series of `x` bytes."""
        result = self.read_bytes_at(0, length)
        self.offset += length
        return result

    def read_bytes_at(self, offset: int, length: int) -> memoryview:
        """Credit: arcusmaximus | Read raw bytes from the given offset for the given number of bytes forward."""
        return memoryview(self.data)[self.offset + offset:self.offset + offset + length]

    # ===================================================================================================================================================================

    # ===========
    # DATA TYPES 
    # ===========

    # ======
    # BYTES
    # ======
        
    def byte(self) -> int:
        """Read a signed 8-bit integer and advance the position forward by 1 byte."""
        return self.read("b")[0]

    def ubyte(self) -> int:
        """Read an unsigned 8-bit integer and advance the position forward by 1 byte."""
        return self.read("B")[0]
    
    # =======
    # SHORTS
    # =======

    def short(self) -> int:
        """Read a signed 16-bit integer and advance the position forward by 2 bytes."""
        return self.read("h")[0]

    def ushort(self) -> int:
        """Read an unsigned 16-bit integer and advance the position forward by 2 bytes."""
        return self.read("H")[0]
    
    # =========
    # INTEGERS
    # =========

    def int24(self) -> int:
        """Read a signed 24-bit integer and advance the position forward by 3 bytes."""
        b = self.read_bytes(3)
        val = int.from_bytes(b, byteorder='little' if self.LE else 'big', signed=False)
        # Sign extend if the 24th bit (0x800000) is set
        if val & 0x800000: val -= 0x1000000
        return val

    def uint24(self) -> int:
        """Read an unsigned 24-bit integer and advance the position forward by 3 bytes."""
        b = self.read_bytes(3)
        return int.from_bytes(b, byteorder='little' if self.LE else 'big', signed=False)

    def int32(self) -> int:
        """Read a signed 32-bit integer and advance the position forward by 4 bytes."""
        return self.read("i")[0]

    def uint32(self) -> int:
        """Read an unsigned 32-bit integer and advance the position forward by 4 bytes."""
        return self.read("I")[0]

    def int48(self) -> int:
        """Read a signed 48-bit integer and advance the position forward by 6 bytes."""
        b = self.read_bytes(6)
        val = int.from_bytes(b, byteorder='little' if self.LE else 'big', signed=False)
        # Sign extend if the 48th bit (0x800000000000) is set
        if val & 0x800000000000: val -= 0x1000000000000
        return val

    def uint48(self) -> int:
        """Read an unsigned 48-bit integer and advance the position forward by 6 bytes."""
        b = self.read_bytes(6)
        return int.from_bytes(b, byteorder='little' if self.LE else 'big', signed=False)

    def int64(self) -> int:
        """Read a signed 64-bit integer and advance the position forward by 8 bytes."""
        return self.read("q")[0]

    def uint64(self) -> int:
        """Read an unsigned 64-bit integer and advance the position forward by 8 bytes."""
        return self.read("Q")[0]
    
    # =======
    # FLOATS
    # =======

    def hfloat16(self) -> float:
        """Read a signed 16-bit half-precision floating point value and advance the position forward by 2 bytes."""
        return self.read("e")[0]
        
    def float32(self) -> float:
        """Read a signed 32-bit floating point value and advance the position forward by 4 bytes."""
        return self.read("f")[0]
        
    # ===================================================================================================================================================================

    # ========
    # VECTORS
    # ========

    # ======
    # BYTES
    # ======

    def vec3sb(self) -> tuple[int, int, int]:
        """Read three signed 8-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 3 bytes."""
        return self.read("3b")
    
    def vec3ub(self) -> tuple[int, int, int]:
        """Read three unsigned 8-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 3 bytes."""
        return self.read("3B")
    
    def vec4sb(self) -> tuple[int, int, int, int]:
        """Read four signed 8-bit numbers and return them as a 4-point vector tuple. Advances the position forward by 4 bytes."""
        return self.read("4b")
    
    def vec4ub(self) -> tuple[int, int, int, int]:
        """Read four unsigned 8-bit numbers and return them as a 4-point vector tuple. Advances the position forward by 4 bytes."""
        return self.read("4B")
    
    # =======
    # SHORTS
    # =======
    
    def vec2ss(self) -> tuple[int, int]:
        """Read two signed 16-bit numbers and return them as a 2-point vector tuple. Advances the position forward by 4 bytes."""
        return self.read("2h")
    
    def vec2us(self) -> tuple[int, int]:
        """Read two unsigned 16-bit numbers and return them as a 2-point vector tuple. Advances the position forward by 4 bytes."""
        return self.read("2H")
        
    def vec3ss(self) -> tuple[int, int, int]:
        """Read three signed 16-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 6 bytes."""
        return self.read("3h")
    
    def vec3us(self) -> tuple[int, int, int]:
        """Read three unsigned 16-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 6 bytes."""
        return self.read("3H")

    def vec4ss(self) -> tuple[int, int, int, int]:
        """Read four signed 16-bit numbers and return them as a 4-point vector tuple. Advances the position forward by 8 bytes."""
        return self.read("4h")

    def vec4us(self) -> tuple[int, int, int, int]:
        """Read four unsigned 16-bit numbers and return them as a 4-point vector tuple. Advances the position forward by 8 bytes."""
        return self.read("4H")
    
    # =========
    # INTEGERS
    # =========
    
    def vec2si(self) -> tuple[int, int]:
        """Read two 32-bit signed integers and return them as a 2-point vector tuple. Advances the position forward by 8 bytes."""
        return self.read("2i")
    
    def vec2ui(self) -> tuple[int, int]:
        """Read two 32-bit unsigned integers and return them as a 2-point vector tuple. Advances the position forward by 8 bytes."""
        return self.read("2I")
    
    def vec3si(self) -> tuple[int, int, int]:
        """Read three 32-bit signed integers and return them as a 3-point vector tuple. Advances the position forward by 12 bytes."""
        return self.read("3i")
    
    def vec3ui(self) -> tuple[int, int, int]:
        """Read three 32-bit unsigned integers and return them as a 3-point vector tuple. Advances the position forward by 12 bytes."""
        return self.read("3I")

    def vec4si(self) -> tuple[int, int, int, int]:
        """Read four 32-bit signed integers as a 4-point vector tuple. Advances the position forward by 16 bytes."""
        return self.read("4i")

    def vec4ui(self) -> tuple[int, int, int, int]:
        """Read four 32-bit unsigned integers as a 4-point vector tuple. Advances the position forward by 16 bytes."""
        return self.read("4I")
    
    # ============
    # HALF-FLOATS
    # ============

    def vec2hf(self) -> tuple[float, float]:
        """Read two 16-bit half-precision floating point numbers and return them as a 2-point vector tuple. Advances the position forward by 4 bytes."""
        return self.read("2e")

    def vec3hf(self) -> tuple[float, float, float]:
        """Read three 16-bit half-precision floating point numbers and return them as a 3-point vector tuple. Advances the position forward by 6 bytes."""
        return self.read("3e")
    
    def vec4hf(self) -> tuple[float, float, float]:
        """Read four 16-bit half-precision floating point numbers and return them as a 4-point vector tuple. Advances the position forward by 8 bytes."""
        return self.read("4e")
    
    # =======
    # FLOATS
    # =======

    def vec2f(self) -> tuple[float, float]:
        """Read two 32-bit floating point numbers and return them as a 2-point vector tuple. Advances the position forward by 8 bytes."""
        return self.read("2f")
    
    def vec3f(self) -> tuple[float, float, float]:
        """Read three 32-bit floating point numbers and return them as a 3-point vector tuple. Advances the position forward by 12 bytes."""
        return self.read("3f")
    
    def vec4f(self) -> tuple[float, float, float, float]:
        """Read four 32-bit floating point numbers and return them as a 4-point vector tuple. Advances the position forward by 16 bytes."""
        return self.read("4f")
    
# ===================================================================================================================================================================

# =============
# WRITER CLASS
# =============

class Writer(object):
    """Data writer class. Used for writing binary data to a file!"""
    def __init__(self, output_file: str | bytearray | None, is_little_endian: bool = True) -> None:
        """Construct the Writer object. Output data and if we want to use Little Endian for our writing process. (Default is BE)"""
        super().__init__()

        # ==============
        # CLASS MEMBERS
        # ==============

        self.offset: int = 0
        """Current position in writing data."""

        self.file = bytearray([]) if output_file is None else output_file
        """The contents of the written file or the written file's location."""

        self.length: int = 0
        """Length of the current buffer."""

        self.is_LE: bool = is_little_endian
        """Is this file Little Endian?"""

        self.raw = isinstance(self.file, bytearray) or self.file is None
        """Are we writing to a raw array of bytes?"""

    # =============
    # CORE METHODS
    # =============

    def close(self):
        """Closes the file."""
        if self.file and not isinstance(self.file, bytearray):
            self.file.close()
            self.file = None

    def save(self, file_path: str) -> None:
        """Save the data from this object to a file."""
        # Is the file data even valid?
        if self.file:

            # If the folder that the file path given didn't exist then let's make it ourselves
            folder_dir = os.path.dirname(file_path)
            if not os.path.exists(folder_dir): os.makedirs(folder_dir)

            # In binary write mode, write our object's data to the file!
            with open(file_path, "wb") as bin_file: bin_file.write(self.file)

    def write(self, fmt, *args):
        """Write a value to the file with a format string and a value."""
        # Packed value based on format string
        packed_val = struct.pack(("" if self.is_LE else ">") + fmt, *args)

        # Write to raw byte array
        if isinstance(self.file, bytearray): self.file.extend(packed_val)

        elif self.raw: # If we're writing to a raw array or bytearray (in-memory data)
            if self.offset == len(self.file): self.file.extend(packed_val) # Append if we're at the end
            else: # Write data at the current offset
                for index, byte in enumerate(packed_val): self.file[self.offset + index] = byte

        # Write directly to a file
        else: self.file.write(packed_val)

        # Move offset forward by the size of the packed value
        self.offset += struct.calcsize(fmt)

        # Update the length
        if isinstance(self.file, bytearray): self.length = len(self.file)
        else: self.length = self.offset # For file, length is tracked by the offset
        
        return packed_val # Ensure this returns the packed value

    def tell(self) -> int:
        """Where are we currently in writing?"""
        return self.offset if self.raw else self.file.tell()
    
    def seek(self, position: int) -> None:
        """Move the write cursor to a specific location."""
        if (self.raw): self.offset = position
        else: self.file.seek(position)

    def ascii_string(self, text: str) -> None:
        """Write an ASCII-based text string and move the position forward by the number of characters in the string."""
        encoded = text.encode('ascii')
        self.write(f"{len(encoded)}s", encoded)

    def num_string(self, text: str) -> None:
        """Write a string but the length of it first then the string afterwards. Moves the position forward by 4 bytes + the length of the text string."""
        self.uint32(len(text))
        self.ascii_string(text)

    # ===================================================================================================================================================================

    # ===========
    # DATA TYPES 
    # ===========

    # ======
    # BYTES
    # ======

    def byte(self, value: int) -> int:
        """Write a signed 8-bit integer and advance the position forward by 1 byte."""
        return self.write("b", value)[0]

    def ubyte(self, value: int) -> int:
        """Write an unsigned 8-bit integer and advance the position forward by 1 byte."""
        return self.write("B", value)[0]

    # =======
    # SHORTS
    # =======

    def short(self, value: int) -> int:
        """Write a signed 16-bit integer and advance the position forward by 2 bytes."""
        return self.write("h", value)[0]

    def ushort(self, value: int) -> int:
        """Write an unsigned 16-bit integer and advance the position forward by 2 bytes."""
        return self.write("H", value)[0]
    
    # =========
    # INTEGERS
    # =========

    def int32(self, value: int) -> int:
        """Write a signed 32-bit integer and advance the position forward by 4 bytes."""
        return self.write("i", value)[0]

    def uint32(self, value: int) -> int:
        """Write an unsigned 32-bit integer and advance the position forward by 4 bytes."""
        return self.write("I", value)[0]

    def int64(self, value: int) -> int:
        """Write a signed 64-bit integer and advance the position forward by 8 bytes."""
        return self.write("q", value)[0]

    def uint64(self, value: int) -> int:
        """Write an unsigned 64-bit integer and advance the position forward by 8 bytes."""
        return self.write("Q", value)[0]
    
    # =======
    # FLOATS
    # =======

    def hfloat16(self, value: float) -> float:
        """Write a signed 16-bit half-precision floating point value and advance the position forward by 2 bytes."""
        return self.write("e", value)[0]

    def float32(self, value: float) -> float:
        """Write a signed 32-bit floating point value and advance the position forward by 4 bytes."""
        return self.write("f", value)[0]
    
    # ===================================================================================================================================================================

    # ========
    # VECTORS
    # ========

    # ======
    # BYTES
    # ======

    def vec3sb(self, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Write three signed 8-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 3 bytes."""
        return self.write("3b", *value)
    
    def vec3ub(self, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Write three unsigned 8-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 3 bytes."""
        return self.write("3B", *value)
    
    def vec4sb(self, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Write four signed 8-bit numbers and return them as a 4-point vector tuple. Advances the position forward by 4 bytes."""
        return self.write("4b", *value)
    
    def vec4ub(self, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Write four unsigned 8-bit numbers and return them as a 4-point vector tuple. Advances the position forward by 4 bytes."""
        return self.write("4B", *value)

    # =======
    # SHORTS
    # =======

    def vec2ss(self, value: tuple[int, int]) -> tuple[int, int]:
        """Write two signed 16-bit numbers and return them as a 2-point vector tuple. Advances the position forward by 4 bytes."""
        return self.write("2h", *value)
    
    def vec2us(self, value: tuple[int, int]) -> tuple[int, int]:
        """Write two unsigned 16-bit numbers and return them as a 2-point vector tuple. Advances the position forward by 4 bytes."""
        return self.write("2H", *value)
        
    def vec3ss(self, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Write three signed 16-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 6 bytes."""
        return self.write("3h", *value)
    
    def vec3us(self, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Write three unsigned 16-bit numbers and return them as a 3-point vector tuple. Advances the position forward by 6 bytes."""
        return self.write("3H", *value)

    def vec4ss(self, value: tuple[int, int, int, int]) -> bytes:
        """Write four signed 16-bit numbers as a 4-point vector tuple. Advances the position forward by 8 bytes."""
        return self.write("4h", *value)

    def vec4us(self, value: tuple[int, int, int, int]) -> bytes:
        """Write four unsigned 16-bit numbers as a 4-point vector tuple. Advances the position forward by 8 bytes."""
        return self.write("4H", *value)

    # =========
    # INTEGERS
    # =========

    def vec2si(self, value: tuple[int, int]) -> tuple[int, int]:
        """Write two 32-bit signed integers and return them as a 2-point vector tuple. Advances the position forward by 8 bytes."""
        return self.write("2i", *value)
    
    def vec2ui(self, value: tuple[int, int]) -> tuple[int, int]:
        """Write two 32-bit unsigned integers and return them as a 2-point vector tuple. Advances the position forward by 8 bytes."""
        return self.write("2I", *value)
    
    def vec3si(self, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Write three 32-bit signed integers and return them as a 3-point vector tuple. Advances the position forward by 12 bytes."""
        return self.write("3i", *value)
    
    def vec3ui(self, value: tuple[int, int, int]) -> tuple[int, int, int]:
        """Write three 32-bit unsigned integers and return them as a 3-point vector tuple. Advances the position forward by 12 bytes."""
        return self.write("3I", *value)

    def vec4si(self, value: tuple[int, int, int, int]) -> bytes:
        """Write four 32-bit signed integers and return them as a 4-point vector tuple. Advances the position forward by 16 bytes."""
        return self.write("4i", *value)

    def vec4ui(self, value: tuple[int, int, int, int]) -> bytes:
        """Write four 32-bit unsigned integers and return them as a 4-point vector tuple. Advances the position forward by 16 bytes."""
        return self.write("4I", *value)

    # ============
    # HALF-FLOATS
    # ============

    def vec2hf(self, value: tuple[float, float]) -> tuple[float, float]:
        """Write two 16-bit half-precision floating point numbers and return them as a 2-point vector tuple. Advances the position forward by 4 bytes."""
        return self.write("2e", *value)

    def vec3hf(self, value: tuple[float, float, float]) -> bytes:
        """Write three 16-bit half-precision floating point numbers and return them as a 3-point vector tuple. Advances the position forward by 6 bytes."""
        return self.write("3e", *value)
    
    def vec4hf(self, value: tuple[float, float, float]) -> bytes:
        """Write four 16-bit half-precision floating point numbers and return them as a 4-point vector tuple. Advances the position forward by 8 bytes."""
        return self.write("4e", *value)

    # =======
    # FLOATS
    # =======

    def vec2f(self, value: tuple[float, float]) -> tuple[float, float]:
        """Write two 32-bit floating point numbers and return them as a 2-point vector tuple. Advances the position forward by 8 bytes."""
        return self.write("2f", *value)

    def vec3f(self, value: tuple[float, float, float]) -> tuple[float, float, float]:
        """Write three 32-bit floating point numbers and return them as a 3-point vector tuple. Advances the position forward by 12 bytes."""
        return self.write("3f", *value)
    
    def vec4f(self, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        """Write four 32-bit floating point numbers and return them as a 4-point vector tuple. Advances the position forward by 16 bytes."""
        return self.write("4f", *value)
    
# ===================================================================================================================================================================
