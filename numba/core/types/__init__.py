import struct

import numpy as np
from numba.core import utils

from .abstract import *
from .containers import *
from .functions import *
from .iterators import *
from .misc import *
from .npytypes import *
from .scalars import *
from .function_type import *

# Short names

pyobject = PyObject('pyobject')
ffi_forced_object = Opaque('ffi_forced_object')
ffi = Opaque('ffi')
none = NoneType('none')
ellipsis = EllipsisType('...')
Any = Phantom('any')
undefined = Undefined('undefined')
py2_string_type = Opaque('str')
unicode_type = UnicodeType('unicode_type')
string = unicode_type
unknown = Dummy('unknown')
npy_rng = NumPyRandomGeneratorType('rng')
npy_bitgen = NumPyRandomBitGeneratorType('bitgen')

# _undef_var is used to represent undefined variables in the type system.
_undef_var = UndefVar('_undef_var')

code_type = Opaque('code')
pyfunc_type = Opaque('pyfunc')

# No operation is defined on voidptr
# Can only pass it around
voidptr = RawPointer('void*')

# optional types
optional = Optional
deferred_type = DeferredType
slice2_type = SliceType('slice<a:b>', 2)
slice3_type = SliceType('slice<a:b:c>', 3)
void = none

if config.USE_LEGACY_TYPE_SYSTEM:
    boolean = bool_ = Boolean('bool')

    byte = uint8 = Integer('uint8')
    uint16 = Integer('uint16')
    uint32 = Integer('uint32')
    uint64 = Integer('uint64')

    int8 = Integer('int8')
    int16 = Integer('int16')
    int32 = Integer('int32')
    int64 = Integer('int64')
    intp = int32 if utils.MACHINE_BITS == 32 else int64
    uintp = uint32 if utils.MACHINE_BITS == 32 else uint64
    intc = int32 if struct.calcsize('i') == 4 else int64
    uintc = uint32 if struct.calcsize('I') == 4 else uint64
    ssize_t = int32 if struct.calcsize('n') == 4 else int64
    size_t = uint32 if struct.calcsize('N') == 4 else uint64

    float32 = Float('float32')
    float64 = Float('float64')
    float16 = Float('float16')

    complex64 = Complex('complex64', float32)
    complex128 = Complex('complex128', float64)

    range_iter32_type = RangeIteratorType(int32)
    range_iter64_type = RangeIteratorType(int64)
    unsigned_range_iter64_type = RangeIteratorType(uint64)
    range_state32_type = RangeType(int32)
    range_state64_type = RangeType(int64)
    unsigned_range_state64_type = RangeType(uint64)

    signed_domain = frozenset([int8, int16, int32, int64])
    unsigned_domain = frozenset([uint8, uint16, uint32, uint64])
    integer_domain = signed_domain | unsigned_domain
    real_domain = frozenset([float32, float64])
    complex_domain = frozenset([complex64, complex128])
    number_domain = real_domain | integer_domain | complex_domain

    # Aliases to NumPy type names

    b1 = bool_
    i1 = int8
    i2 = int16
    i4 = int32
    i8 = int64
    u1 = uint8
    u2 = uint16
    u4 = uint32
    u8 = uint64

    f2 = float16
    f4 = float32
    f8 = float64

    c8 = complex64
    c16 = complex128

    float_ = float32
    double = float64

    _make_signed = lambda x: globals()["int%d" % (np.dtype(x).itemsize * 8)]
    _make_unsigned = lambda x: globals()["uint%d" % (np.dtype(x).itemsize * 8)]

    char = _make_signed(np.byte)
    uchar = byte = _make_unsigned(np.byte)
    short = _make_signed(np.short)
    ushort = _make_unsigned(np.short)
    int_ = _make_signed(np.int_)
    uint = _make_unsigned(np.int_)
    intc = _make_signed(np.intc) # C-compat int
    uintc = _make_unsigned(np.uintc) # C-compat uint
    long_ = _make_signed(np.int_)  # C-compat long
    ulong = _make_unsigned(np.int_)  # C-compat ulong
    longlong = _make_signed(np.longlong)
    ulonglong = _make_unsigned(np.longlong)

    all_str = '''
    int8
    int16
    int32
    int64
    uint8
    uint16
    uint32
    uint64
    intp
    uintp
    intc
    uintc
    ssize_t
    size_t
    boolean
    float32
    float64
    complex64
    complex128
    bool_
    byte
    char
    uchar
    short
    ushort
    int_
    uint
    long_
    ulong
    longlong
    ulonglong
    float_
    double
    void
    none
    b1
    i1
    i2
    i4
    i8
    u1
    u2
    u4
    u8
    f4
    f8
    c8
    c16
    optional
    ffi_forced_object
    ffi
    deferred_type
    '''
else:
    from .python_types import *
    from .numpy_types import *
    from .machine_types import *

    # Machine Integers
    byte = MachineInteger('c_int8')

    ### Python Datatypes ###
    # Python Integers
    boolean = PythonBoolean('py_bool')
    py_int32 = PythonInteger('py_int32')
    py_int64 = PythonInteger('py_int64')

    intp = py_intp = py_int32 if utils.MACHINE_BITS == 32 else py_int64

    # Python Float
    float64 = py_float64 = PythonFloat('py_float64')

    # Python Complex
    complex128 = py_complex128 = PythonComplex('py_complex128', py_float64)

    # Numpy Integers
    np_int8 = NumPyInteger('np_int8')
    np_int16 = NumPyInteger('np_int16')
    np_int32 = NumPyInteger('np_int32')
    np_int64 = NumPyInteger('np_int64')
    np_uint8 = NumPyInteger('np_uint8')
    np_uint16 = NumPyInteger('np_uint16')
    np_uint32 = NumPyInteger('np_uint32')
    np_uint64 = NumPyInteger('np_uint64')

    np_intp = np_int32 if utils.MACHINE_BITS == 32 else np_int64
    np_uintp = np_uint32 if utils.MACHINE_BITS == 32 else np_uint64

    all_str = '''
    py_int32
    py_int64
    py_intp
    np_int8
    np_int16
    np_int32
    np_int64
    np_uint8
    np_uint16
    np_uint32
    np_uint64
    np_intp
    np_uintp
    '''

__all__ = all_str.split()
