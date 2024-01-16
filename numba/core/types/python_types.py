import enum

import numpy as np

from .abstract import Dummy, Hashable, Literal, Number, Type
from .utils import parse_integer_bitwidth, parse_integer_signed
from .scalars import BaseInteger, BaseIntegerLiteral, BaseBoolean, BaseBooleanLiteral, BaseFloat, BaseComplex
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers



@total_ordering
class PythonInteger(BaseInteger):
    def __init__(self, name, bitwidth=None, signed=None):
        super(PythonInteger, self).__init__(name)
        if bitwidth is None:
            bitwidth = parse_integer_bitwidth(name)
        if signed is None:
            signed = parse_integer_signed(name)
        self.bitwidth = bitwidth
        self.signed = signed

    @classmethod
    def from_bitwidth(cls, bitwidth, signed=True):
        name = ('py_int%d' if signed else 'py_uint%d') % bitwidth
        return cls(name)

    def cast_python_value(self, value):
        return int(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        if self.signed != other.signed:
            return NotImplemented
        return self.bitwidth < other.bitwidth

    @property
    def maxval(self):
        """
        The maximum value representable by this type.
        """
        if self.signed:
            return (1 << (self.bitwidth - 1)) - 1
        else:
            return (1 << self.bitwidth) - 1

    @property
    def minval(self):
        """
        The minimal value representable by this type.
        """
        if self.signed:
            return -(1 << (self.bitwidth - 1))
        else:
            return 0


class PythonIntegerLiteral(BaseIntegerLiteral, PythonInteger):
    def __init__(self, value):
        self._literal_init(value)
        name = 'Literal[int]({})'.format(value)
        basetype = self.literal_type
        PythonInteger.__init__(
            self,
            name=name,
            bitwidth=basetype.bitwidth,
            signed=basetype.signed,
            )

    def can_convert_to(self, typingctx, other):
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, Conversion.promote)


Literal.ctor_map[int] = PythonIntegerLiteral

class PythonBoolean(BaseBoolean):
    def cast_python_value(self, value):
        return bool(value)


class PythonBooleanLiteral(BaseBooleanLiteral, PythonBoolean):

    def __init__(self, value):
        self._literal_init(value)
        name = 'Literal[bool]({})'.format(value)
        PythonBoolean.__init__(
            self,
            name=name
            )

    def can_convert_to(self, typingctx, other):
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, Conversion.promote)


Literal.ctor_map[bool] = PythonBooleanLiteral

@total_ordering
class PythonFloat(BaseFloat):
    def __init__(self, *args, **kws):
        super(PythonFloat, self).__init__(*args, **kws)
        # Determine bitwidth
        assert self.name.startswith('py_float')
        bitwidth = int(self.name[8:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return float(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth


@total_ordering
class PythonComplex(BaseComplex):
    def __init__(self, name, underlying_float, **kwargs):
        super(PythonComplex, self).__init__(name, **kwargs)
        self.underlying_float = underlying_float
        # Determine bitwidth
        assert self.name.startswith('py_complex')
        bitwidth = int(self.name[10:])
        self.bitwidth = bitwidth

    def cast_python_value(self, value):
        return complex(value)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        return self.bitwidth < other.bitwidth
