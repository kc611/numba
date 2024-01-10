
import enum

import numpy as np

from .abstract import Dummy, Hashable, Literal, Number, Type
from .utils import parse_integer_bitwidth, parse_integer_signed
from functools import total_ordering, cached_property
from numba.core import utils
from numba.core.typeconv import Conversion
from numba.np import npdatetime_helpers


@total_ordering
class MachineInteger(Number):
    def __init__(self, name, bitwidth=None, signed=None):
        super(MachineInteger, self).__init__(name)
        if bitwidth is None:
            bitwidth = parse_integer_bitwidth(name)
        if signed is None:
            signed = parse_integer_signed(name)
        self.bitwidth = bitwidth
        self.signed = signed

    @classmethod
    def from_bitwidth(cls, bitwidth, signed=True):
        name = ('int%d' if signed else 'uint%d') % bitwidth
        return cls(name)

    def cast_python_value(self, value):
        return getattr(np, self.name)(value)

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


class MachineIntegerLiteral(Literal, MachineInteger):
    def __init__(self, value):
        self._literal_init(value)
        name = 'Literal[int]({})'.format(value)
        basetype = self.literal_type
        MachineInteger.__init__(
            self,
            name=name,
            bitwidth=basetype.bitwidth,
            signed=basetype.signed,
            )

    def can_convert_to(self, typingctx, other):
        conv = typingctx.can_convert(self.literal_type, other)
        if conv is not None:
            return max(conv, Conversion.promote)


Literal.ctor_map[int] = MachineIntegerLiteral