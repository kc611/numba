"""
Typing declarations for np.timedelta64.
"""


import numpy as np
import operator

from numba.core import types, errors
from numba.core.typing.templates import (
    AbstractTemplate, infer_global, signature,
    infer_getattr, AttributeTemplate
)
from numba.np import npdatetime_helpers
from functools import total_ordering
from numba.core.datamodel.registry import register_default
from llvmlite import ir
from numba.core.datamodel.models import PrimitiveModel
from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.extending import overload
from numba.core.errors import NumbaTypeError

from numba.core.typing.templates import (AbstractTemplate, infer_global,
                                         signature)
from numba.core.typing.asnumbatype import as_numba_type

from numba.core.imputils import (lower_builtin, impl_ret_untracked)

numpy_version = tuple(map(int, np.__version__.split('.')[:2]))

class _NPDatetimeBase(types.Type):
    """
    Common base class for np.datetime64 and np.timedelta64.
    """

    def __init__(self, unit, *args, **kws):
        name = '%s[%s]' % (self.type_name, unit)
        self.unit = unit
        self.unit_code = npdatetime_helpers.DATETIME_UNITS[self.unit]
        super(_NPDatetimeBase, self).__init__(name, *args, **kws)

    def __lt__(self, other):
        if self.__class__ is not other.__class__:
            return NotImplemented
        # A coarser-grained unit is "smaller", i.e. less precise values
        # can be represented (but the magnitude of representable values is
        # also greater...).
        return self.unit_code < other.unit_code

    def cast_python_value(self, value):
        cls = getattr(np, self.type_name)
        if self.unit:
            return cls(value, self.unit)
        else:
            return cls(value)


@total_ordering
class NPTimedelta(_NPDatetimeBase):
    type_name = 'timedelta64'

@total_ordering
class NPDatetime(_NPDatetimeBase):
    type_name = 'datetime64'

@register_default(NPDatetime)
@register_default(NPTimedelta)
class NPDatetimeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir.IntType(64)
        super(NPDatetimeModel, self).__init__(dmm, fe_type, be_type)


@box(NPDatetime)
def box_npdatetime(typ, val, c):
    return c.pyapi.create_np_datetime(val, typ.unit_code)

@unbox(NPDatetime)
def unbox_npdatetime(typ, obj, c):
    val = c.pyapi.extract_np_datetime(obj)
    return NativeValue(val, is_error=c.pyapi.c_api_error())


@box(NPTimedelta)
def box_nptimedelta(typ, val, c):
    return c.pyapi.create_np_timedelta(val, typ.unit_code)

@unbox(NPTimedelta)
def unbox_nptimedelta(typ, obj, c):
    val = c.pyapi.extract_np_timedelta(obj)
    return NativeValue(val, is_error=c.pyapi.c_api_error())


# timedelta64-only operations

class TimedeltaUnaryOp(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 2:
            # Guard against binary + and -
            return
        op, = args
        if not isinstance(op, NPTimedelta):
            return
        return signature(op, op)


class TimedeltaBinOp(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary + and -
            return
        left, right = args
        if not all(isinstance(tp, NPTimedelta) for tp in args):
            return
        if npdatetime_helpers.can_cast_timedelta_units(left.unit, right.unit):
            return signature(right, left, right)
        elif npdatetime_helpers.can_cast_timedelta_units(right.unit, left.unit):
            return signature(left, left, right)


class TimedeltaCmpOp(AbstractTemplate):

    def generic(self, args, kws):
        # For equality comparisons, all units are inter-comparable
        left, right = args
        if not all(isinstance(tp, NPTimedelta) for tp in args):
            return
        return signature(types.boolean, left, right)


class TimedeltaOrderedCmpOp(AbstractTemplate):

    def generic(self, args, kws):
        # For ordered comparisons, units must be compatible
        left, right = args
        if not all(isinstance(tp, NPTimedelta) for tp in args):
            return
        if (npdatetime_helpers.can_cast_timedelta_units(left.unit, right.unit) or
            npdatetime_helpers.can_cast_timedelta_units(right.unit, left.unit)):
            return signature(types.boolean, left, right)


class TimedeltaMixOp(AbstractTemplate):

    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        ({int, float}, timedelta64) -> timedelta64
        """
        left, right = args
        if isinstance(right, NPTimedelta):
            td, other = right, left
            sig_factory = lambda other: signature(td, other, td)
        elif isinstance(left, NPTimedelta):
            td, other = left, right
            sig_factory = lambda other: signature(td, td, other)
        else:
            return
        if not isinstance(other, (types.Float, types.Integer)):
            return
        # Force integer types to convert to signed because it matches
        # timedelta64 semantics better.
        if isinstance(other, types.Integer):
            other = types.int64
        return sig_factory(other)


class TimedeltaDivOp(AbstractTemplate):

    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        (timedelta64, timedelta64) -> float
        """
        left, right = args
        if not isinstance(left, NPTimedelta):
            return
        if isinstance(right, NPTimedelta):
            if (npdatetime_helpers.can_cast_timedelta_units(left.unit, right.unit)
                or npdatetime_helpers.can_cast_timedelta_units(right.unit, left.unit)):
                return signature(types.float64, left, right)
        elif isinstance(right, (types.Float)):
            return signature(left, left, right)
        elif isinstance(right, (types.Integer)):
            # Force integer types to convert to signed because it matches
            # timedelta64 semantics better.
            return signature(left, left, types.int64)


@infer_global(operator.pos)
class TimedeltaUnaryPos(TimedeltaUnaryOp):
    key = operator.pos

@infer_global(operator.neg)
class TimedeltaUnaryNeg(TimedeltaUnaryOp):
    key = operator.neg

@infer_global(operator.add)
@infer_global(operator.iadd)
class TimedeltaBinAdd(TimedeltaBinOp):
    key = operator.add

@infer_global(operator.sub)
@infer_global(operator.isub)
class TimedeltaBinSub(TimedeltaBinOp):
    key = operator.sub

@infer_global(operator.mul)
@infer_global(operator.imul)
class TimedeltaBinMult(TimedeltaMixOp):
    key = operator.mul

@infer_global(operator.truediv)
@infer_global(operator.itruediv)
class TimedeltaTrueDiv(TimedeltaDivOp):
    key = operator.truediv

@infer_global(operator.floordiv)
@infer_global(operator.ifloordiv)
class TimedeltaFloorDiv(TimedeltaDivOp):
    key = operator.floordiv

if numpy_version >= (1, 25):
    @infer_global(operator.eq)
    class TimedeltaCmpEq(TimedeltaOrderedCmpOp):
        key = operator.eq

    @infer_global(operator.ne)
    class TimedeltaCmpNe(TimedeltaOrderedCmpOp):
        key = operator.ne
else:
    @infer_global(operator.eq)
    class TimedeltaCmpEq(TimedeltaCmpOp):
        key = operator.eq

    @infer_global(operator.ne)
    class TimedeltaCmpNe(TimedeltaCmpOp):
        key = operator.ne

@infer_global(operator.lt)
class TimedeltaCmpLt(TimedeltaOrderedCmpOp):
    key = operator.lt

@infer_global(operator.le)
class TimedeltaCmpLE(TimedeltaOrderedCmpOp):
    key = operator.le

@infer_global(operator.gt)
class TimedeltaCmpGt(TimedeltaOrderedCmpOp):
    key = operator.gt

@infer_global(operator.ge)
class TimedeltaCmpGE(TimedeltaOrderedCmpOp):
    key = operator.ge


@infer_global(abs)
class TimedeltaAbs(TimedeltaUnaryOp):
    pass


# datetime64 operations

@infer_global(operator.add)
@infer_global(operator.iadd)
class DatetimePlusTimedelta(AbstractTemplate):
    key = operator.add

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary +
            return
        left, right = args
        if isinstance(right, NPTimedelta):
            dt = left
            td = right
        elif isinstance(left, NPTimedelta):
            dt = right
            td = left
        else:
            return
        if isinstance(dt, NPDatetime):
            unit = npdatetime_helpers.combine_datetime_timedelta_units(dt.unit,
                                                                       td.unit)
            if unit is not None:
                return signature(NPDatetime(unit), left, right)

@infer_global(operator.sub)
@infer_global(operator.isub)
class DatetimeMinusTimedelta(AbstractTemplate):
    key = operator.sub

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary -
            return
        dt, td = args
        if isinstance(dt, NPDatetime) and isinstance(td,
                                                           NPTimedelta):
            unit = npdatetime_helpers.combine_datetime_timedelta_units(dt.unit,
                                                                       td.unit)
            if unit is not None:
                return signature(NPDatetime(unit), dt, td)

@infer_global(operator.sub)
class DatetimeMinusDatetime(AbstractTemplate):
    key = operator.sub

    def generic(self, args, kws):
        if len(args) == 1:
            # Guard against unary -
            return
        left, right = args
        if isinstance(left, NPDatetime) and isinstance(right,
                                                             NPDatetime):
            unit = npdatetime_helpers.get_best_unit(left.unit, right.unit)
            return signature(NPTimedelta(unit), left, right)


class DatetimeCmpOp(AbstractTemplate):

    def generic(self, args, kws):
        # For datetime64 comparisons, all units are inter-comparable
        left, right = args
        if not all(isinstance(tp, NPDatetime) for tp in args):
            return
        return signature(types.boolean, left, right)


@infer_global(operator.eq)
class DatetimeCmpEq(DatetimeCmpOp):
    key = operator.eq

@infer_global(operator.ne)
class DatetimeCmpNe(DatetimeCmpOp):
    key = operator.ne

@infer_global(operator.lt)
class DatetimeCmpLt(DatetimeCmpOp):
    key = operator.lt

@infer_global(operator.le)
class DatetimeCmpLE(DatetimeCmpOp):
    key = operator.le

@infer_global(operator.gt)
class DatetimeCmpGt(DatetimeCmpOp):
    key = operator.gt

@infer_global(operator.ge)
class DatetimeCmpGE(DatetimeCmpOp):
    key = operator.ge


@infer_global(npdatetime_helpers.datetime_minimum)
@infer_global(npdatetime_helpers.datetime_maximum)
class DatetimeMinMax(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        error_msg = "DatetimeMinMax requires both arguments to be NPDatetime type or both arguments to be NPTimedelta types"
        assert isinstance(args[0], (NPDatetime, NPTimedelta)), error_msg
        if isinstance(args[0], NPDatetime):
            if not isinstance(args[1], NPDatetime):
                raise errors.TypingError(error_msg)
        else:
            if not isinstance(args[1], NPTimedelta):
                raise errors.TypingError(error_msg)
        return signature(args[0], *args)

@infer_getattr
class NPTimedeltaAttribute(AttributeTemplate):
    key = NPTimedelta

    def resolve___class__(self, ty):
        return types.NumberClass(ty)


@infer_getattr
class NPDatetimeAttribute(AttributeTemplate):
    key = NPDatetime

    def resolve___class__(self, ty):
        return types.NumberClass(ty)


@overload(isinstance)
def ol_isinstance(var, typs):

    def true_impl(var, typs):
        return True

    def false_impl(var, typs):
        return False

    var_ty = as_numba_type(var)

    if isinstance(var_ty, types.Optional):
        msg = f'isinstance cannot handle optional types. Found: "{var_ty}"'
        raise NumbaTypeError(msg)

    supported_var_ty = (NPDatetime, NPTimedelta,)
    if not isinstance(var_ty, supported_var_ty):
        msg = f'isinstance() does not support variables of type "{var_ty}".'
        raise NumbaTypeError(msg)

    t_typs = typs

    # Check the types that the var can be an instance of, it'll be a scalar,
    # a unituple or a tuple.
    if isinstance(t_typs, types.UniTuple):
        # corner case - all types in isinstance are the same
        t_typs = (t_typs.key[0])

    if not isinstance(t_typs, types.Tuple):
        t_typs = (t_typs, )

    for typ in t_typs:

        if isinstance(typ, types.Function):
            key = typ.key[0]  # functions like int(..), float(..), str(..)
        elif isinstance(typ, types.ClassType):
            key = typ  # jitclasses
        else:
            key = typ.key

        # corner cases for bytes, range, ...
        # avoid registering those types on `as_numba_type`
        types_not_registered = {
            bytes: types.Bytes,
            range: types.RangeType,
            dict: (types.DictType, types.LiteralStrKeyDict),
            list: types.List,
            tuple: types.BaseTuple,
            set: types.Set,
        }
        if key in types_not_registered:
            if isinstance(var_ty, types_not_registered[key]):
                return true_impl
            continue

        if isinstance(typ, types.TypeRef):
            if key not in (types.ListType, types.DictType):
                msg = ("Numba type classes (except numba.typed.* container "
                       "types) are not supported.")
                raise NumbaTypeError(msg)
            return true_impl if type(var_ty) is key else false_impl
        else:
            numba_typ = as_numba_type(key)
            if var_ty == numba_typ:
                return true_impl
            if isinstance(numba_typ, (NPDatetime, NPTimedelta)):
                if isinstance(var_ty, type(numba_typ)):
                    return true_impl

    return false_impl
