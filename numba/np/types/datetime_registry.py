from numba.core.pythonapi import box, unbox, NativeValue
from numba.core.extending import (overload, intrinsic)
from numba.np.types import NPDatetime, NPTimedelta
from numba import types
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.errors import NumbaTypeError
from numba.cpython.builtins import cast_int

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

@overload(int)
def ol_int(x):
    if isinstance(x, (NPDatetime, NPTimedelta)):
        if isinstance(x, NPDatetime):
            if x.unit != 'ns':
                raise NumbaTypeError(f"Only datetime64[ns] can be converted, but got datetime64[{x.unit}]")

        def impl(x):
            return cast_int(x)

        return impl
