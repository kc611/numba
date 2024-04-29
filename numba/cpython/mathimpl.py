import sys
from numba.core.utils import _RedirectSubpackage
from numba.core import config

if config.USE_LEGACY_TYPE_SYSTEM: # type: ignore
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.cpython.old_mathimpl"
    )
else:
    sys.modules[__name__] = _RedirectSubpackage(
        locals(), "numba.cpython.new_mathimpl"
    )
