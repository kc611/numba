"""NumPy-specific context plugins.

Importing this module registers the NumPy typing declarations and target
implementations with the generic context plugin registries.
"""
from numba.core.context_plugins import typing_plugins, target_plugins


@typing_plugins.register('numpy')
def register_numpy_typing(context):
    from numba.core.typing import npydecl
    context.install_registry(npydecl.registry)


@target_plugins.register('numpy')
def register_numpy_target(context):
    from numba.np import npyimpl
    context.install_registry(npyimpl.registry)
