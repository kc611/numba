import logging
import warnings

from importlib import metadata as importlib_metadata


_already_initialized = False
logger = logging.getLogger(__name__)

# Entry point group advertising context plugin providers.  A provider is a
# module (or callable) that registers typing/target context plugins on import.
_CONTEXT_PLUGINS_GROUP = "numba_context_plugins"


def _iter_entry_points(group, name=None):
    """Iterate the entry points in *group*, optionally filtered by *name*.

    Split, Python 3.10+ and importlib_metadata 3.6+ have the "selectable"
    interface, versions prior to that do not. See "compatibility note" in:
    https://docs.python.org/3.10/library/importlib.metadata.html#entry-points
    """
    eps = importlib_metadata.entry_points()
    if hasattr(eps, 'select'):
        if name is None:
            return list(eps.select(group=group))
        return list(eps.select(group=group, name=name))
    return [ep for ep in eps.get(group, ()) if name is None or ep.name == name]


def load_context_plugins():
    """Load the context plugin providers advertised by entry points.

    Providers register their typing and target context plugins on import.
    Numba advertises its own NumPy registration module in the
    ``numba_context_plugins`` group.  Returns ``True`` if at least one
    provider was discovered, so callers can fall back to the in-tree provider
    when the installed distribution metadata predates the entry point (for
    example, a source checkout that has not been reinstalled).
    """
    discovered = False
    for entry_point in _iter_entry_points(group=_CONTEXT_PLUGINS_GROUP):
        discovered = True
        logger.debug('Loading context plugin: %s', entry_point)
        try:
            entry_point.load()
        except Exception as e:
            msg = (f"Numba context plugin '{entry_point.module}' "
                   f"failed to load due to '{type(e).__name__}({str(e)})'.")
            warnings.warn(msg, stacklevel=3)
            logger.debug('Context plugin loading failed for: %s', entry_point)
    return discovered


def init_all():
    """Execute all `numba_extensions` entry points with the name `init`

    If extensions have already been initialized, this function does nothing.
    """
    global _already_initialized
    if _already_initialized:
        return

    # Must put this here to avoid extensions re-triggering initialization
    _already_initialized = True

    def load_ep(entry_point):
        """Loads a given entry point. Warns and logs on failure.
        """
        logger.debug('Loading extension: %s', entry_point)
        try:
            func = entry_point.load()
            func()
        except Exception as e:
            msg = (f"Numba extension module '{entry_point.module}' "
                   f"failed to load due to '{type(e).__name__}({str(e)})'.")
            warnings.warn(msg, stacklevel=3)
            logger.debug('Extension loading failed for: %s', entry_point)

    for entry_point in _iter_entry_points("numba_extensions", "init"):
        load_ep(entry_point)

    # Numba's own context plugin providers (such as the NumPy registrations)
    # are discovered and imported through the same entry point machinery.
    load_context_plugins()
