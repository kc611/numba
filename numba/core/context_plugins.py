"""
A plugin is a callable taking the context being initialized::

    @typing_plugins.register('mypackage')
    def register_mypackage_typing(context):
        from mypackage import decls
        context.install_registry(decls.registry)
"""


class _ContextPluginRegistry(object):
    """Ordered collection of context initializer callables."""

    def __init__(self, kind):
        self._kind = kind
        self._plugins = {}

    def register(self, name):
        """Decorator registering *func* under *name*."""
        def decorator(func):
            if name in self._plugins:
                raise ValueError(
                    "a %s context plugin named %r is already registered"
                    % (self._kind, name)
                )
            self._plugins[name] = func
            return func
        return decorator

    def run(self, context):
        """Invoke every registered plugin against *context*."""
        for func in self._plugins.values():
            func(context)


typing_plugins = _ContextPluginRegistry('typing')
target_plugins = _ContextPluginRegistry('target')
