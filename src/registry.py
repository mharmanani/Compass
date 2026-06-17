_REGISTRY = {}


def register(kind: str, name: str):
    def deco(fn):
        _REGISTRY.setdefault(kind, {})
        if name in _REGISTRY[kind]:
            raise ValueError(f"{kind}:{name} already registered")
        _REGISTRY[kind][name] = fn
        return fn

    return deco


def get_entrypoint(kind: str, name: str):
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY.get(kind, {}).keys()))
        raise ValueError(f"Unknown {kind} '{name}'. Known: [{known}]")


def build(kind: str, name: str | None = None, **kwargs):
    fn = get_entrypoint(kind, name)
    return fn(**kwargs)


def list_kinds() -> list[str]:
    return sorted(_REGISTRY.keys())


def list_names(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}).keys())


def get_help(kind: str, name: str):
    help(get_entrypoint(kind, name))
