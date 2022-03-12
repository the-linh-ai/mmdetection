from typing import Dict, Any


class Registry(dict):
    def __init__(self, name: str):
        super(Registry, self).__init__()
        self.name = name

    def __getitem__(self, key):
        try:
            return super(Registry, self).__getitem__(key)
        except KeyError:
            raise KeyError(
                f"'{key}' not in the '{self.name}' registry. "
                f"Available options: {list(self.keys())}."
            )

    def __setitem__(self, key, value):
        if super(Registry, self).__contains__(key):  # avoid duplicates
            raise KeyError(f"Key '{key}' already in the '{self.name}' registry.")
        super(Registry, self).__setitem__(key, value)


registry: Dict[str, Dict[str, Any]] = {}


def register(type: str):
    if type not in registry:
        registry[type] = Registry(name=type)
    def register_fn(cls):
        registry[type][cls.__name__] = cls
        return cls
    return register_fn
