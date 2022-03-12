import os
import json
import yaml

from typing import Any, Tuple, Dict, List

from .registry import registry



"""
====================================== DictConfig ======================================
"""


def update_config(
    config: dict,
    update_dict: Dict[str, Any],
    path: List[str] = None,  # list of keys that leads to the current config
):
    """
    Update config with command line arguments, e.g,, "general.seed=136".
    Remove found keys from `update_dict` in-place.
    """
    if update_dict is None:
        return
    path = ".".join(path) + "."
    found_keys = []

    for key, value in update_dict.items():
        # Skip if invalid
        if not key.startswith(path):
            continue

        keys = key[len(path):].split(".")
        sub_config = config
        recognized = True

        for i in range(len(keys) - 1):
            if (not isinstance(sub_config, dict)) or (keys[i] not in sub_config):
                recognized = False
                break
            sub_config = sub_config[keys[i]]

        if (not isinstance(sub_config, dict)):
            recognized = False
        # Don't update sub-config referring to another file
        # Usage: `training.num_iters=80000`
        elif "_choice_" in sub_config and keys[-1] != "_choice_":
            recognized = False

        if recognized:
            sub_config[keys[-1]] = parse_str(value)
            found_keys.append(key)

    # Remove found configs in-place
    for key in found_keys:
        del update_dict[key]


def convert_to_dict_config(
    root_obj,
    obj,
    config_dir: str,  # used to handle relative paths
    float_handler: bool = True,
    unknown_args: dict = None,
    path = [],  # list of keys that leads to `obj` from `root_obj`
    # Whether this is the outermost stack of reading from a module file;
    # This should only be set internally (see below), NOT externally (i.e.,
    #  when calling this function)
    read_from_module_file: bool = False,
    module_choice: str = None,  # activated once `read_from_module_file` is True
):
    """
    Recursively convert an object to its `DictConfig` counterpart whenever possible.
    """
    # Try to update config at every level
    if not read_from_module_file:
        update_config(obj, unknown_args, path=path)

    if isinstance(obj, dict):
        # Initialize from another config file; following hydra's idea
        if "_choice_" in obj:
            assert ("_file_" in obj and len(obj) == 2) or ("_file_" not in obj and len(obj) == 1), (
                f"'_file_' (optional) and '_choice_' must be the only keys in this "
                f"dict config: {obj}"
            )
            assert obj["_choice_"] is not None, f"'_choice_' must be specified: {obj}"

            # If "_file_" is present, read from this file
            if "_file_" in obj:
                # Read options
                options = DictConfig.from_yaml(
                    os.path.join(config_dir, obj["_file_"]),
                    config_dir=config_dir,
                    float_handler=float_handler,
                    path=path,
                    unknown_args=unknown_args,
                    read_from_module_file=True,
                    module_choice=obj["_choice_"],
                )
                # Retrieve choice
                new_obj = options[obj["_choice_"]]
            # Otherwise, read from the same config
            else:
                new_obj = root_obj[obj["_choice_"]]

        else:
            new_obj = {}
            for key, value in obj.items():
                # When reading from a module file, we dont add current key
                # to the path stack
                if read_from_module_file:
                    # Don't read unnecessary keys, as it will unnecessarily
                    # read and modify `unknown_args`, causing the true config
                    # not updated correctly
                    assert module_choice is not None
                    if key != module_choice:
                        continue
                    new_path = path
                else:
                    new_path = path + [key]

                new_obj[key] = convert_to_dict_config(
                    root_obj=root_obj,
                    obj=value,
                    config_dir=config_dir,
                    float_handler=float_handler,
                    path=new_path,
                    unknown_args=unknown_args,
                )
            new_obj = DictConfig(**new_obj)

    elif isinstance(obj, (list, tuple)):
        new_obj = [
            convert_to_dict_config(
                root_obj=root_obj,
                obj=ob,
                config_dir=config_dir,
                float_handler=float_handler,
                # Disable parameter overriding
                path=None,
                unknown_args=None,
            )
            for ob in obj
        ]

    else:
        # Some simple heuristics to find, e.g., "5e-4"
        exclude = ["min", "max"]
        if float_handler and isinstance(obj, str) and \
                obj not in exclude and "e" in obj:
            try:
                obj = eval(obj)
            except Exception:
                pass

        new_obj = obj

    return new_obj


def convert_to_dict(obj):
    """
    Recursively convert an object to its dictionary counterpart whenever possible.
    """
    if isinstance(obj, (dict, DictConfig)):
        new_obj = {}
        for key, value in obj.items():
            new_obj[key] = convert_to_dict(value)

    elif isinstance(obj, (list, tuple)):
        new_obj = [convert_to_dict(ob) for ob in obj]

    else:
        new_obj = obj

    return new_obj


def is_jsonable(x: Any):
    """
    Check if an object is json serializable.
    Source: https://stackoverflow.com/a/53112659
    """
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def _remove_non_jsonable(obj: Any) -> Tuple[bool, Any]:
    """
    Recursively remove non-json serializable objects from the given object.
    """
    if is_jsonable(obj):
        return True, obj

    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            jsonable, obj = _remove_non_jsonable(value)
            if jsonable:
                new_obj[key] = obj

    elif isinstance(obj, (list, tuple)):
        new_obj = []
        for value in obj:
            jsonable, obj = _remove_non_jsonable(value)
            if jsonable:
                new_obj.append(obj)

    else:
        return False, None

    return True, new_obj


def remove_non_jsonable(obj: Any) -> Any:
    _, obj =  _remove_non_jsonable(obj)
    return obj


class DictConfig(dict):
    @classmethod
    def from_object(
        cls,
        dictionary: Any,
        config_dir: str,
        float_handler: bool = True,
        **kwargs,
    ):
        return convert_to_dict_config(
            root_obj=dictionary,
            obj=dictionary,
            config_dir=config_dir,
            float_handler=float_handler,
            **kwargs,
        )

    def __setattr__(self, name: str, value: Any):
        self[name] = convert_to_dict_config(value, value, config_dir=None)

    def __getattr__(self, name: str):
        if name not in self:
            raise AttributeError
        return self[name]

    def to_dict(self, remove_non_jsonable_objects: bool = False):
        self_dict = convert_to_dict(self)
        if remove_non_jsonable_objects:
            self_dict = remove_non_jsonable(self_dict)
        return self_dict

    @classmethod
    def from_yaml(
        cls,
        file: str,
        config_dir: str,
        float_handler: bool = True,
        **kwargs,
    ):
        with open(file, "r") as conf:
            config_dict = yaml.load(conf, Loader=yaml.FullLoader)
        return cls.from_object(
            config_dict,
            config_dir=config_dir,
            float_handler=float_handler,
            **kwargs,
        )


def parse_str(string: str):
    """Parse using yaml for true, false, etc."""
    try:
        string = yaml.safe_load(string)
    except Exception:
        pass
    return string


def initialize_from_config(cfg: DictConfig, name: str, **kwargs):
    """
    Initialize any object from its config.
    A name is required:
    1. The corresponding registry will be retrieved based on `name`.
    2. The corresponding initializer (usually class) will be retrieved
    from `cfg["{name}_class"]`.
    3. The corresponding keyword arguments (kwargs) will be retrieved
    from `cfg["{name}_init_kwargs"]`.

    Any keyword arguments passed to this function will be passed along
    to the initializer.
    """
    object_init = registry[name][cfg[f"{name}_class"]]
    obj = object_init(
        **kwargs,
        **cfg[f"{name}_init_kwargs"],
    )
    return obj
