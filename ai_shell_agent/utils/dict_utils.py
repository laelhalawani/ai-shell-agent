# ai_shell_agent/utils/dict_utils.py
"""
Dictionary utility functions.
"""
import collections.abc
from typing import Dict, Any

def deep_merge_dicts(base: Dict[Any, Any], merge: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Recursively merges two dictionaries.

    Values in `merge` take precedence over values in `base`.
    If a key exists in both dictionaries and both values are dictionaries,
    it recursively merges them. Otherwise, the value from `merge` is used.

    Args:
        base: The base dictionary.
        merge: The dictionary to merge into the base.

    Returns:
        A new dictionary representing the merged result. Does not modify inputs.
    """
    merged = base.copy()  # Start with a copy of the base dictionary

    for key, value in merge.items():
        if key in merged:
            if isinstance(merged[key], dict) and isinstance(value, collections.abc.Mapping):
                # If both are dictionaries, recursively merge
                merged[key] = deep_merge_dicts(merged[key], value)
            else:
                # Otherwise, merge value overrides base value
                merged[key] = value
        else:
            # If key is not in base, just add it
            merged[key] = value

    return merged