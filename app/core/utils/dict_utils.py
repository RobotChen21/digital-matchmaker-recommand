# -*- coding: utf-8 -*-
from typing import Dict, Any

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary for MongoDB dot notation updates.
    Example: {'a': {'b': 1}} -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive deep merge of two dictionaries.
    Modifies the target dictionary in place.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            deep_merge(target[key], value)
        else:
            target[key] = value
    return target

def smart_merge(target: dict, source: dict):
    """
    Smart merge for profile updates.
    - Dicts: Recursive merge.
    - Lists: Append and deduplicate (Union).
    - Scalars: Overwrite with new value.
    """
    for key, value in source.items():
        if key in target:
            if isinstance(target[key], dict) and isinstance(value, dict):
                smart_merge(target[key], value)
            elif isinstance(value, list):
                # Robust List Merge:
                # If source is list, we MUST treat target as list.
                
                # 1. Ensure target[key] is a list
                if not isinstance(target[key], list):
                    # If it's a scalar (e.g. legacy dirty data), wrap it
                    if target[key] is not None:
                        target[key] = [target[key]]
                    else:
                        target[key] = []
                
                # 2. Append & Deduplicate
                try:
                    # Attempt set-based deduplication for hashable items
                    current_set = set(target[key])
                    for item in value:
                        if item not in current_set:
                            target[key].append(item)
                            current_set.add(item)
                except TypeError:
                    # Fallback for unhashable items
                    for item in value:
                        if item not in target[key]:
                            target[key].append(item)
            else:
                # Scalar overwrite (assume new info is correction)
                target[key] = value
        else:
            # New key
            target[key] = value