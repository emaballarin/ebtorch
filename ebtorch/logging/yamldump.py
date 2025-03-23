#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
import os

import yaml


# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["write_dict_as_yaml"]


# ~~ Functions ~~ ──────────────────────────────────────────────────────────────
def write_dict_as_yaml(data: dict, file_path: str) -> bool:
    if not isinstance(data, dict):
        raise TypeError("`data` must be a dictionary.")
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)
        return True
    except (IOError, TypeError, ValueError) as e:
        print(f"Error writing YAML: {e}.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}.")
        return False
