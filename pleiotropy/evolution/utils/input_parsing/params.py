import json
from typing import Dict


def get_params(path: str) -> Dict:
    """
    Loads the given experiment parameter file.
    """
    with open(path, 'r') as json_file:
        return json.load(json_file)
