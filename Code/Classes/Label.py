import numpy as np
from pathlib import Path
from yaml import safe_load


class Label:

    """Load labels"""

    def __init__(self, path):

        """Constructor for Label class"""

        self.path = path
        self.label = self.read()

    def read(self):

        """Read labels"""

        # Retrieve path
        path = Path(self.path)

        # Retrieve extension
        extension = path.suffix

        # Read label
        with open(path, "r") as file:

            if extension == ".yaml":
                label = safe_load(file)

            if extension == ".txt":
                label = file.readline()

        return label
