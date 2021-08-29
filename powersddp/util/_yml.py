"""Utilitarian module to deal with .yml files 
"""

import yaml
import os


class YmlLoader(yaml.SafeLoader):
    """Class to extend yaml loader capabilities

    Attributes
    ----------
    """

    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(YmlLoader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))  # type: ignore

        with open(filename, "r") as f:
            return yaml.load(f, YmlLoader)


YmlLoader.add_constructor("!include", YmlLoader.include)
