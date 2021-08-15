"""Module to handle classes and methods related to a selected Power System.
This module should follow a systems.json file standar:
{
  "{name}": {
    "shedding_cost": float,
    "load": [float, float, float],
    "n_disc": int,
    "n_est": int,
    "n_cen": int,
    "generation_units": [
      {"type": "hydro",
       "name": "str",
       "v_max": float,
       "v_min": float,
       "prod": float,
       "flow_max": float,
       "inflow_scenarios":[<list>]},
      {"type": "thermal", "name": "str", "capacity": "float", "cost": float},
      ...
    ]
  }
}
Where {name} should be changed to whatever name you may choose to your system.
For example, 'Test01'. Check README.md file.
"""

from abc import ABC, abstractclassmethod
import yaml

from powersddp.util._yml import YmlLoader

YmlLoader.add_constructor("!include", YmlLoader.include)


class PowerSystemInterface(ABC):
    @abstractclassmethod
    def load_system(self):
        raise NotImplementedError


class PowerSystem(PowerSystemInterface):
    def __init__(self, verbose: bool = False, **kwargs):
        self.__verbose = verbose
        self.__dict__.update(kwargs)
        self.load_system()

    def load_system(self):
        if "path" in self.__dict__:
            with open(self.path, "r") as f:
                data = yaml.load(f, YmlLoader)

                self.data = data
                if self.__verbose:
                    print("System loaded from {} file".format(self.path))
        elif "data" in self.__dict__:
            if self.__verbose:
                print("System loaded from 'data' payload")
        else:
            raise NotImplementedError
