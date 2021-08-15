"""Module to handle classes and methods related to a selected Power System.
This module should follow a systems.yml file standard:

# system.yml
load: [float,float,float]
discretizations: int
stages: int
scenarios: int
outage_cost: float
hydro-units: !include system-hydro.yml
thermal-units: !include system-thermal.yml

# system-hydro.yml
-
  name: str
  v_max: float
  v_min: float
  prod: float
  flow_max: float
  inflow_scenarios:
    - list<float>
    - list<float>
    - list<float>

# system-thermal.yml
-
  name: str
  capacity: float
  cost: float
"""

from abc import ABC, abstractclassmethod
import yaml

from powersddp.util._yml import YmlLoader

YmlLoader.add_constructor("!include", YmlLoader.include)


class PowerSystemInterface(ABC):
    # Abstract Class to Power System.

    @abstractclassmethod
    def load_system(self):
        raise NotImplementedError


class PowerSystem(PowerSystemInterface):
    """Singleton Class to instantiate a Power System.

    A Power System is defined based on set of parameters, including the systems parameters
    and all the generation units. Both thermal and hydro.

    Note: Either initialize a Power System by providing the path to a systems.yml file or
    by providing a dictionary containing all the necessary data.

    Attributes
    ----------
    path : str, optional
        Path to the systems.yml file
    data : dict, optional
        Dictionary containing all of the .

    """

    def __init__(self, path: str = None, data: dict = None):
        self.data = self.load_system(path=path, data=data)

    def load_system(self, *, path: str = None, data: dict = None):
        if path:
            with open(path, "r") as f:
                return yaml.load(f, YmlLoader)
        elif data:
            return data
        else:
            raise ValueError(
                "load_system() should receive path=str or data=dict as arguments"
            )
