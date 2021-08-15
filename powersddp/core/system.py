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
        Dictionary containing all of the power system parameters, including the generation units.

    """

    def __init__(self, path: str = None, data: dict = None):
        """ __init__ method.

        Parameters
        ----------
        path : str, optional
            Path to the systems.yml file.
        data : :obj:`dict`, optional
            Description of `param2`. Multiple
            lines are supported.
        param3 : :obj:`int`, optional
            Dictionary containing all of the power system parameters, including the generation units.

        """

        self.load_system(path=path, data=data)

    def load_system(self, *, path: str = None, data: dict = None):
        """Loads a Power System from file or dictionary payload

        A Power System data be loaded from both file or dictionary payload. In case both
        positional parameters are suplied the file path will have priority and data payload
        will be ignored.

        PowerSystem loads the data by default during initialization, but can be reloaded ad-hoc.

        Parameters
        ----------
        path : str, optional
            Path to the .yml file containing the system data.
        data : dict, optional
            Dictionary containing the structured data of the system.

        """
        if path:
            with open(path, "r") as f:
                self.data = yaml.load(f, YmlLoader)
        elif data:
            self.data = data
        else:
            raise ValueError(
                "load_system() should receive path=str or data=dict as arguments"
            )
