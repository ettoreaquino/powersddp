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

from abc import ABC, abstractmethod
from itertools import product
import numpy as np
import pandas as pd
import yaml

from powersddp.utils._yml import YmlLoader
from powersddp.utils._solver import sdp, plot_future_cost_function


class PowerSystemInterface(ABC):
    @abstractmethod
    def load_system(self):
        raise NotImplementedError

    @abstractmethod
    def dispatch(self):
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
        """__init__ method.

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

    def dispatch(
        self, *, solver: str = "sdp", plot: bool = False, verbose: bool = False
    ) -> pd.DataFrame:
        """Solves a financial dispatch of a Power System class

        Once instantiated a Power System can deploy the generation units based on the
        minimization of an objective function. This method iterates over every stage
        and scenario of the Power System, finding the optimal solution of the problem
        using the Dual Stochastic Dynamic Programming technique.

        Parameters
        ----------
        plot : bool, optional
            Boolean to plot the future cost function of every stage.
        verbose : bool, optional
            Dictionary containing the structured data of the system.

        Returns
        -------
        operation : pandas.DataFrame
            A Dataframe containing the operation on every stage and scenario.

        """

        n_hgu = len(self.data["hydro-units"])

        step = 100 / (self.data["discretizations"] - 1)
        discretizations = list(product(np.arange(0, 100 + step, step), repeat=n_hgu))

        operation = []
        cuts = []  # type: ignore
        for stage in range(self.data["stages"], 0, -1):
            for discretization in discretizations:

                v_i = []
                # For Every Hydro Unit
                for i, hgu in enumerate(self.data["hydro-units"]):
                    v_i.append(
                        hgu["v_min"]
                        + (hgu["v_max"] - hgu["v_min"]) * discretization[i] / 100
                    )

                # For Every Scenario
                average = 0.0
                avg_water_marginal_cost = [0 for _ in self.data["hydro-units"]]
                for scenario in range(self.data["scenarios"]):
                    inflow = []
                    for i, hgu in enumerate(self.data["hydro-units"]):
                        inflow.append(hgu["inflow_scenarios"][stage - 1][scenario])

                    if verbose:
                        print(
                            "STAGE: {} | DISC.: {}% | SCENARIO: {}".format(
                                stage, int(discretization[0]), scenario
                            )
                        )
                    result = sdp(
                        system_data=self.data,
                        v_i=v_i,
                        inflow=inflow,
                        cuts=cuts,
                        stage=stage + 1,
                        verbose=verbose,
                    )
                    average += result["total_cost"]
                    for i, hgu in enumerate(result["hydro_units"]):
                        avg_water_marginal_cost[i] += hgu["water_marginal_cost"]

                # Calculating the average of the scenarios
                average = average / self.data["scenarios"]
                coef_b = average
                for i, hgu in enumerate(result["hydro_units"]):
                    # ! Invert the coefficient because of the minimization problem inverts the signal
                    avg_water_marginal_cost[i] = (
                        -avg_water_marginal_cost[i] / self.data["scenarios"]
                    )
                    coef_b -= v_i[i] * avg_water_marginal_cost[i]

                cuts.append(
                    {"stage": stage, "coef_b": coef_b, "coefs": avg_water_marginal_cost}
                )
                operation.append(
                    {
                        "stage": stage,
                        "storage_percentage": "{}%".format(int(discretization[i])),
                        "initial_volume": v_i[0],
                        "average_cost": round(average, 2),
                    }
                )
        operation_df = pd.DataFrame(operation)

        if n_hgu == 1 and plot:
            plot_future_cost_function(operation=operation_df)

        return operation_df
