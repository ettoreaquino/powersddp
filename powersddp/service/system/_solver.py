"""Module to handle the Strategy Pattern for 

"""


import cvxopt.modeling as model
import numpy as np
import pandas as pd

from cvxopt import solvers
from itertools import product

import powersddp.service.system._verbose as verbose_service

solvers.options["glpk"] = dict(msg_lev="GLP_MSG_OFF")

# General Linear Programming
def _glp(
    system_data: dict,
    initial_volume: list,
    inflow: list,
    cuts: list,
    stage: int,
    verbose: bool = False,
):

    n_hgu = len(system_data["hydro_units"])
    n_tgu = len(system_data["thermal_units"])

    ## Initializing Model Variables
    v_f = model.variable(n_hgu, "Final Volume")
    v_t = model.variable(n_hgu, "Turbined Flow")
    v_v = model.variable(n_hgu, "Shed Flow")
    g_t = model.variable(n_tgu, "Power Generated")
    shortage = model.variable(1, "Power Shortage")
    alpha = model.variable(1, "Future Cost")

    ## Objective Function
    objective_function = 0
    for i, tgu in enumerate(system_data["thermal_units"]):
        objective_function += tgu["cost"] * g_t[i]
    objective_function += system_data["outage_cost"] * shortage[0]
    for i, _ in enumerate(system_data["hydro_units"]):
        objective_function += 0.01 * v_v[i]
    objective_function += 1.0 * alpha[0]

    ## Constraints
    ### Hydro Balance
    constraints = []
    for i, hgu in enumerate(system_data["hydro_units"]):
        constraints.append(
            v_f[i] == float(initial_volume[i]) + float(inflow[i]) - v_t[i] - v_v[i]
        )

    ### Load Supply
    supplying = 0
    for i, hgu in enumerate(system_data["hydro_units"]):
        supplying += hgu["prod"] * v_t[i]

    for i, tgu in enumerate(system_data["thermal_units"]):
        supplying += g_t[i]

    supplying += shortage[0]

    constraints.append(supplying == system_data["load"][stage - 1])

    ### Bounds
    for i, hgu in enumerate(system_data["hydro_units"]):
        constraints.append(v_f[i] >= hgu["v_min"])
        constraints.append(v_f[i] <= hgu["v_max"])
        constraints.append(v_t[i] >= 0)
        constraints.append(v_t[i] <= hgu["flow_max"])
        constraints.append(v_v[i] >= 0)

    for i, tgu in enumerate(system_data["thermal_units"]):
        constraints.append(g_t[i] >= 0)
        constraints.append(g_t[i] <= tgu["capacity"])

    constraints.append(shortage[0] >= 0)
    constraints.append(alpha[0] >= 0)

    ### Cut constraint (Future cost function of forward stage)
    for cut in cuts:
        if cut["stage"] == stage + 1:
            equation = 0
            for hgu in range(n_hgu):
                equation += float(cut["coefs"][hgu]) * v_f[hgu]
            equation += float(cut["coef_b"])  # type: ignore
            constraints.append(alpha[0] >= equation)

    ## Solving
    opt_problem = model.op(objective=objective_function, constraints=constraints)
    opt_problem.solve(format="dense", solver="glpk")

    if verbose:
        verbose_service.spd_result(
            total_cost=round(objective_function.value()[0], 2),  # type: ignore
            future_cost=round(alpha[0].value()[0], 2),
            hydro_units=system_data["hydro_units"],
            thermal_units=system_data["thermal_units"],
            final_volume=v_f,
            turbined_volume=v_t,
            shedded_volume=v_v,
            constraints=constraints,
            power_generated=g_t,
            shortage=shortage,
        )

    return {
        "total_cost": round(float(objective_function.value()[0]), 2),  # type: ignore
        "future_cost": round(float(alpha[0].value()[0]), 2),
        "operational_marginal_cost": -round(
            float(constraints[n_hgu].multiplier.value[0]), 2
        ),
        "shortage": round(float(shortage[0].value()[0]), 2),
        "hydro_units": pd.DataFrame(
            [
                {
                    "stage": stage,
                    "name": system_data["hydro_units"][i]["name"],
                    "vi": round(float(initial_volume[i]), 2),
                    "inflow": round(float(inflow[i]), 2),
                    "vf": round(float(v_f[i].value()[0]), 2),
                    "vt": round(float(v_t[i].value()[0]), 2),
                    "vv": round(float(v_v[i].value()[0]), 2),
                    "wmc": round(float(constraints[i].multiplier.value[0]), 2),
                }
                for i in range(n_hgu)
            ]
        ),
        "thermal_units": pd.DataFrame(
            [
                {
                    "stage": stage,
                    "name": system_data["thermal_units"][i]["name"],
                    "g_t": round(float(g_t[i].value()[0]), 2),
                }
                for i in range(n_tgu)
            ]
        ),
    }


# Stochastic Dual Programming
def sdp(system_data: dict, verbose: bool = False):
    """Stochastic Dual Programming Solver

    Method to abstract the Dual Stochastic Programming solver applied to the power system
    problem.

    Parameters
    ----------
    system_data : dict,
        Dict containing data structured as used to instantiate a PowerSystem.
    v_i : list
        List containing the initial volume of the Hydro Units, for each unit.
    inflow : list,
        List containing the inflow to the Hydro Units, for each unit.
    cuts : list,
        List containing the overall result of the stage analyzed.
    stage : int,
        Int-value containing the stage information to be analyzed.
    verbose : bool, optional,
        Dictionary containing the structured data of the system.

    Returns
    -------
    operation : dict
        A dictionary representing the operation
    """

    n_hgu = len(system_data["hydro_units"])

    step = 100 / (system_data["discretizations"] - 1)
    discretizations = list(product(np.arange(0, 100 + step, step), repeat=n_hgu))

    operation = []
    cuts = []  # type: ignore
    for stage in range(system_data["stages"], 0, -1):
        for discretization in discretizations:

            # Initial Volume
            v_i = [
                hgu["v_min"] + (hgu["v_max"] - hgu["v_min"]) * discretization[i] / 100
                for i, hgu in enumerate(system_data["hydro_units"])
            ]

            # For Every Scenario
            for scenario in range(system_data["scenarios"]):
                inflow = [
                    hgu["inflow_scenarios"][stage - 1][scenario]
                    for hgu in system_data["hydro_units"]
                ]

                if verbose:
                    verbose_service.iteration(
                        stage=stage,
                        scenario=scenario,
                    )

                result = _glp(
                    system_data=system_data,
                    initial_volume=v_i,
                    inflow=inflow,
                    cuts=cuts,
                    stage=stage,
                    verbose=verbose,
                )

                result["stage"] = stage
                result["discretization"] = discretization[0] if len(discretization) == 1 else discretization
                result["initial_volume"] = v_i[0] if len(v_i) == 1 else v_i
                result["scenario"] = scenario + 1
                result["hydro_units"]["scenario"] = scenario + 1
                result["hydro_units"]["discretization"] = discretization[0] if len(discretization) == 1 else discretization
                result["thermal_units"]["scenario"] = scenario + 1
                operation.append(result)

            # Adding coef to cuts
            df = pd.DataFrame(operation)
            result_hydro = pd.concat([df for df in df["hydro_units"]])
            result_avg = result_hydro.groupby(["name", "stage"]).mean()
            result_avg["coef_b"] = result_avg["vi"] * result_avg["wmc"]

            cuts.append(
                {
                    "stage": stage,
                    "coef_b": result_avg["coef_b"].sum(),
                    "coefs": result_avg["wmc"].tolist(),
                }
            )

    result_columns = [
        "stage",
        "scenario",
        "discretization",
        "initial_volume",
        "total_cost",
        "future_cost",
        "operational_marginal_cost",
        "shortage",
        "hydro_units",
        "thermal_units",
    ]
    return {"operation_df": pd.DataFrame(operation, columns=result_columns), "cuts": cuts}


def ulp(system_data: dict, scenario: int = 0, verbose: bool = False):
    """Unique Linear Programming Solver

    Parameters
    ----------
    system_data : dict,
        Dict containing data structured as used to instantiate a PowerSystem.
    scenario: int,
        Inflow scenario index.
    verbose : bool, optional
        Dictionary containing the structured data of the system.

    Returns
    -------
    operation : dict
        A dictionary representing the operation
    """

    n_hgu = len(system_data["hydro_units"])
    n_tgu = len(system_data["thermal_units"])
    ## Initializing Model Variables
    v_f = []
    v_t = []
    v_v = []
    g_t = []
    ### Hydro Units variables
    for _, hgu in enumerate(system_data["hydro_units"]):
        v_f.append(model.variable(system_data["stages"], "Final Volume"))
        v_t.append(
            model.variable(
                system_data["stages"],
                "Turbined Flow",
            )
        )
        v_v.append(
            model.variable(
                system_data["stages"],
                "Shed Flow",
            )
        )
    ### Thermal Units variables
    for _, tgu in enumerate(system_data["thermal_units"]):
        g_t.append(
            model.variable(
                system_data["stages"],
                "Power Generated",
            )
        )
    ### Shortage variable
    shortage = model.variable(system_data["stages"], "Power Shortage")
    ## Objective Function
    objective_function = 0
    for stage in range(system_data["stages"]):
        objective_function += system_data["outage_cost"] * shortage[stage]
        for i, tgu in enumerate(system_data["thermal_units"]):
            objective_function += tgu["cost"] * g_t[i][stage]
        for i, _ in enumerate(system_data["hydro_units"]):
            objective_function += 0.01 * v_v[i][stage]

    ## Constraints
    ### Hydro Balance
    constraints = []
    for i, hgu in enumerate(system_data["hydro_units"]):
        for stage in range(system_data["stages"]):
            if stage == 0:
                constraints.append(
                    v_f[i][stage]
                    == float(hgu["v_ini"])
                    + float(hgu["inflow_scenarios"][stage][scenario])
                    - v_t[i][stage]
                    - v_v[i][stage]
                )
            else:
                constraints.append(
                    v_f[i][stage]
                    == v_f[i][stage - 1]
                    + float(hgu["inflow_scenarios"][stage][scenario])
                    - v_t[i][stage]
                    - v_v[i][stage]
                )
    ### Load Supply
    for stage in range(system_data["stages"]):
        load_supply = 0
        for i, hgu in enumerate(system_data["hydro_units"]):
            load_supply += hgu["prod"] * v_t[i][stage]
        for i, _ in enumerate(system_data["thermal_units"]):
            load_supply += g_t[i][stage]
        load_supply += shortage[stage]
        constraints.append(load_supply == system_data["load"][stage])
    ### Bounds
    for stage in range(system_data["stages"]):
        for i, hgu in enumerate(system_data["hydro_units"]):
            constraints.append(v_f[i][stage] >= hgu["v_min"])
            constraints.append(v_f[i][stage] <= hgu["v_max"])
            constraints.append(v_t[i][stage] >= 0)
            constraints.append(v_t[i][stage] <= hgu["flow_max"])
            constraints.append(v_v[i][stage] >= 0)
        for i, tgu in enumerate(system_data["thermal_units"]):
            constraints.append(g_t[i][stage] >= 0)
            constraints.append(g_t[i][stage] <= tgu["capacity"])
        constraints.append(shortage[stage] >= 0)

    ## Solving
    opt_problem = model.op(objective=objective_function, constraints=constraints)
    opt_problem.solve(format="dense", solver="glpk")

    ## Print
    if verbose:
        verbose_service.ulp_result(
            stages=system_data["stages"],
            scenario=scenario + 1,
            total_cost=round(objective_function.value()[0], 2),  # type: ignore
            hydro_units=system_data["hydro_units"],
            thermal_units=system_data["thermal_units"],
            final_volume=v_f,
            turbined_volume=v_t,
            shedded_volume=v_v,
            constraints=constraints,
            power_generated=g_t,
            shortage=shortage,
        )

    hgu_results, tgu_results = [], []
    for stage in range(system_data["stages"]):
        for i in range(n_hgu):
            hgu_results.append(
                {
                    "stage": stage + 1,
                    "name": system_data["hydro_units"][i]["name"],
                    "vf": round(float(v_f[i][stage].value()[0]), 2),
                    "vt": round(float(v_t[i][stage].value()[0]), 2),
                    "vv": round(float(v_v[i][stage].value()[0]), 2),
                    "wmc": round(float(constraints[i].multiplier.value[0]), 2),
                }
            )
        for i in range(n_tgu):
            tgu_results.append(
                {
                    "stage": stage + 1,
                    "name": system_data["thermal_units"][i]["name"],
                    "gt": round(float(g_t[i][stage].value()[0]), 2),
                }
            )

    return {
        "total_cost": round(float(objective_function.value()[0]), 2),  # type: ignore
        "operational_marginal_cost": round(
            float(constraints[n_hgu].multiplier.value[0]), 2
        ),
        "shortage": round(float(shortage[0].value()[0]), 2),
        "hydro_units": pd.DataFrame(hgu_results),
        "thermal_units": pd.DataFrame(tgu_results),
    }
