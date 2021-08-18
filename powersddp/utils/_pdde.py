"""Utilitarian module to solve power systems.
"""


import cvxopt.modeling as model
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cvxopt import solvers
from plotly.subplots import make_subplots


def solve(system_data: dict, v_i: list, inflow: list, verbose: bool = False):
    """Dual Dynamic Stochastic Programming Solver

    Method to abstract the Dual Stochastic Programming solver applied to the power system
    problem.

    Parameters
    ----------
    system_data : dict,
        Dict containing data structured as used to instantiate a PowerSystem.
    v_i : list
        List containing the initial volume of the Hydro Units, for each
    v_i : list
        List containing the inflow to the Hydro Units
    verbose : bool, optional
        Dictionary containing the structured data of the system.
    """

    if not verbose:
        solvers.options["glpk"] = dict(msg_lev="GLP_MSG_OFF")

    n_tgu = len(system_data["thermal-units"])
    n_hgu = len(system_data["hydro-units"])

    ## Initializing Model Variables
    v_f = model.variable(n_hgu, "Final Volume")
    v_t = model.variable(n_hgu, "Turbined Flow")
    v_v = model.variable(n_hgu, "Shed Flow")
    g_t = model.variable(n_tgu, "Power Generated")
    shortage = model.variable(1, "Power Shortage")

    ## Objective Function
    objective_function = 0
    for i, tgu in enumerate(system_data["thermal-units"]):
        objective_function += tgu["cost"] * g_t[i]
    objective_function += system_data["outage_cost"] * shortage[0]
    for i, _ in enumerate(system_data["hydro-units"]):
        objective_function += 0.01 * v_v[i]

    ## Constraints
    ### Hydro Balance
    constraints = []
    for i, hgu in enumerate(system_data["hydro-units"]):
        constraints.append(v_f[i] == float(v_i[i]) + float(inflow[i]) - v_t[i] - v_v[i])

    ### Load Supply
    supplying = 0
    for i, hgu in enumerate(system_data["hydro-units"]):
        supplying += hgu["prod"] * v_t[i]

    for i, tgu in enumerate(system_data["thermal-units"]):
        supplying += g_t[i]

    supplying += shortage[0]

    constraints.append(supplying == system_data["load"][2])

    ### Bounds
    for i, hgu in enumerate(system_data["hydro-units"]):
        constraints.append(v_f[i] >= hgu["v_min"])
        constraints.append(v_f[i] <= hgu["v_max"])
        constraints.append(v_t[i] >= 0)
        constraints.append(v_t[i] <= hgu["flow_max"])
        constraints.append(v_v[i] >= 0)

    for i, tgu in enumerate(system_data["thermal-units"]):
        constraints.append(g_t[i] >= 0)
        constraints.append(g_t[i] <= tgu["capacity"])

    constraints.append(shortage[0] >= 0)

    ## Solving
    opt_problem = model.op(objective=objective_function, constraints=constraints)
    opt_problem.solve(format="dense", solver="glpk")

    ## Print
    if verbose:
        print("--------------------------------------")
        print("Total Cost: ${}".format(round(objective_function.value()[0], 2)))  # type: ignore
        print("--------------------------------------")
        for i, hgu in enumerate(system_data["hydro-units"]):
            print(
                "HGU {} | {:>15s}: {:>7.2f} hm3".format(i, v_f.name, v_f[i].value()[0])
            )
            print(
                "HGU {} | {:>15s}: {:>7.2f} hm3".format(i, v_t.name, v_t[i].value()[0])
            )
            print(
                "HGU {} | {:>15s}: {:>7.2f} hm3".format(i, v_v.name, v_v[i].value()[0])
            )
            print(
                "HGU {} | {:>15s}: {:>7.2f} $/hm3".format(
                    i, "Water Cost", constraints[i].multiplier.value[0]
                )
            )
            print("--------------------------------------")

        for i, tgu in enumerate(system_data["thermal-units"]):
            print("TGU {} | {}: {:>7.2f} MWmed".format(i, g_t.name, g_t[i].value()[0]))
            print("--------------------------------------")

        print(
            """{}: {:.2f} MWmed\nMarginal Cost: {:.2f}\n======================================\n
        """.format(
                shortage.name,
                shortage[0].value()[0],
                constraints[n_hgu].multiplier.value[0],
            )
        )

    return {
        "shortage": shortage[0].value()[0],
        "operational_marginal_cost": constraints[n_hgu].multiplier.value[0],
        "total_cost": objective_function.value()[0],  # type: ignore
        "hydro_units": [
            {
                "v_f": v_f[i].value()[0],
                "v_t": v_t[i].value()[0],
                "v_v": v_v[i].value()[0],
                "water_marginal_cost": constraints[i].multiplier.value[0],
            }
            for i in range(n_hgu)
        ],
        "thermal_units": [{"g_t": g_t[i].value()[0]} for i in range(n_tgu)],
    }


def plot_future_cost_function(operation: pd.DataFrame):

    n_stages = len(operation["stage"].unique())

    fig = make_subplots(rows=n_stages, cols=1)

    for i,stage in enumerate(operation["stage"].unique()):
        stage_df = operation.loc[operation["stage"] == stage]
        fig.add_trace(
            go.Scatter(
                x=stage_df["initial_volume"],
                y=stage_df["average_cost"],
                mode="lines",
                name="Stage {}".format(stage),
            ),
            row=i+1,
            col=1,
        )

    fig.update_xaxes(title_text="Final Volume [hm3]")
    fig.update_yaxes(title_text="$/MW")

    fig.update_layout(height=300 * n_stages, title_text="Future Cost Function")
    fig.show()
