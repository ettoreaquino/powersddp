"""Utilitarian module to solve power systems.
"""


import cvxopt.modeling as model
import pandas as pd
import plotly.graph_objects as go

from cvxopt import solvers
from plotly.subplots import make_subplots

solvers.options["glpk"] = dict(msg_lev="GLP_MSG_OFF")


# Unique Linear Programming
def ulp(
    system_data: dict,
    scenario: int = 0,
    verbose: bool = False,
):
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
        print("============ SCENARIO {} =============".format(scenario + 1))
        print("Total Cost (All stages): ${}".format(round(objective_function.value()[0], 2)))  # type: ignore
        print("======================================\n")
        for stage in range(system_data["stages"]):
            print("============== STAGE {} ==============".format(stage + 1))
            for i, hgu in enumerate(system_data["hydro_units"]):
                print(
                    "{} | {:>15s}: {:>7.2f} hm3".format(
                        hgu["name"], v_f[i].name, v_f[i][stage].value()[0]
                    )
                )
                print(
                    "{} | {:>15s}: {:>7.2f} hm3".format(
                        hgu["name"], v_t[i].name, v_t[i][stage].value()[0]
                    )
                )
                print(
                    "{} | {:>15s}: {:>7.2f} hm3".format(
                        hgu["name"], v_v[i].name, v_v[i][stage].value()[0]
                    )
                )
                print(
                    "{} | {:>15s}: {:>7.2f} $/hm3".format(
                        hgu["name"], "Water Cost", constraints[i].multiplier.value[0]
                    )
                )

            for i, tgu in enumerate(system_data["thermal_units"]):
                print("--------------------------------------")
                print(
                    "{} | {}: {:>7.2f} MWmed".format(
                        tgu["name"], g_t[i].name, g_t[i][stage].value()[0]
                    )
                )
            print("======================================\n")
        print(
            """======================================\n{}: {:.2f} MWmed\nMarginal Cost: {:.2f}\n======================================\n
        """.format(
                shortage.name,
                shortage[0].value()[0],
                constraints[n_hgu].multiplier.value[0],
            )
        )

    hgu_results, tgu_results = [], []
    for stage in range(system_data["stages"]):
        for i in range(n_hgu):
            hgu_results.append(
                {
                    "stage": stage + 1,
                    "name": system_data["hydro_units"][i]["name"],
                    "vf": round(v_f[i][stage].value()[0], 3),
                    "vt": round(v_t[i][stage].value()[0], 3),
                    "vv": round(v_v[i][stage].value()[0], 3),
                    "wmc": round(constraints[i].multiplier.value[0], 3),
                }
            )
        for i in range(n_tgu):
            tgu_results.append(
                {
                    "stage": stage + 1,
                    "name": system_data["thermal_units"][i]["name"],
                    "gt": round(g_t[i][stage].value()[0], 3),
                }
            )

    return {
        "total_cost": objective_function.value()[0],  # type: ignore
        "operational_marginal_cost": constraints[n_hgu].multiplier.value[0],
        "shortage": shortage[0].value()[0],
        "hydro_units": pd.DataFrame(hgu_results),
        "thermal_units": pd.DataFrame(tgu_results),
    }


# Stochastic Dual Programming
def sdp(
    system_data: dict,
    v_i: list,
    inflow: list,
    cuts: list,
    stage: int,
    verbose: bool = False,
):
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
        constraints.append(v_f[i] == float(v_i[i]) + float(inflow[i]) - v_t[i] - v_v[i])

    ### Load Supply
    supplying = 0
    for i, hgu in enumerate(system_data["hydro_units"]):
        supplying += hgu["prod"] * v_t[i]

    for i, tgu in enumerate(system_data["thermal_units"]):
        supplying += g_t[i]

    supplying += shortage[0]

    constraints.append(supplying == system_data["load"][stage - 2])

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
        if cut["stage"] == stage:
            equation = 0
            for hgu in range(n_hgu):
                equation += float(cut["coefs"][hgu]) * v_f[hgu]
            equation += float(cut["coef_b"])  # type: ignore
            constraints.append(alpha[0] >= equation)

    ## Solving
    opt_problem = model.op(objective=objective_function, constraints=constraints)
    opt_problem.solve(format="dense", solver="glpk")

    ## Print
    if verbose:
        print("======================================")
        print("Total Cost: ${}".format(round(objective_function.value()[0], 2)))  # type: ignore
        print("Future Cost: ${}".format(round(alpha[0].value()[0], 2)))
        print("======================================")
        for i, hgu in enumerate(system_data["hydro_units"]):
            print(
                "{} | {:>15s}: {:>7.2f} hm3".format(
                    hgu["name"], v_f.name, v_f[i].value()[0]
                )
            )
            print(
                "{} | {:>15s}: {:>7.2f} hm3".format(
                    hgu["name"], v_t.name, v_t[i].value()[0]
                )
            )
            print(
                "{} | {:>15s}: {:>7.2f} hm3".format(
                    hgu["name"], v_v.name, v_v[i].value()[0]
                )
            )
            print(
                "{} | {:>15s}: {:>7.2f} $/hm3".format(
                    hgu["name"], "Water Cost", constraints[i].multiplier.value[0]
                )
            )

        for i, tgu in enumerate(system_data["thermal_units"]):
            print("--------------------------------------")
            print(
                "{} | {}: {:>7.2f} MWmed".format(
                    tgu["name"], g_t.name, g_t[i].value()[0]
                )
            )
        print("======================================")
        print(
            """{}: {:.2f} MWmed\nMarginal Cost: {:.2f}\n======================================\n
        """.format(
                shortage.name,
                shortage[0].value()[0],
                constraints[n_hgu].multiplier.value[0],
            )
        )

    return {
        "total_cost": objective_function.value()[0],  # type: ignore
        "future_cost": alpha[0].value()[0],
        "operational_marginal_cost": constraints[n_hgu].multiplier.value[0],
        "shortage": shortage[0].value()[0],
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

    n_stages = operation["stage"].unique().size

    fig = make_subplots(rows=n_stages, cols=1)

    for i, stage in enumerate(operation["stage"].unique()):
        stage_df = operation.loc[operation["stage"] == stage]
        fig.add_trace(
            go.Scatter(
                x=stage_df["initial_volume"],
                y=stage_df["average_cost"],
                mode="lines",
                name="Stage {}".format(stage),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_xaxes(title_text="Final Volume [hm3]")
    fig.update_yaxes(title_text="$/MW")

    fig.update_layout(height=300 * n_stages, title_text="Future Cost Function")
    fig.show()


def plot_future_cost_3D_function(costs: pd.DataFrame):

    n_stages = costs["stage"].unique().size

    fig = make_subplots(
        rows=n_stages,
        cols=1,
        specs=[[{"type": "surface"}]] * n_stages,
        subplot_titles=["Stage {}".format(stage + 1) for stage in range(n_stages)],
    )

    for i, stage in enumerate(costs["stage"].unique()):
        stage_df = costs.loc[costs["stage"] == stage]
        fig.add_trace(
            go.Surface(
                x=stage_df["xaxis"].values[0],
                y=stage_df["yaxis"].values[0],
                z=stage_df["zaxis"].values[0],
                showscale=False,
                colorscale="Viridis",
            ),
            row=i + 1,
            col=1,
        )

    fig.update_layout(
        scene=dict(
            xaxis_title="{} Initial Volume [hm3]".format(costs["HGUs"][0][0]),
            yaxis_title="{} Initial Volume [hm3]".format(costs["HGUs"][0][1]),
            zaxis_title="$/MW",
        ),
        scene2=dict(
            xaxis_title="{} Initial Volume [hm3]".format(costs["HGUs"][0][0]),
            yaxis_title="{} Initial Volume [hm3]".format(costs["HGUs"][0][1]),
            zaxis_title="$/MW",
        ),
        scene3=dict(
            xaxis_title="{} Initial Volume [hm3]".format(costs["HGUs"][0][0]),
            yaxis_title="{} Initial Volume [hm3]".format(costs["HGUs"][0][1]),
            zaxis_title="$/MW",
        ),
        height=900 * n_stages,
        width=1500,
        title_text="Future Cost Function",
        autosize=False,
    )
    fig.show()


def plot_ulp(
    gu_operation: pd.DataFrame, yaxis_column: str, yaxis_title: str, plot_title: str
):

    n_gu = gu_operation["name"].unique().size

    fig = make_subplots(rows=n_gu, cols=1)

    for i, gu in enumerate(gu_operation["name"].unique()):
        gu_df = gu_operation.loc[gu_operation["name"] == gu]
        fig.add_trace(
            go.Scatter(
                x=gu_df["stage"],
                y=gu_df[yaxis_column],
                mode="lines",
                name="{}".format(gu),
            ),
            row=i + 1,
            col=1,
        )

    fig.update_xaxes(title_text="Stages")
    fig.update_yaxes(title_text=yaxis_title)

    fig.update_layout(height=300 * n_gu, title_text=plot_title)
    fig.show()
