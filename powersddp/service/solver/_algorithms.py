import cvxopt.modeling as model

from powersddp.service.solver.api import (logger_service)

# Stochastic Dual Programming
def sdp(
    system_data: dict,
    v_i: list,
    inflow: list,
    cuts: list,
    stage: int,
    verbose: bool = False
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
        if cut["stage"] == stage - 1:
            equation = 0
            for hgu in range(n_hgu):
                equation += float(cut["coefs"][hgu]) * v_f[hgu]
            equation += float(cut["coef_b"])  # type: ignore
            constraints.append(alpha[0] >= equation)

    ## Solving
    opt_problem = model.op(objective=objective_function, constraints=constraints)
    opt_problem.solve(format="dense", solver="glpk")

    if verbose:
        logger_service.spd_result(total_cost=round(objective_function.value()[0], 2),
                                  future_cost=round(alpha[0].value()[0], 2),
                                  hydro_units=system_data["hydro_units"],
                                  thermal_units=system_data["thermal_units"],
                                  final_volume=v_f,
                                  turbined_volume=v_t,
                                  shedded_volume=v_v,
                                  constraints=constraints,
                                  power_generated=g_t,
                                  shortage=shortage)

    return {
        "stage": stage,
        "total_cost": round(float(objective_function.value()[0]), 2),  # type: ignore
        "future_cost": round(float(alpha[0].value()[0]), 2),
        "operational_marginal_cost": round(
            float(constraints[n_hgu].multiplier.value[0]), 2
        ),
        "shortage": round(float(shortage[0].value()[0]), 2),
        "hydro_units": [
            {
                "name": system_data["hydro_units"][i]["name"],
                "stage": stage,
                "v_i": round(float(v_i[i]), 2),
                "inflow": round(float(inflow[i]), 2),
                "v_f": round(float(v_f[i].value()[0]), 2),
                "v_t": round(float(v_t[i].value()[0]), 2),
                "v_v": round(float(v_v[i].value()[0]), 2),
                "water_marginal_cost": -round(
                    float(constraints[i].multiplier.value[0]), 2
                ),
            }
            for i in range(n_hgu)
        ],
        "thermal_units": [
            {"g_t": round(float(g_t[i].value()[0]), 2)} for i in range(n_tgu)
        ],
    }
