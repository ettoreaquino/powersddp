from typing import Any


def iteration(stage: int, discretization: int, scenario: int):
    print("===================================")
    print(
        "STAGE: {} | DISC.: {}% | SCENARIO: {}".format(stage, discretization, scenario)
    )


def _add_row(name: str, title: str, value: float):
    fmt = "{} | {:>15s}: {:>7.2f} hm3"
    print(fmt.format(name, title, value))


def spd_result(
    total_cost: float,
    future_cost: float,
    hydro_units: list,
    thermal_units: list,
    final_volume: Any,
    turbined_volume: Any,
    shedded_volume: Any,
    constraints: Any,
    power_generated: Any,
    shortage: Any,
):
    print("===================================")
    print("{:>21}: ${:>.2f}".format("Total Cost", total_cost))
    print("{:>21}: ${:.2f}".format("Future Cost", future_cost))
    print("===================================")
    for i, hgu in enumerate(hydro_units):
        _add_row(
            name=hgu["name"], title="Final Volume", value=final_volume[i].value()[0]
        )
        _add_row(
            name=hgu["name"],
            title="Turbined Volume",
            value=turbined_volume[i].value()[0],
        )
        _add_row(
            name=hgu["name"], title="Shedded Volume", value=shedded_volume[i].value()[0]
        )
        _add_row(
            name=hgu["name"],
            title="Water Cost",
            value=constraints[i].multiplier.value[0],
        )

    print("-----------------------------------")
    for i, tgu in enumerate(thermal_units):
        _add_row(
            name=tgu["name"],
            title="Power Generated",
            value=power_generated[i].value()[0],
        )

    print("===================================")
    print("{:>21}: {:.2f} MWmed".format("Power Shortage", shortage[0].value()[0]))
    print(
        "{:>21}: {:.2f}".format(
            "Marginal Cost", constraints[len(hydro_units)].multiplier.value[0]
        )
    )
    print("===================================\n")


def ulp_result(
    stages: int,
    scenario: int,
    total_cost: float,
    hydro_units: list,
    thermal_units: list,
    final_volume: Any,
    turbined_volume: Any,
    shedded_volume: Any,
    constraints: Any,
    power_generated: Any,
    shortage: Any,
):
    print("============ SCENARIO {} ===========".format(scenario))
    print("{:>21}: ${:>.2f}".format("Total Cost", total_cost))
    print("===================================")
    for stage in range(stages):
        print("-------------- STAGE {} ------------".format(stage + 1))
        for i, hgu in enumerate(hydro_units):
            _add_row(
                name=hgu["name"],
                title="Final Volume",
                value=final_volume[i][stage].value()[0],
            )
            _add_row(
                name=hgu["name"],
                title="Turbined Volume",
                value=turbined_volume[i][stage].value()[0],
            )
            _add_row(
                name=hgu["name"],
                title="Shedded Volume",
                value=shedded_volume[i][stage].value()[0],
            )
            _add_row(
                name=hgu["name"],
                title="Water Cost",
                value=constraints[i].multiplier.value[0],
            )

        print("-----------------------------------")
        for i, tgu in enumerate(thermal_units):
            _add_row(
                name=tgu["name"],
                title="Power Generated",
                value=power_generated[i][stage].value()[0],
            )
    print("===================================")
    print("{:>21}: {:.2f} MWmed".format("Power Shortage", shortage[0].value()[0]))
    print(
        "{:>21}: {:.2f}".format(
            "Marginal Cost", constraints[len(hydro_units)].multiplier.value[0]
        )
    )
    print("===================================\n")
