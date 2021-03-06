[![PyPI version](https://badge.fury.io/py/powersddp.svg)](https://badge.fury.io/py/powersddp)

[![Python versions](https://img.shields.io/pypi/pyversions/powersddp.svg)](https://pypi.python.org/pypi/powersddp)

[![Downloads](https://pepy.tech/badge/powersddp)](https://pepy.tech/project/powersddp)


# **Power** System **S**tochastic **D**ual **D**ynamic **P**rogramming

The main goal of this library is to provide support for studies regarding the optimal dispatch of power systems, majorly comprised of Thermoelectric and Hydroelectric Generators.

> **Note 1** This is an under development library.

A special thank should be given to professor **André Marcato**. This project does not intend to substitute the similar library `PySDDP`.

> **Note 1** This project is being developed alongside the masters course: _Planejamento de Sistemas Elétricos_, as part of the masters program in Energy Systems at the [_Electrical Engineering Graduate Program_](https://www2.ufjf.br/ppee-en/) from the  _Universidade Federal de Juiz de Fora - Brazil_

> **Note 2** The code will evolve alongside the video lectures provided by professor Marcato at: [Curso de Planejamento de Sistemas Elétricos](https://www.youtube.com/watch?v=a4D_mouXoUw&list=PLz7tpQ4EY_ne0gfWIqw6pJFrCglT6fjq7)

## Installation

```
pip install powersddp
```

## Example

There are two ways of initializing a `Power System`. Either by providing a `.yml` file, or by passing a dictionary as an initialization data. Both are depicted bellow:

> **Note:** When using the file input method (`.yml` format) check the  [example](system.yml) of how to declare the parameters.


### Initializing a `PowerSystem`
```Python
import powersddp as psddp

system = psddp.PowerSystem(path='system.yml')

print("System Load: {}\n"
      "Number of HGUs: {}\n"
      "Number of TGUs: {}".format(system.data['load'],
                                  len(system.data['hydro_units']),
                                  len(system.data['thermal_units'])))
```

```Python
import powersddp as psddp

data = {'load': [50, 50, 50],
        'discretizations': 3,
        'stages': 3,
        'scenarios': 2,
        'outage_cost': 500,
        'hydro_units': [{'name': 'HU1',
                         'v_max': 100,
                         'v_min': 20,
                         'v_ini': 100,
                         'prod': 0.95,
                         'flow_max': 60,
                         'inflow_scenarios': [[23, 16], [19, 14], [15, 11]]}],
        'thermal_units': [{'name': 'GT1', 'capacity': 15, 'cost': 10},
                          {'name': 'GT2', 'capacity': 10, 'cost': 25}]}

PowerSystem = psddp.PowerSystem(data=data)

print("System Load: {}\n"
      "Number of HGUs: {}\n"
      "Number of TGUs: {}".format(PowerSystem.data['load'],
                                  len(PowerSystem.data['hydro_units']),
                                  len(PowerSystem.data['thermal_units'])))
```

### Dispatching a `PowerSystem`

#### **dispatch()** accepts the following arguments:

- `solver : str, optional defaults to 'sdp'`
  - Selects the solver option for the minimization objective function.

- `scenario : int, optional defaults to 0`
  - Chooses either a specific scenario to investigate (`scenario>1`) or all scenarios to evaluate (`scenario= 0`). Starting from 0 to the number of declared scenarios in the `hydro_units['inflow_scenarios']` parameter.

- `verbose : bool, optional defaults to False`
  - Displays the PDDE solution for every stage of the execution. Use with care, solutions of complex systems with too many stages and scenarios might overflow the console.

- `plot : bool, optional, defaults to False`
  - Displays a sequence of plots showing the future cost function for every stage of the execution. 

The following example executes the Power System dispatch using the Unique Linear Programming method for the first scenario (id = 1) and outputs the optimization steps.

```Python
import powersddp as psddp

data = {'load': [50, 50, 50],
        'discretizations': 3,
        'stages': 3,
        'scenarios': 2,
        'outage_cost': 500,
        'hydro_units': [{'name': 'HU1',
                         'v_max': 100,
                         'v_min': 20,
                         'v_ini': 100,
                         'prod': 0.95,
                         'flow_max': 60,
                         'inflow_scenarios': [[23, 16], [19, 14], [15, 11]]}],
        'thermal_units': [{'name': 'GT1', 'capacity': 15, 'cost': 10},
                          {'name': 'GT2', 'capacity': 10, 'cost': 25}]}

PowerSystem = psddp.PowerSystem(data=data)
operation = PowerSystem.dispatch(solver='ulp', scenario=1, verbose=True)

print(operation)
```
<!-- <img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1"> -->
