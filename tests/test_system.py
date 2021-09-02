import pandas as pd

from unittest import TestCase
from powersddp import PowerSystem


class TestSystem(TestCase):
    def test_PowerSystem_should_load_from_file(self):
        with self.subTest():
            System = PowerSystem(path="systems-data/system.yml")

            # Structure
            self.assertEqual(type(System.data), dict)

            # Content
            self.assertTrue(System.data["load"] == [50, 50, 50])
            self.assertTrue(System.data["hydro_units"][0]["name"] == "HU1")

    def test_PowerSystem_should_load_from_payload(self):
        with self.subTest():
            payload = {
                "load": [50, 50, 50],
                "discretizations": 3,
                "stages": 3,
                "scenarios": 2,
                "outage_cost": 500,
                "hydro_units": [
                    {
                        "name": "HU1",
                        "v_max": 100,
                        "v_min": 20,
                        "v_ini": 100,
                        "prod": 0.95,
                        "flow_max": 60,
                        "inflow_scenarios": [[23, 16], [19, 14], [15, 11]],
                    }
                ],
                "thermal_units": [
                    {"name": "GT1", "capacity": 15, "cost": 10},
                    {"name": "GT2", "capacity": 10, "cost": 25},
                ],
            }

            System = PowerSystem(data=payload)
            # Structure
            self.assertEqual(type(System.data), dict)

            # Content
            self.assertTrue(System.data["load"] == [50, 50, 50])
            self.assertTrue(System.data["hydro_units"][0]["name"] == "HU1")

    def test_PowerSystem_should_dispatch_sdp_and_get_correct_results(self):
        with self.subTest():
            payload = {
                "load": [50, 50, 50],
                "discretizations": 3,
                "stages": 3,
                "scenarios": 2,
                "outage_cost": 500,
                "hydro_units": [
                    {
                        "name": "HU1",
                        "v_max": 100,
                        "v_min": 20,
                        "prod": 0.95,
                        "flow_max": 60,
                        "inflow_scenarios": [[23, 16], [19, 14], [15, 11]],
                    }
                ],
                "thermal_units": [
                    {"name": "GT1", "capacity": 15, "cost": 10},
                    {"name": "GT2", "capacity": 10, "cost": 25},
                ],
            }

            System = PowerSystem(data=payload)

            # Dispatching
            result = System.dispatch(solver="sdp")
            operation = result["operation_df"]

            # Calculating Mean results
            df = operation.drop(columns=["hydro_units", "thermal_units"], axis=1)
            df_mean = (
                df.groupby(["stage", "initial_volume"])
                .mean()
                .reset_index()
                .sort_values(by=["stage", "initial_volume"], ascending=[False, True])
                .round(3)
            )

            # Assert Structure
            self.assertEqual(type(result), dict)
            self.assertEqual(type(operation), pd.DataFrame)

            # Assert Values
            self.assertEqual(
                df_mean.total_cost.tolist(),
                [6725.0, 7.75, 0.0, 11787.5, 226.93, 0.625, 15425.0, 576.31, 161.68],
            )

    def test_PowerSystem_should_dispatch_ulp(self):
        with self.subTest():
            payload = {
                "load": [50, 50, 50],
                "discretizations": 3,
                "stages": 3,
                "scenarios": 2,
                "outage_cost": 500,
                "hydro_units": [
                    {
                        "name": "HU1",
                        "v_max": 100,
                        "v_min": 20,
                        "v_ini": 100,
                        "prod": 0.95,
                        "flow_max": 60,
                        "inflow_scenarios": [[23, 16], [19, 14], [15, 11]],
                    }
                ],
                "thermal_units": [
                    {"name": "GT1", "capacity": 15, "cost": 10},
                    {"name": "GT2", "capacity": 10, "cost": 25},
                ],
            }

            System = PowerSystem(data=payload)

            # Dispatching
            optimistic_operation = System.dispatch(solver="ulp", scenario=1)
            pessimistic_operation = System.dispatch(solver="ulp", scenario=2)

            # Assert Structure
            self.assertEqual(type(optimistic_operation), dict)
            self.assertEqual(type(optimistic_operation["hydro_units"]), pd.DataFrame)
            self.assertEqual(type(optimistic_operation["thermal_units"]), pd.DataFrame)

            # Assert Values
            self.assertEqual(optimistic_operation["total_cost"], 198.5)
            self.assertEqual(pessimistic_operation["total_cost"], 350.5)

    def test_PowerSystem_should_dispatch_two_hgus_using_ulp(self):
        with self.subTest():
            System = PowerSystem(path="systems-data/system-2hgu.yml")

            # Dispatching
            operation = System.dispatch(solver="ulp", scenario=1)

            # Structure
            self.assertEqual(type(System.data), dict)

            # Content
            self.assertTrue(System.data["load"] == [150, 150, 150])

            # Results
            self.assertEqual(operation["total_cost"], 7175.0)
