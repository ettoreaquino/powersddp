from unittest import TestCase

from powersddp import PowerSystem

import pandas as pd


class TestSystem(TestCase):
    def test_PowerSystem_should_load_from_file(self):
        with self.subTest():
            System = PowerSystem(path="system.yml")

            # Structure
            self.assertEqual(type(System.data), dict)

            # Content
            self.assertTrue(System.data["load"] == [50, 50, 50])
            self.assertTrue(System.data["hydro-units"][0]["name"] == "HU1")

    def test_PowerSystem_should_load_from_payload(self):
        with self.subTest():
            payload = {
                "load": [50, 50, 50],
                "discretizations": 3,
                "stages": 3,
                "scenarios": 2,
                "outage_cost": 500,
                "hydro-units": [
                    {
                        "name": "HU1",
                        "v_max": 100,
                        "v_min": 20,
                        "prod": 0.95,
                        "flow_max": 60,
                        "inflow_scenarios": [[23, 16], [19, 14], [15, 11]],
                    }
                ],
                "thermal-units": [
                    {"name": "GT1", "capacity": 15, "cost": 10},
                    {"name": "GT2", "capacity": 10, "cost": 25},
                ],
            }

            System = PowerSystem(data=payload)
            # Structure
            self.assertEqual(type(System.data), dict)

            # Content
            self.assertTrue(System.data["load"] == [50, 50, 50])
            self.assertTrue(System.data["hydro-units"][0]["name"] == "HU1")

    def test_PowerSystem_should_dispatch(self):
        with self.subTest():
            payload = {
                "load": [50, 50, 50],
                "discretizations": 3,
                "stages": 3,
                "scenarios": 2,
                "outage_cost": 500,
                "hydro-units": [
                    {
                        "name": "HU1",
                        "v_max": 100,
                        "v_min": 20,
                        "prod": 0.95,
                        "flow_max": 60,
                        "inflow_scenarios": [[23, 16], [19, 14], [15, 11]],
                    }
                ],
                "thermal-units": [
                    {"name": "GT1", "capacity": 15, "cost": 10},
                    {"name": "GT2", "capacity": 10, "cost": 25},
                ],
            }

            System = PowerSystem(data=payload)
            
            # Dispatching                
            operation = System.dispatch()

            # Assert Structure
            self.assertEqual(type(operation), pd.DataFrame)

            # Assert Values
            self.assertEqual(operation.iloc[0].average_cost, 6725.0)


