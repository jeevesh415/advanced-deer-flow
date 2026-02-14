import unittest
import os
import shutil
from recursive_ai.core.simulation import WorldModelSimulator

class TestSimulation(unittest.TestCase):
    def tearDown(self):
        # Cleanup any leftover dirs
        if os.path.exists("recursive_ai/test_sim"):
            shutil.rmtree("recursive_ai/test_sim")
        if os.path.exists("recursive_ai/test_sim_fail"):
            shutil.rmtree("recursive_ai/test_sim_fail")

    def test_simulation_success(self):
        """Test successful code simulation."""
        # The simulator creates the base_path directory in __init__
        # but cleanup only removes the *specific sim_id* subdir.
        # The base dir remains. This is expected behavior.

        sim = WorldModelSimulator(base_path="recursive_ai/test_sim")
        code = "print('Hello World')"
        success, output = sim.simulate_execution(code, "python main.py")

        self.assertTrue(success)
        self.assertIn("Hello World", output)

        # Verify base dir still exists (it's persistent for the simulator instance)
        self.assertTrue(os.path.exists("recursive_ai/test_sim"))

        # Verify subdirs are empty (sim_id folder deleted)
        self.assertEqual(len(os.listdir("recursive_ai/test_sim")), 0)

    def test_simulation_failure(self):
        """Test failing code."""
        sim = WorldModelSimulator(base_path="recursive_ai/test_sim_fail")
        code = "raise Exception('Fail')"
        success, output = sim.simulate_execution(code, "python main.py")

        self.assertFalse(success)
        self.assertIn("Fail", output)

        self.assertEqual(len(os.listdir("recursive_ai/test_sim_fail")), 0)

if __name__ == "__main__":
    unittest.main()
