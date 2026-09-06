import unittest

from embodied_agent.simulation import TabletopWorld


def complete(world):
    for _ in range(100):
        if world.tick(0.04):
            return
    raise AssertionError("motion did not finish")


class SimulationTests(unittest.TestCase):
    def test_grasp_attachment_and_position_based_completion(self):
        world = TabletopWorld()
        self.assertFalse(world.inside_box())
        world.begin("pick", {"object": "red_block"})
        complete(world)
        self.assertTrue(world.holding)
        self.assertEqual(world.block, world.tip)
        world.begin("place", {"object": "red_block", "destination": "box"})
        complete(world)
        self.assertTrue(world.inside_box())
        self.assertFalse(world.holding)
        self.assertNotEqual(world.tip, world.block)
        self.assertTrue(world.begin("verify", {"object": "red_block", "destination": "box"}))

    def test_verify_before_place_fails(self):
        with self.assertRaisesRegex(ValueError, "verification failed"):
            TabletopWorld().begin("verify", {"object": "red_block"})

    def test_reject_wrong_object_and_missing_destination(self):
        world = TabletopWorld()
        with self.assertRaisesRegex(ValueError, "graspable"):
            world.begin("pick", {"object": "box"})
        world.begin("pick", {"object": "red_block"})
        complete(world)
        with self.assertRaisesRegex(ValueError, "destination"):
            world.begin("place", {"object": "red_block"})

    def test_stop_freezes_motion(self):
        world = TabletopWorld()
        world.begin("pick", {"object": "red_block"})
        world.tick(0.1)
        world.begin("stop", {})
        position = world.tip[:]
        for _ in range(100):
            world.tick(0.04)
        self.assertEqual(position, world.tip)
        self.assertFalse(world.active)

    def test_two_link_geometry_reaches_tip(self):
        world = TabletopWorld()
        state = world.snapshot()
        import math

        self.assertAlmostEqual(math.dist(state["base"], state["elbow"]), 0.42)
        self.assertAlmostEqual(math.dist(state["elbow"], state["tip"]), 0.42)
