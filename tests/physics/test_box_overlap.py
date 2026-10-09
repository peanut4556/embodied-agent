"""Static SAT regressions, including an independent geometry-engine oracle."""

import unittest

import mujoco
import numpy as np

from embodied_agent.box_overlap import aabb_as_obb, obb_overlap
from embodied_agent.gripper_sweep import box_aabb, overlaps


class BoxOverlapTests(unittest.TestCase):
    def test_touch_overlap_and_separation(self):
        a = (np.zeros(3), np.eye(3), np.ones(3))
        self.assertTrue(obb_overlap(a, ([2, 0, 0], np.eye(3), np.ones(3))))
        self.assertFalse(obb_overlap(a, ([2.001, 0, 0], np.eye(3), np.ones(3))))
        self.assertTrue(obb_overlap(a, (np.zeros(3), np.eye(3), np.full(3, 0.1))))

    def test_rotated_slender_boxes_aabbs_overlap_but_boxes_do_not(self):
        c = np.sqrt(0.5)
        rotation = np.array([[c, -c, 0], [c, c, 0], [0, 0, 1]])
        a = (np.zeros(3), rotation, [0.8, 0.02, 0.02])
        b = (np.array([-0.2, 0.2, 0]), rotation, [0.8, 0.02, 0.02])
        self.assertTrue(overlaps(box_aabb(*a), box_aabb(*b)))
        self.assertFalse(obb_overlap(a, b))
        self.assertFalse(obb_overlap(b, a))

    def test_aabb_conversion(self):
        bounds = (np.array([-1, -2, -3]), np.array([3, 2, 1]))
        np.testing.assert_allclose(box_aabb(*aabb_as_obb(bounds)), bounds)

    def test_invalid_geometry_rejected(self):
        valid = (np.zeros(3), np.eye(3), np.ones(3))
        for invalid in (
            ([np.nan, 0, 0], np.eye(3), np.ones(3)),
            (np.zeros(3), np.eye(3) * 2, np.ones(3)),
            (np.zeros(3), np.eye(3), [-1, 1, 1]),
        ):
            with self.assertRaises(ValueError):
                obb_overlap(valid, invalid)
        with self.assertRaises(ValueError):
            obb_overlap(valid, valid, tolerance=-1)

    def test_random_rotated_boxes_against_mujoco_contacts(self):
        model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
          <body mocap="true"><geom name="a" type="box" size=".1 .04 .02"/></body>
          <body mocap="true"><geom name="b" type="box" size=".08 .03 .05"/></body>
        </worldbody><contact><pair geom1="a" geom2="b"/></contact></mujoco>""")
        data = mujoco.MjData(model)
        rng = np.random.default_rng(73)
        counts = [0, 0]
        for _ in range(300):
            data.mocap_pos[:] = rng.uniform(-0.1, 0.1, (2, 3))
            quat = rng.normal(size=(2, 4))
            data.mocap_quat[:] = quat / np.linalg.norm(quat, axis=1)[:, None]
            mujoco.mj_forward(model, data)
            boxes = [
                (data.geom_xpos[i], data.geom_xmat[i].reshape(3, 3), model.geom_size[i])
                for i in range(2)
            ]
            expected = any(c.dist <= 1e-9 for c in data.contact)
            counts[int(expected)] += 1
            self.assertEqual(obb_overlap(*boxes), expected)
        self.assertTrue(all(n > 30 for n in counts))
