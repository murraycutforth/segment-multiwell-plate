import unittest

import numpy as np
import matplotlib.pyplot as plt

from segment_multiwell_plate.segment_multiwell_plate import _find_rotation_angle, correct_small_rotations

class TestRotationCorrection(unittest.TestCase):

        def test_rotation_angle(self):
            # Apply and correct small rotation

            x = np.linspace(0, 10, 11)
            y = np.linspace(0, 5, 6)
            X, Y = np.meshgrid(x, y)
            points = np.stack([X.flatten(), Y.flatten()], axis=-1)

            theta = 0.1
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            rotated_points = points @ rotation_matrix.T

            correction_angle = _find_rotation_angle(rotated_points)

            rotation_matrix = np.array([[np.cos(correction_angle), -np.sin(correction_angle)],
                                        [np.sin(correction_angle), np.cos(correction_angle)]])

            corrected_points = np.array(rotated_points) @ rotation_matrix.T

            corrected_points_mean = np.mean(corrected_points, axis=0)
            original_points_mean = np.mean(points, axis=0)
            corrected_points += original_points_mean - corrected_points_mean

            plt.scatter(points[:, 0], points[:, 1], label='Original points')
            plt.scatter(rotated_points[:, 0], rotated_points[:, 1], label='Rotated points')
            plt.scatter(corrected_points[:, 0], corrected_points[:, 1], label='Corrected points')
            plt.legend()
            plt.show()

            # Check that corrected points are equal to original points
            self.assertTrue(np.allclose(np.array(corrected_points), points))

