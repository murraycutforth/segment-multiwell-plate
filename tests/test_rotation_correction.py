import unittest
import logging

import numpy as np
import matplotlib.pyplot as plt

from segment_multiwell_plate.segment_multiwell_plate import correct_rotations


class TestRotationCorrection(unittest.TestCase):
    def test_many_rotation_angles_fullgrid(self):
        # Apply and correct small rotation on rectangular grid of points, no missing wells

        image = np.zeros((10, 10))  # Arbitrary image for this test

        n_tests = 10
        thetas = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(n_tests)]
        offsets = [np.random.uniform(-2, 2, 2) for _ in range(n_tests)]
        Ls = [10**np.random.uniform(-1, 4) for _ in range(n_tests)]
        side_ratios = [np.random.uniform(0.2, 5) for _ in range(n_tests)]
        num_points = [np.random.randint(10, 100) for _ in range(n_tests)]

        for i in range(n_tests):
            x = np.linspace(0, Ls[i], num_points[i])
            y = np.linspace(0, Ls[i] / side_ratios[i], num_points[i])
            X, Y = np.meshgrid(x, y)
            points = np.stack([X.flatten(), Y.flatten()], axis=-1)
            points += offsets[i]

            offset = np.mean(points, axis=0)
            points_centred = points - offset
            theta = thetas[i]

            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points_centred_rotated = points_centred @ rotation_matrix.T
            rotated_points = points_centred_rotated + offset

            corrected_image, corrected_points = correct_rotations(image, rotated_points)

            if False:  # For debugging
                plt.scatter(points[:, 0], points[:, 1], label='Original points')
                plt.scatter(points_centred[:, 0], points_centred[:, 1], label='Original points centered')
                plt.scatter(points_centred_rotated[:, 0], points_centred_rotated[:, 1], label='Rotated points centered')
                plt.scatter(rotated_points[:, 0],rotated_points[:, 1], label='Rotated points')
                plt.scatter(corrected_points[:, 0], corrected_points[:, 1], marker='x', label='Corrected points')
                plt.legend()
                plt.show()

            # Check that corrected points are equal to original points
            self.assertTrue(np.allclose(corrected_points, points, atol=1e-2 * Ls[i]),
                            f'Failed on test {i}, theta={theta}, offset={offset}, L={Ls[i]}, side_ratio={side_ratios[i]}, num_points={num_points[i]}, avg_distance={np.mean(np.sqrt(np.sum((points - corrected_points)**2, axis=1)))}')
