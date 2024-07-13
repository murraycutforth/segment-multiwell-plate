import unittest
import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage
from scipy import ndimage

from segment_multiwell_plate.segment_multiwell_plate import correct_rotations, find_well_centres


class TestRotationCorrection(unittest.TestCase):
    def test_many_rotation_angles_fullgrid(self):
        # Apply and correct small rotation on rectangular grid of points, no missing wells

        image = np.zeros((10, 10))  # Arbitrary image for this test

        n_tests = 10
        thetas = [np.random.uniform(-np.pi / 4, np.pi / 4) for _ in range(n_tests)]
        offsets = [np.random.uniform(-2, 2, 2) for _ in range(n_tests)]
        Ls = [10**np.random.uniform(-1, 4) for _ in range(n_tests)]
        side_ratios = [np.random.uniform(0.2, 5) for _ in range(n_tests)]
        num_points = [np.random.randint(5, 25) for _ in range(n_tests)]

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


    def test_rotated_real_image(self):
        # Test a real rotated image
        image = skimage.io.imread("test_plate_onerow.png", as_gray=True)

        n_tests = 10

        thetas = np.random.uniform(-0.5, 0.5, n_tests)

        for theta in thetas:
            im_rotated = ndimage.rotate(image, np.rad2deg(theta), mode='nearest')
            well_coords_rotated = np.array(find_well_centres(im_rotated, threshold=0.25))
            corrected_image, corrected_points, theta_star = correct_rotations(im_rotated, well_coords_rotated, return_theta=True)

            if False:
                fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
                fig.suptitle(f'Rotation: {theta:.2g} rad')
                axs[0].imshow(im_rotated)
                axs[0].scatter(well_coords_rotated[:, 1], well_coords_rotated[:, 0])
                axs[1].imshow(corrected_image)
                axs[1].scatter(corrected_points[:, 1], corrected_points[:, 0])
                plt.show()

            self.assertTrue(np.isclose(theta_star, -theta, atol=0.05),
                            f'Failed on theta={theta}, theta_star={theta_star}')




