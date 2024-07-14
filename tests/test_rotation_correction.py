import unittest
import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage
from scipy import ndimage

from segment_multiwell_plate.segment_multiwell_plate import _correct_rotations_l1nn, find_well_centres, _average_d_min


class TestRotationCorrection(unittest.TestCase):
    def test_average_d_min(self):
        # Test that the average_d_min function works as expected
        # Create a set of points in a 2D grid
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 5, 6)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.flatten(), Y.flatten()], axis=-1)

        # Test that the average distance between points is correct
        d_min = _average_d_min(points)
        self.assertAlmostEqual(d_min, 1, places=5)

        # Now rotate this grid a tiny bit
        for theta in np.random.uniform(-0.05, 0.05, 100):
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points_rotated = points @ rotation_matrix.T

            # Test that the average distance between points is larger than before
            d_min_rotated = _average_d_min(points_rotated)
            self.assertGreater(d_min_rotated, d_min)


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

            corrected_image, theta_star = _correct_rotations_l1nn(image, rotated_points, return_theta=True)
            rotation_theta_star = np.array([[np.cos(theta_star), -np.sin(theta_star)], [np.sin(theta_star), np.cos(theta_star)]])
            corrected_points = (rotated_points - offset) @ rotation_theta_star.T + offset

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

        n_tests = 30

        thetas = np.random.uniform(-0.5, 0.5, n_tests)

        errors = []

        for theta in thetas:
            im_rotated = ndimage.rotate(image, np.rad2deg(theta), mode='nearest')
            well_coords_rotated = np.array(find_well_centres(im_rotated, threshold=0.20))
            f0 = _average_d_min(well_coords_rotated)

            corrected_image, theta_star = _correct_rotations_l1nn(im_rotated, well_coords_rotated, return_theta=True)
            rot_mat = np.array([[np.cos(theta_star), -np.sin(theta_star)], [np.sin(theta_star), np.cos(theta_star)]])
            well_coords_rotated_corrected = well_coords_rotated @ rot_mat.T
            f_final = _average_d_min(well_coords_rotated_corrected)

            # print(f'Rotation: {theta:.4g} rad, theta_star = {theta_star:.4g}, f0 = {f0:.6g}, f_final = {f_final:.6g}')

            if False:
                fig, axs = plt.subplots(1, 2, figsize=(6, 3), dpi=200)
                fig.suptitle(f'Rotation: {theta:.2g} rad, error: {abs(theta_star + theta):.2g} rad')
                axs[0].imshow(im_rotated)
                axs[0].scatter(well_coords_rotated[:, 1], well_coords_rotated[:, 0], s=1, c='red')
                axs[1].imshow(corrected_image)
                plt.show()

            self.assertTrue(np.isclose(theta_star, -theta, atol=0.04),
                            f'Failed on theta={theta}, theta_star={theta_star}')

            errors.append(np.abs(theta_star + theta))

        self.assertGreater(0.02, np.mean(errors), 'Mean error too large')


    @unittest.skip("This test is just used for some diagnostic plots")
    def test_cost_vs_rotation_real_image(self):
        image = skimage.io.imread("test_plate_onerow.png", as_gray=True)
        well_coords_original = find_well_centres(image, threshold=0.20)

        thetas = np.linspace(-0.05, 0.05, 51)

        costs = []
        num_different_wells = []

        for theta in thetas:
            im_rotated = ndimage.rotate(image, np.rad2deg(theta), mode='nearest')
            well_coords_rotated = np.array(find_well_centres(im_rotated, threshold=0.20))
            cost = _average_d_min(well_coords_rotated)
            costs.append(cost)
            original_well_set = set(tuple(well) for well in well_coords_original.astype(int))
            rotated_well_set = set(tuple(well) for well in well_coords_rotated.astype(int))
            num_different_wells.append(len(rotated_well_set - original_well_set))

        plt.plot(thetas, costs)
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Cost')
        plt.title('Cost vs rotation angle (finding coords after rotation)')
        plt.show()

        plt.plot(thetas, num_different_wells)
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Number of different wells')
        plt.title('Number of different wells vs rotation angle (finding coords after rotation)')
        plt.show()

        well_coords_original = find_well_centres(image, threshold=0.20)

        costs = []
        for theta in thetas:
            rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            well_coords_rotated = well_coords_original @ rot_mat.T
            cost = _average_d_min(well_coords_rotated)
            costs.append(cost)

        plt.plot(thetas, costs)
        plt.xlabel('Rotation angle (rad)')
        plt.ylabel('Cost')
        plt.title('Cost vs rotation angle (rotating coords after finding once)')
        plt.show()


    def test_rotated_real_image_small_rotations(self):
        # Test that a real image which is already almost aligned will remain aligned
        image = skimage.io.imread("test_plate_onerow.png", as_gray=True)
        well_coords = np.array(find_well_centres(image, threshold=0.20))

        n_tests = 50

        np.random.seed(0)
        thetas = np.random.uniform(-0.05, 0.05, n_tests)

        errors = []

        for theta in thetas:
            im_rotated = ndimage.rotate(image, np.rad2deg(theta), mode='nearest')
            rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            well_coords_rotated = well_coords @ rot_matrix.T  # Note that these coords won't exactly match the image due to the padding

            corrected_image, theta_star = _correct_rotations_l1nn(im_rotated, well_coords_rotated, return_theta=True)

            self.assertTrue(np.isclose(theta_star, -theta, atol=1e-3),
                            f'Failed on theta={theta}, theta_star={theta_star}')

            errors.append(np.abs(theta_star + theta))

        self.assertGreater(5e-4, np.mean(errors), 'Mean error too large')



