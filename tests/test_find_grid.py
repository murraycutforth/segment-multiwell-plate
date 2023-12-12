import itertools
import unittest
import numpy as np
from segment_multiwell_plate.segment_multiwell_plate import _generate_grid_crop_coordinates, _find_grid_coords, _find_cell_edges, _fit_grid_parameters

class TestGridCropCoordinates(unittest.TestCase):

    def setUp(self):
        # Create a test image with Gaussian blobs
        self.image_2d = np.random.rand(100, 100)
        self.well_coords = [(i, j) for i, j in itertools.product(range(20, 100, 10), range(20, 100, 10))]

    def test_find_cell_edges(self):
        # Test case for _find_cell_edges function
        x0, dx, nx = 10.0, 2.0, 5
        cell_edges = _find_cell_edges(x0, dx, nx)

        # Assert that the result is a numpy array
        self.assertIsInstance(cell_edges, np.ndarray)
        self.assertEqual(len(cell_edges), nx + 1)

        # Assert that the cell edges are correctly computed
        expected_edges = np.linspace(x0 - dx / 2, x0 + (nx - 0.5) * dx, nx + 1)
        self.assertTrue(np.allclose(cell_edges, expected_edges))

    def test_fit_grid_parameters(self):
        # Test case for _fit_grid_parameters function
        peaks = np.array([10, 20, 30, 40, 50])
        grid_start, grid_cell_width = _fit_grid_parameters(peaks)

        # Assert that the result is a tuple of two floats
        self.assertIsInstance(grid_start, float)
        self.assertIsInstance(grid_cell_width, float)

    def test_find_grid_size(self):
        # Test case for _find_grid_size function
        image_shape = (100, 100)
        prominence = 0.2
        width = 2
        peak_atol = 2.0

        peaks_i, peaks_j = _find_grid_coords(self.well_coords, image_shape, prominence, width, peak_atol)

        print(peaks_i)

        # Assert that the result is a list of integers
        self.assertIsInstance(peaks_i, list)
        self.assertIsInstance(peaks_j, list)

    def test_generate_grid_crop_coordinates(self):
        # Test case for generate_grid_crop_coordinates function
        peak_prominence = 0.5
        width = 2
        peak_spacing_atol = 2.0

        i_vals, j_vals = _generate_grid_crop_coordinates(self.image_2d, self.well_coords, peak_prominence, width, peak_spacing_atol)

        # Assert that the result is a tuple of two numpy arrays
        self.assertIsInstance(i_vals, np.ndarray)
        self.assertIsInstance(j_vals, np.ndarray)
        self.assertEqual(len(i_vals), 9)
        self.assertEqual(len(j_vals), 9)


if __name__ == '__main__':
    unittest.main()
