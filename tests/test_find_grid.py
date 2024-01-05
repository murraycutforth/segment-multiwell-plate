import itertools
import unittest
import numpy as np
import skimage

from segment_multiwell_plate.segment_multiwell_plate import _generate_grid_crop_coordinates, _find_histogram_peaks, \
    _find_cell_edges, _fit_grid_parameters, find_well_centres, _filter_spurious_peaks, segment_multiwell_plate


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

        peaks_i, peaks_j = _find_histogram_peaks(self.well_coords, image_shape, prominence, width)

        self.assertIsInstance(peaks_i, list)
        self.assertIsInstance(peaks_j, list)
        self.assertListEqual([20, 30, 40, 50, 60, 70, 80, 90], peaks_i)

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

class TestFindGridOneRow(unittest.TestCase):

    def setUp(self):
        # Create a test image with Gaussian blobs
        self.image_2d = skimage.io.imread("test_plate_onerow.png", as_gray=True)
        self.well_coords = find_well_centres(self.image_2d, threshold=0.12)

    def test_find_row_peaks_one_row(self):
        # Test case for generate_grid_crop_coordinates function for edge case where there is a single well in a row
        peaks_i, peaks_j = _find_histogram_peaks(self.well_coords, self.image_2d.shape, prominence=1/25, width=2)

        peaks_i, spacing = _filter_spurious_peaks(peaks_i, threshold=0.2)
        
        self.assertAlmostEquals(22.0, spacing)
        self.assertListEqual([58, 79, 100, 122, 144, 165, 187, 209, 231, 253, 275, 296, 318, 339, 361, 380], peaks_i)

    def test_grid_crop_coords_onerow(self):
        i_vals, j_vals = _generate_grid_crop_coordinates(self.image_2d, self.well_coords, peak_prominence=1/25, width=2)
        self.assertEqual(17, len(i_vals))

    def test_segment_multiwell_plate_onerow(self):
        img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
            self.image_2d,
            peak_finder_kwargs={"peak_prominence": 1/25, "filter_threshold": 0.2},
            blob_log_kwargs={"threshold": 0.12},
            output_full=True)

        self.assertEqual(16, img_array.shape[0])
        self.assertEqual(24, img_array.shape[1])
        self.assertEqual(21, img_array.shape[-1])
        self.assertEqual(21, img_array.shape[-2])

    def test_segment_multiwell_plate_onerow_extra_wells(self):
        # In this test, we turn down the log threshold so there are FP well detections, and see if the peak finder can
        # successfully filter these. At 0.09 there are 3 FPs, at 0.08 the algorithm appears to get overwhelmed.
        img_array, well_coords, i_vals, j_vals = segment_multiwell_plate(
            self.image_2d,
            peak_finder_kwargs={"peak_prominence": 1/25, "filter_threshold": 0.2},
            blob_log_kwargs={"threshold": 0.09},
            output_full=True)

        self.assertEqual(16, img_array.shape[0])
        self.assertEqual(24, img_array.shape[1])


class TestFilterSpuriousPeaks(unittest.TestCase):
    def test_empty_list(self):
        _ = _filter_spurious_peaks([], threshold=0.2)

    def test_no_spurious_peaks(self):
        peaks_original = [10, 20, 30, 40, 50]
        peaks_filtered, _ = _filter_spurious_peaks(peaks_original, threshold=0.2)
        self.assertListEqual(peaks_original, peaks_filtered)

    def test_extra_peak_middle(self):
        peaks_original = [10, 20, 25, 30, 40, 50]
        peaks_filtered, _ = _filter_spurious_peaks(peaks_original, threshold=0.2)
        self.assertListEqual([10, 20, 30, 40, 50], peaks_filtered)

    def test_extra_peak_end(self):
        peaks_original = [10, 20, 30, 40, 50, 70]
        peaks_filtered, _ = _filter_spurious_peaks(peaks_original, threshold=0.2)
        self.assertListEqual([10, 20, 30, 40, 50], peaks_filtered)

    def test_extra_peak_start(self):
        peaks_original = [5, 10, 20, 30, 40, 50]
        peaks_filtered, _ = _filter_spurious_peaks(peaks_original, threshold=0.2)
        self.assertListEqual([10, 20, 30, 40, 50], peaks_filtered)

    def test_extra_two_peaks_start(self):
        peaks_original = [4, 5, 10, 20, 30, 40, 50]
        peaks_filtered, _ = _filter_spurious_peaks(peaks_original, threshold=0.2)
        self.assertListEqual([10, 20, 30, 40, 50], peaks_filtered)



if __name__ == '__main__':
    unittest.main()
