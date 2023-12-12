import unittest
import numpy as np

from segment_multiwell_plate import find_well_centres
from segment_multiwell_plate.segment_multiwell_plate import _find_well_centres_2d


class TestFindWellCentresSimple(unittest.TestCase):

    def setUp(self):
        # Create some test data
        self.image_2d = np.random.rand(100, 100)
        self.image_3d = np.random.rand(3, 100, 100)

    def test_find_well_centres_2d(self):
        # Test case for _find_well_centres_2d function
        min_sigma = 2
        max_sigma = 5
        num_sigma = 4
        threshold = 0.03
        overlap = 0.0
        exclude_border = 1

        well_coords = _find_well_centres_2d(self.image_2d, min_sigma, max_sigma, num_sigma, threshold, overlap, exclude_border)

        # Assert that the result is a list of numpy arrays
        self.assertIsInstance(well_coords, list)
        for coord in well_coords:
            self.assertIsInstance(coord, np.ndarray)
            self.assertEqual(len(coord), 2)  # Each coordinate should have two values

    def test_find_well_centres(self):
        # Test case for find_well_centres function
        well_coords = find_well_centres(self.image_3d)

        # Assert that the result is a list of numpy arrays
        self.assertIsInstance(well_coords, list)
        for coord in well_coords:
            self.assertIsInstance(coord, np.ndarray)
            self.assertEqual(len(coord), 2)  # Each coordinate should have two values

    def test_invalid_image_shape(self):
        # Test case for invalid image shape
        invalid_image = np.random.rand(7, 4, 100, 100)  # 4D image

        with self.assertRaises(ValueError):
            find_well_centres(invalid_image)


class TestFindWellCentres(unittest.TestCase):

    def setUp(self):
        # Create a test image with Gaussian blobs
        self.image_2d = self.create_test_image()

    def create_test_image(self):
        # Create a 2D image with Gaussian blobs
        image_size = (100, 100)
        blob_positions = [(25, 25), (50, 50), (75, 75)]  # Example blob positions
        blob_sigma = 2

        image = np.zeros(image_size)
        for pos in blob_positions:
            y, x = np.ogrid[-pos[0]:image_size[0]-pos[0], -pos[1]:image_size[1]-pos[1]]
            blob = np.exp(-(x**2 + y**2) / (2.0 * blob_sigma**2))
            image += blob

        return image

    def test_find_well_centres(self):
        # Test case for find_well_centres function
        well_coords = find_well_centres(self.image_2d)

        # Assert that the result is a list of numpy arrays
        self.assertIsInstance(well_coords, list)

        # Check if the detected well coordinates are close to the actual blob positions
        for detected_coord, actual_coord in zip(well_coords, [(25, 25), (50, 50), (75, 75)]):
            self.assertTrue(np.allclose(detected_coord, actual_coord, atol=1e-1))


if __name__ == '__main__':
    unittest.main()

