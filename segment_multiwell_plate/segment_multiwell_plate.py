import functools
import itertools
import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import ndimage
from skimage import feature, filters

logger = logging.getLogger(__name__)


def segment_multiwell_plate(image: np.array,
                            resampling_order: int = 1,
                            subcell_resolution: int = None,
                            blob_log_kwargs: dict = None,
                            peak_finder_kwargs: dict = None,
                            output_full: bool = False):
    """Split an image of a multiwell plate into array of sub-images of each well

    Note: we assume that the image axes align with the well grid axes.

    Example with all possible args:

    img_array = segment_multiwell_plate(
      image,
      resampling_order=1,
      subcell_resolution=20,
      blob_log_kwargs=dict(min_sigma=1, max_sigma=6, num_sigma=7, threshold=0.05, overlap=0.0, exclude_border=1),
      peak_finder_kwargs=dict(peak_prominence=0.2, width=2, filter_threshold=0.2))

    :param image: 2D or 3D image of a multiwell plate
    :param resampling_order: order of interpolation used when resampling the image to each well
    :param subcell_resolution: resolution of the resampled image of each well. If None, this is chosen automatically.
    :param blob_log_kwargs: kwargs passed to skimage.feature.blob_log
    :param peak_finder_kwargs: kwargs passed to _generate_grid_crop_coordinates
    :param output_full: if True, return the full output of the segmentation algorithm, including well coordinates and
    grid crop coordinates
    :return: array of sub-images of each well, or tuple of (img_array, well_coords, i_vals, j_vals) if output_full=True
    """
    if blob_log_kwargs is None:
        blob_log_kwargs = {}

    well_coords = find_well_centres(image, **blob_log_kwargs)

    image, well_coords = correct_rotations(image, well_coords)

    if peak_finder_kwargs is None:
        peak_finder_kwargs = {}

    i_vals, j_vals = _generate_grid_crop_coordinates(image, well_coords, **peak_finder_kwargs)

    img_array = _grid_crop(image, i_vals, j_vals, subcell_resolution, resampling_order)

    if not output_full:
        return img_array
    else:
        return img_array, well_coords, i_vals, j_vals


def find_well_centres(image: np.array,
                      min_sigma=2,
                      max_sigma=5,
                      num_sigma=4,
                      threshold=0.2,
                      overlap=0.0,
                      exclude_border=1
                      ) -> np.array:
    """Use laplacian of gaussian method to find coordinates centred over each well.

    If a 3D image is given, it is averaged over axis 0 before locating wells.
    """
    if len(image.shape) == 2:
        image_2d = filters.gaussian(image, sigma=1, channel_axis=None)
    elif len(image.shape) == 3:
        image = filters.gaussian(image, sigma=1, channel_axis=0)
        image_2d = np.mean(image, axis=0)
    else:
        raise ValueError("Image must be 2D or 3D with shape (channels, height, width)")

    well_coords = _find_well_centres_2d(image_2d, min_sigma, max_sigma, num_sigma, threshold, overlap, exclude_border)

    return well_coords


def correct_rotations(image: np.array, well_coords: np.array, return_theta: bool = False) \
        -> tuple[np.array, np.array] | tuple[np.array, np.array, float]:
    """Automatically rotate the image so that it is parallel to the coordinate axes.

    Algorithm:
        Find smallest rotation angle such that the Manhattan distance metric between neighbouring points is minimised.
        We assume that a rotation of +- pi/4 is sufficient to correct the rotation. If the image has mistakenly
        been rotated by more than this, this algorithm will converge to an unexpected orientation (e.g. vertically
        rather than horizontally). This optimisation method is sensitive to the scale of the coordinates, so we
        rescale the well coords to unit length during the algorithm.

    Returns:
        - rotated image
        - rotated well coordinates
    """
    assert len(image.shape) == 2
    assert len(well_coords.shape) == 2
    assert well_coords.shape[1] == 2

    def average_d_min(X: np.array) -> float:
        """Compute the min Manhattan distance, averaged across all entries of X
        """
        assert X.shape[1] == 2
        assert len(X.shape) == 2

        # Compute all pairwise manhattan distances: dists[i, j] = || x_i - x_j ||_1
        diffs = X[:, :, None] - X.T[None, :, :]
        dists = np.sum(np.abs(diffs), axis=1)

        # Now for each column, find the min value, excepting the diagonal
        inf_diags = np.diag(np.full(len(dists), fill_value=np.inf))
        dists = dists + inf_diags

        min_dists = np.min(dists, axis=0)
        mean_min_dist = np.mean(min_dists)
        return mean_min_dist

    def cost(rotation_angle) -> float:
        """Cost function to be minimised: find the rotation angle which minimises the L1 distance between neighbours
        """
        assert -np.pi <= rotation_angle <= np.pi

        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        rotated_coords = well_coords @ rotation_matrix.T

        cost = average_d_min(rotated_coords)

        return cost

    # First, centre the well coordinates so the rotation does not affect average position
    original_well_centroid = np.mean(well_coords, axis=0)
    well_coords -= original_well_centroid

    # Rescale the well cords to approx unit length (must be isotropic)
    scale_fac = np.max(well_coords.max(axis=0) - well_coords.min(axis=0))
    well_coords /= scale_fac

    result = scipy.optimize.minimize_scalar(cost, bounds=(-np.pi / 4, np.pi / 4), method='bounded', tol=1e-3)
    dx = result.fun - result.x
    theta = result.x

    logger.debug(f"Correcting rotation by {np.rad2deg(theta):.2f} degrees")
    logger.debug(f"Converged Manhattan distance: {dx:.2f}")

    rotated_image = ndimage.rotate(image, np.rad2deg(theta), mode='nearest', reshape=False)
    rotated_wells = well_coords @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]).T

    # Undo the normalisations to the data
    rotated_wells *= scale_fac
    well_coords *= scale_fac
    rotated_wells += original_well_centroid
    well_coords += original_well_centroid

    if return_theta:
        return rotated_image, rotated_wells, theta
    else:
        return rotated_image, rotated_wells


def _find_well_centres_2d(image_2d: np.array,
                          min_sigma,
                          max_sigma,
                          num_sigma,
                          threshold,
                          overlap,
                          exclude_border
                          ) -> np.array:
    """Use laplacian of gaussian method to find coordinates centred over each well
    """
    assert len(image_2d.shape) == 2

    image_2d = image_2d / image_2d.max()  # Normalise to [0,1]

    well_coords = feature.blob_log(image_2d,
                                   min_sigma=min_sigma,
                                   max_sigma=max_sigma,
                                   num_sigma=num_sigma,
                                   threshold=threshold,
                                   overlap=overlap,
                                   exclude_border=exclude_border)

    well_coords = list(map(lambda x: x[:2], well_coords))  # Discard sigmas of blobs
    return np.array(well_coords)


def _generate_grid_crop_coordinates(image: np.array,
                                    well_coords: list[np.array],
                                    peak_prominence: float = 0.2,
                                    width: int = 2,
                                    filter_threshold: float = 0.2,
                                    ) -> tuple[np.array, np.array]:
    """Automatically find the grid of wells in the image stack

    :returns: Arrays of the cell edge coordinates in each dimension
    """
    # TODO: we assume that image axes align with well grid axes here
    peaks_i, peaks_j = _find_histogram_peaks(well_coords,
                                             image_shape=(image.shape[-2], image.shape[-1]),
                                             prominence=peak_prominence,
                                             width=width)

    peaks_i, median_di = _filter_spurious_peaks(peaks_i, threshold=filter_threshold)
    peaks_j, median_dj = _filter_spurious_peaks(peaks_j, threshold=filter_threshold)

    num_rows = len(peaks_i)
    num_cols = len(peaks_j)

    logger.debug(f"Found num_rows={num_rows} and num_cols={num_cols} of wells in image")

    # Find the grid start coord and dx through a linear least squares problem
    i0, di = _fit_grid_parameters(peaks_i)
    j0, dj = _fit_grid_parameters(peaks_j)

    logger.debug(f"Optimal grid params found: i0={i0}, di={di}, j0={j0}, dj={dj}")

    # Assume that wells are equally spaced in both dimensions
    if abs(di - dj) / di > 0.1:
        logger.warning(f"Unequal well spacing found. di={di:.2f}, dj={dj:.2f}")

    # Convert to cell edges and return
    i_vals = _find_cell_edges(i0, di, num_rows)
    j_vals = _find_cell_edges(j0, dj, num_cols)

    return i_vals, j_vals


def _filter_spurious_peaks(peaks: list, threshold=0.2) -> tuple[list, float]:
    """Detect and remove false positive wells, using assumption of a regular grid
    """
    if not peaks:
        return peaks

    intervals = np.diff(peaks)

    median_interval = np.median(intervals)  # We use this as our estimate of grid spacing, robust to outliers

    relative_differences = np.abs((intervals - median_interval) / median_interval)

    # True where the interval is anomalous - probably not a true peak of the grid
    outliers = relative_differences > threshold

    # We pad so that outliers on either end are found
    outliers = np.pad(outliers, pad_width=1, mode="constant", constant_values=True)

    # Find pairs of "True" values
    fn = lambda x: x[0] & x[1]
    outliers = list(map(fn, itertools.pairwise(outliers)))

    outlier_inds = np.where(outliers)[0]

    filtered_time_points = list(np.delete(peaks, outlier_inds))

    return filtered_time_points, median_interval


def _find_histogram_peaks(well_coords: list[np.array],
                          image_shape: tuple[int, int],
                          prominence: float,
                          width: int,
                          ) -> tuple[list, list]:
    """Count the number of peaks on the histogram of well centre coordinates, in order to find cell centres on each axis
    """
    def find_n_one_axis(coords, n) -> list:
        # Find the well coordinates along one axis by finding peaks in the histogram
        hist, _ = np.histogram(coords, range=(0, n), bins=n)
        smoothed_hist = ndimage.gaussian_filter1d(hist.astype(np.float32), 1)
        peaks, _ = scipy.signal.find_peaks(smoothed_hist / smoothed_hist.max(), prominence=prominence, width=width)
        return list(peaks)

    i_coords = [x[0] for x in well_coords]
    j_coords = [x[1] for x in well_coords]
    peaks_i = find_n_one_axis(i_coords, image_shape[0])
    peaks_j = find_n_one_axis(j_coords, image_shape[1])

    return peaks_i, peaks_j


def _find_cell_edges(x0: float, dx: float, nx: int) -> np.array:
    """Given grid parameters, find coords of grid cell edges
    """
    return np.linspace(x0 - dx / 2, x0 + (nx - 0.5) * dx, nx + 1)


def _fit_grid_parameters(peaks: np.array) -> tuple[float, float]:
    """Linear least squares solver to find grid start coordinate and increment, fitted to the well centres.

    This function is fitting a regular grid to the irregular cell centre points found previously.

    :returns: tuple of (grid start coordinate, grid cell width)
    """
    @functools.lru_cache
    def compute_qr(N):
        # Use lru_cache so we don't needlessly recompute the QR decomposition
        A = np.ones((N, 2), dtype=np.float32)
        A[:, 1] = range(N)
        Q, R = np.linalg.qr(A)
        return Q, R

    # Least squares solution to Ax=b -> QRx=b -> x=R^(-1)Q^(T)b
    peaks = np.array(peaks)
    N = len(peaks)

    assert N > 0

    Q, R = compute_qr(N)
    b = Q.T @ peaks
    x = scipy.linalg.solve_triangular(R, b, check_finite=False)

    # x[0] is grid start coord, x[1] is grid cell width
    return x[0], x[1]


#def visualise_peak_hist(hist, peaks, smoothed_hist, savedir):
#    logger.debug(f"Writing out plots of all time points in {savedir}")
#
#    savedir.mkdir(parents=True, exist_ok=True)
#
#    fig, ax = plt.subplots(1, 1)
#
#    ax.plot(hist, label="raw")
#    ax.plot(smoothed_hist, label="smoothed")
#    ax.vlines(peaks, -1, 1, color="red")
#    ax.legend()
#    fig.savefig(savedir / f"peak_hist_{len(hist)}.png")
#    fig.clf()
#    plt.close(fig)


def _resample_2d_image(image_2d, iv, jv, subcell_resolution, order):
    subcell_img = ndimage.map_coordinates(image_2d, [iv.ravel(), jv.ravel()], order=order)
    subcell_img = subcell_img.reshape((subcell_resolution, subcell_resolution))
    return subcell_img


def _grid_crop(image: np.array,
               i_vals: np.array,
               j_vals: np.array,
               subcell_resolution: int,
               resampling_order: int,
               ) -> np.array:
    """Crop and resample the image to each well
    """

    logger.debug("Starting grid_crop...")

    for i, i_next in itertools.pairwise(i_vals):
        assert abs(i_next - i - i_vals[1] + i_vals[0]) < np.finfo(np.float32).eps, "Unequal height of grid cells"
    for j, j_next in itertools.pairwise(j_vals):
        assert abs(j_next - j - j_vals[1] + j_vals[0]) < np.finfo(np.float32).eps, "Unequal width of grid cells"

    img_shape = image.shape
    grid_shape = (len(i_vals), len(j_vals))

    if subcell_resolution is None:
        dx = j_vals[1] - j_vals[0]
        dy = i_vals[1] - i_vals[0]
        subcell_resolution = int(0.5 * (dx + dy))
        logger.debug(f"Automatically setting subcell_resolution to {subcell_resolution}")

    # Prepare storage for resampled well images
    if len(img_shape) == 3:
        img_array = np.zeros(shape=(grid_shape[0] - 1, grid_shape[1] - 1, img_shape[0], subcell_resolution, subcell_resolution),
                          dtype=np.float32)
    else:
        img_array = np.zeros(shape=(grid_shape[0] - 1, grid_shape[1] - 1, subcell_resolution, subcell_resolution),
                          dtype=np.float32)
        image = [image]  # So we can seamlessly use map() below

    for i, j in itertools.product(range(grid_shape[0] - 1), range(grid_shape[1] - 1)):
        # We want to resample a subcell_res x subcell_res image from the cell centred over well (i,j)
        i_start, i_end, j_start, j_end = i_vals[i], i_vals[i+1], j_vals[j], j_vals[j+1]

        # These are the coordinates in image space which we want to resample from original image, for sub-image i,j
        iv, jv = np.meshgrid(np.linspace(i_start, i_end, subcell_resolution, dtype=np.float32),
                             np.linspace(j_start, j_end, subcell_resolution, dtype=np.float32),
                             indexing="ij")

        resample_fn = functools.partial(_resample_2d_image, iv=iv, jv=jv, subcell_resolution=subcell_resolution, order=resampling_order)

        # Can be parallelised - speeds up 3rd order but slows down 1st order due to overhead
        #if USE_MULTIPROCESSING:
        #    with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as p:
        #        subcell_images = list(p.starmap(_resample_2d_image, resample_args))

        subcell_images = list(map(resample_fn, image))

        img_array[i, j] = np.squeeze(np.array(subcell_images))  # Squeeze used to reduce dimensionality if input is 2D

    return img_array


