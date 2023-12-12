import functools
import itertools
import logging

import numpy as np
import scipy
from scipy import ndimage
from skimage import feature, filters

logger = logging.getLogger(__name__)


def segment_multiwell_plate(image: np.array,
                            resampling_order: int = 1,
                            subcell_resolution: int = 20,
                            blob_log_kwargs: dict = None,
                            peak_finder_kwargs: dict = None,
                            output_full: bool = False):
    """Split an image of a multiwell plate into array of sub-images of each well

    Note: we assume that the image axes align with the well grid axes.
    """
    if blob_log_kwargs is None:
        blob_log_kwargs = {}

    well_coords = find_well_centres(image, **blob_log_kwargs)

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
                      threshold=0.03,
                      overlap=0.0,
                      exclude_border=1
                      ) -> list[np.array]:
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


def _find_well_centres_2d(image_2d: np.array,
                          min_sigma,
                          max_sigma,
                          num_sigma,
                          threshold,
                          overlap,
                          exclude_border
                          ) -> list[np.array]:
    """Use laplacian of gaussian method to find coordinates centred over each well
    """
    assert len(image_2d.shape) == 2

    image_2d = image_2d / image_2d.max()

    well_coords = feature.blob_log(image_2d,
                                   min_sigma=min_sigma,
                                   max_sigma=max_sigma,
                                   num_sigma=num_sigma,
                                   threshold=threshold,
                                   overlap=overlap,
                                   exclude_border=exclude_border)

    well_coords = list(map(lambda x: x[:2], well_coords))  # Discard sigmas of blobs
    return well_coords


def _generate_grid_crop_coordinates(image_2d: np.array,
                                    well_coords: list[np.array],
                                    peak_prominence: float = 0.2,
                                    width: int = 2,
                                    peak_spacing_atol: float = 2.
                                    ) -> tuple[np.array, np.array]:
    """Automatically find the grid of wells in the image stack
    """
    # TODO: we assume that image axes align with well grid axes here
    peaks_i, peaks_j = _find_grid_coords(well_coords, image_2d.shape, peak_prominence, width, peak_spacing_atol)

    num_rows = len(peaks_i)
    num_cols = len(peaks_j)

    logger.debug(f"Found num_rows={num_rows} and num_cols={num_cols} of wells in image")

    # Find the grid start coord and dx through a linear least squares problem
    i0, di = _fit_grid_parameters(peaks_i)
    j0, dj = _fit_grid_parameters(peaks_j)

    logger.debug(f"Optimal grid params found: i0={i0}, di={di}, j0={j0}, dj={dj}")

    # Assume that wells are equally spaced in both dimensions
    assert abs(di - dj) < peak_spacing_atol

    # Convert to cell edges and return
    i_vals = _find_cell_edges(i0, di, num_rows)
    j_vals = _find_cell_edges(j0, dj, num_cols)

    return i_vals, j_vals


def _find_grid_coords(well_coords: list[np.array],
                      image_shape: tuple[int, int],
                      prominence: float,
                      width: int,
                      peak_atol: float,
                      ) -> tuple[list, list]:
    """Count the number of peaks on the histogram of well centre coordinates, in order to find cell centres on each axis
    """
    def find_n_one_axis(coords, n) -> list:
        # Find the well coordinates along one axis by finding peaks in the histogram

        hist, _ = np.histogram(coords, range=(0, n), bins=n)
        smoothed_hist = ndimage.gaussian_filter1d(hist, 2)
        peaks, _ = scipy.signal.find_peaks(smoothed_hist / smoothed_hist.max(), prominence=prominence, width=width)

        # Check that this result is approximately correct by looking at the difference between each peak coordinate
        diffs = np.diff(peaks, n=1)
        assert np.allclose(diffs, np.mean(diffs), atol=peak_atol)  # Peaks should be regularly spaced to within +/- 2 pixels

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

        subcell_images = map(resample_fn, image)

        img_array[i, j] = np.squeeze(np.array(subcell_images))  # Squeeze used to reduce dimensionality if input is 2D

    return img_array


