# Segment Multiwell Plates

This is an image analysis python package, for automatically segmenting an image of a multiwell plate into an array of
sub-images. This is useful as part of a pipeline in high-throughput screening experiments.

![segment_multiwell_plate_schematic](https://github.com/murraycutforth/segment-multiwell-plate/assets/11088372/43852418-7767-4e7f-aba9-2da69ed3eaad)


## Installation

To use functions from this package, install into your environment using pip:

`pip install segment-multiwell-plate`

For a developer install, this repo can be installed with pipenv:

`pipenv install --dev`

The only code dependencies are `python-3.11`, `numpy`, `scipy`, and `scikit-image`. 


## Usage

With defaults:

    img_array = segment_multiwell_plate(image)

Image can either by a 2D or 3D (channel, height, width) numpy array. Adjust resampling spline order and resolution of sub-images:

    img_array = segment_multiwell_plate(image, resampling_order=3, subcell_resolution=32)

 Detailed control of algorithm parameters can be obtained by passing extra parameters:

    img_array = segment_multiwell_plate(
      image,
      resampling_order=1,
      subcell_resolution=20,
      blob_log_kwargs=dict(min_sigma=1, max_sigma=6, num_sigma=7, threshold=0.05, overlap=0.0, exclude_border=1),
      peak_finder_kwargs=dict(peak_prominence=0.2, width=2, filter_threshold=0.2))

Extra output (the well coordinates, and the peak coordinates, both in image space) can be obtained like:

    img_array, well_coords, i_peak_vals, j_peak_vals = segment_multiwell_plate(image, output_full=True)



## The Algorithm

1. Use the Laplacian of Gaussians method (implemented in `scikit-image`) to find well centre coordinates
2. Rotate the image and well centres so that the well grid is aligned with the coordinate axes. This is done by finding a rotation which minimises the Manhattan (L1) distance between neighbouring well centres.
3. For each of the x- and y- axes in turn:

     a. Project all well centres onto this axis
  
     b. Compute a histogram of well centre coordinates
  
     c. Find peaks in this histogram using `scipy.signal.find_peaks()` - these correspond to estimated x/y coordinates of cell centres in the grid. However, at this point the estimated cell centres will be slightly irregular.
An example of this histogram is: ![peak_hist_640](https://github.com/murraycutforth/segment-multiwell-plate/assets/11088372/f65e0ef3-e483-464f-8608-67d44eb4d869)

4.  A regular 2D Cartesian grid is defined by $x0, \Delta x, N_x$ and $y0, \Delta y, N_y$ - the start point, spacing, and number of cells along each axis.
The number of cells is the number of peaks estimated in the previous step. The other two parameters are computed as the solution to an overdetermined (N x 2) linear
system fitting a regular grid to the estimated cell centres, where we get the optimal (minimal L2 error) solution using a QR decomposition. For the x-axis, the linear system looks like:

$$
\begin{bmatrix}
    1 & 0 \\
    1 & 1 \\
    \vdots & \vdots \\
    1 & N-1
\end{bmatrix} \begin{bmatrix}
    x_0 \\
    \Delta x
\end{bmatrix} = \begin{bmatrix}
    \text{Peak}_1 \\
    \text{Peak}_2 \\
    \vdots \\
    \text{Peak}_N
\end{bmatrix}
$$

5. Finally we partition the original image into an array of sub-images, using this grid. Each sub-image is resampled from the original image using `scipy.ndimage.map_coordinates`,
which has options for high order spline interpolation.

 
## TODO

- The QR decomposition used in the linear least squares sub-problem could be replaced by an analytic solution, but the runtime is currently bottlenecked by the resampling so there's probably no need.


## Release steps

1. Update version number in `__version__.py`
2. `pipenv run python setup.py sdist bdist_wheel`
3. `pipenv run twine upload dist/*`
