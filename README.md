# Segment Multiwell Plates

This is an image analysis python package, for automatically segmenting an image of a multiwell plate into an array of
sub-images. This is useful as part of a pipeline in high-throughput screening experiments.

![segment_multiwell_plate_schematic](https://github.com/murraycutforth/segment-multiwell-plate/assets/11088372/43852418-7767-4e7f-aba9-2da69ed3eaad)


## Installation

To use functions from this package, install into your environment using pip:

`pip install segment-multiwell-plate`

For a developer install, this repo can be installed with pipenv:

`pipenv install --dev`


## The Algorithm

1. Use the Laplacian of Gaussians method (implemented in `scikit-image`) to find well centre coordinates
2. For each of the x- and y- axes in turn:
  a. Project all well centres onto this axis
  b. Compute a histogram of well centre coordinates
  c. Find peaks in this histogram using `scipy.signal.find_peaks()` - these correspond to estimated x/y coordinates of cell centres in the grid. However, at this point the estimated cell centres will be slightly irregular.
3.  A regular 2D Cartesian grid is defined by $x0, \Delta x, N_x$ and $y0, \Delta y, N_y$ - the start point, spacing, and number of cells along each axis.
The number of cells is the number of peaks estimated in the previous step. The other two parameters are computed as the solution to an overdetermined (N x 2) linear
system fitting a regular grid to the estimated cell centres, where we get the optimal (minimal L2 error) solution using a QR decomposition.
4. Finally we partition the original image into an array of sub-images, using this grid. Each sub-image is resampled from the original image using `scipy.ndimage.map_coordinates`,
which has options for high order spline interpolation.

 
## TODO

- The method currently assumes that the array of wells is aligned with the image axes, future work could relax this assumption by implementing a rotation finding step, perhaps optimising the entropy of the histogram?
- The QR decomposition used in the linear least squares sub-problem could be replaced by an analytic solution, but the runtime is currently bottlenecked by the resampling so there's probably no need.
