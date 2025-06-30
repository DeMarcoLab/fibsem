import logging
from typing import List, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit, leastsq

# migrated from tdct.beadPos and refactored

### GAUSSIAN FITTING ###
def get_z_gauss(image: np.ndarray, x: int, y: int, show: bool = False) -> Tuple[float, int, float]:
    """Get the best fitting z-value for a 2D point in an ZYX image using a 1D Gaussian fit:
    Args:
        x: x coordinate
        y: y coordinate
        image: 3D numpy array (Z,Y,X)
        show: show the plot of the fit (for debugging)
    Returns:
        z: z coordinate (index)
    """

    # check that img is ndim=3
    if image.ndim != 3:
        raise ValueError(f"img must be a ZYX array, got {image.ndim}")

    # check that x, y are inside the image shape
    if x >= image.shape[-1] or y >= image.shape[-2]:
        raise ValueError(
            f"x and y must be within the image shape, x: {x}, y: {y}, {image.shape}"
        )

    # NOTE: round is important, don't directly cast to int
    if not isinstance(x, int):
        x = round(x)
    if not isinstance(y, int):
        y = round(y)

    # fit the z data for the given x,y
    poptZ, pcov = fit_guass1d(image[:, y, x], show=show)

    return np.array(poptZ)  # zval, zidx, zsigma


def fit_guass1d(data: np.ndarray, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 1D Gaussian to the data
    Args:
        data: 1D numpy array
        show: show the plot of the fit (for debugging)
    Returns:
        popt: optimal parameters
        pcov: covariance matrix
    """

    data = data - data.min()  # shift data to 0
    p0 = [data.max(), data.argmax(), 1]  # initial guess
    x = np.arange(len(data))  # x values
    popt, pcov = curve_fit(gauss1d, x, data, p0=p0)

    # plot the data and the fit
    if show:
        import matplotlib.pyplot as plt
        plt.title("1D Gaussian fit")
        plt.plot(data, label="Data")
        plt.plot(gauss1d(x, *popt), label="Gaussian 1D fit")
        plt.legend()
        plt.show()

    return popt, pcov

def fit_gauss1d_mod(data: np.ndarray, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 1D Gaussian to the data. Modified for negative hole images
    Args:
        data: 1D numpy array
        show: show the plot of the fit (for debugging)
    Returns:
        popt: optimal parameters
        pcov: covariance matrix
    """

    z=data
    xz=np.arange(z.shape[0])
    offset=np.max(z)
    z-=offset # shift data to 0

    p0=[np.min(z)-offset,np.argmin(z),5,offset]

    popt, pcov = curve_fit(gauss1d_offset, xz, z, p0, maxfev=10000)

    # plot the data and the fit
    if show:
        import matplotlib.pyplot as plt
        plt.title("1D Gaussian fit")
        plt.plot(data, label="Data")
        plt.plot(gauss1d(x, *popt), label="Gaussian 1D fit")
        plt.legend()
        plt.show()

    return popt, pcov

def gauss1d_offset(x, a, x0, sigma, offset):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

def gauss1d(x: np.ndarray, A: float, mu: float, sigma: float) -> float:
    """Gaussian 1D fit
    Args:
        x: x values
        A: magnitude
        mu: offset on x axis
        sigma: width
    Returns:
        y: gaussian values
    """
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

##### 2D GAUSSIAN FIT #####

def gauss2d_offset(coords, a, x0, y0, sigma_x, sigma_y, offset):
    x, y = coords  # unpack the coordinates
    return a * np.exp(-(((x - x0)**2) / (2 * sigma_x**2) + ((y - y0)**2) / (2 * sigma_y**2))) + offset

def fit_gauss_2d_mod(slc: np.ndarray, show: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    y_indices, x_indices = np.indices(slc.shape)
    # Flatten the coordinate arrays and the slice data for fitting.
    x_data = x_indices.ravel()
    y_data = y_indices.ravel()
    slc_data = slc.ravel()
    
    # define initial guess
    p0 = [float(np.min(slc)) - float(np.max(slc)), slc.shape[1] / 2, slc.shape[0] / 2, 1, 1, np.max(slc)]

    popt, pcov = curve_fit(gauss2d_offset, (x_data, y_data), slc_data, p0=p0, maxfev=10000)
    return popt, pcov

## Gaussian 2D fit from http://scipy.github.io/old-wiki/pages/Cookbook/FittingData
def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a Gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the Gaussian parameters of a 2D distribution by calculating its
    moments"""
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Returns (height, x, y, width_x, width_y)
    the Gaussian parameters of a 2D distribution found by a fit"""

    def errorfunction(p):
        return np.ravel(gaussian(*p)(*np.indices(data.shape)) - data)

    try:
        params = moments(data)
    except ValueError:
        return None
    p, success = leastsq(errorfunction, params)
    if np.isnan(p).any():
        return None

    return p

def extract_image_patch(img: np.ndarray, x: int, y:int, z: int, cutout: int) -> np.ndarray:
    # Get image dimensions
    z_max, height, width = img.shape
    
    # Calculate patch bounds
    x_min = int(x - cutout)
    x_max = int(x + cutout)
    y_min = int(y - cutout)
    y_max = int(y + cutout)
    z = int(round(z))

    # Check if patch is within bounds
    is_valid = (
        x_min >= cutout and
        x_max < width and
        y_min >= cutout and
        y_max < height and
        0 <= z < z_max
    )

    if not is_valid:
        logging.warning("Point(s) too close to edge or out of bounds.")
        return None

    # Extract and return the patch
    return np.copy(img[z, y_min:y_max, x_min:x_max])

def threshold_image(data: np.ndarray, threshold_val: float):
    """
    Zero out values below a threshold relative to the data range.
    
    Args:
        data: numpy array of image data
        threshold_percent: float between 0-1, normalized threshold value
    """
    data_min = data.min()
    data_max = data.max()
    data_range = data_max - data_min
    
    # Calculate threshold value
    threshold_value = data_max - (data_range * threshold_val)
    
    # Zero out values below threshold 
    data[data < threshold_value] = 0
    
    return data

#### INTERPOLATION ####

INTERPOLATION_METHODS = ["linear", "cubic"]

def interpolate_z_stack(
    image: np.ndarray, pixelsize_in: float, pixelsize_out: float, method: str = "linear"
) -> np.ndarray:
    """Interpolate a 3D image array along the z-axis using scipy's zoom function."""

    # check for multi-channel images, only single channel images are supported currently
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                f"Muli-channel images are not supported, got an image of shape {image.shape}, expected (1, Z, Y, X) or (Z, Y, X)"
            )

        # remove the channel dimension
        logging.info(f"Squeezing channel dimension from image of shape {image.shape}")
        image = np.squeeze(image, axis=0)

    if image.ndim != 3:
        raise ValueError(f"image must be a ZYX array, but got {image.ndim}")

    if method not in INTERPOLATION_METHODS:
        raise ValueError(
            f"interpolation method  must be in {INTERPOLATION_METHODS} got {method}"
        )

    # interpolate the image
    return scipy_interpolation(image_3d=image, original_z_size=pixelsize_in, target_z_size=pixelsize_out, method=method)

def scipy_interpolation(
    image_3d: np.ndarray, original_z_size: float, target_z_size: float, method: str = "linear"
) -> np.ndarray:
    """
    Fast interpolation of a 3D image array along the z-axis using scipy's zoom function.

    Parameters:
    -----------
    image_3d : ndarray
        Input 3D image array with shape (Z, Y, X)
    original_z_size : float
        Original pixel size in z-axis (e.g., 10 for 10µm)
    target_z_size : float
        Desired pixel size in z-axis (e.g., 5 for 5µm)
    method : str
        Interpolation method ('linear' or 'cubic')

    Returns:
    --------
    ndarray
        Interpolated 3D image with adjusted z-axis resolution
    """
    # Calculate the scaling factor
    scale_factor = original_z_size / target_z_size

    # Create zoom factors for each dimension
    # Only scale the z-axis (first dimension)
    zoom_factors = (scale_factor, 1, 1)

    # Determine the interpolation order
    if method not in INTERPOLATION_METHODS:
        method = "linear"
    order = 1 if method == "linear" else 3

    # Perform the interpolation using scipy's zoom function
    # mode='reflect' to handle edge cases
    # prefilter=True for better quality
    interpolated = ndimage.zoom(
        image_3d, 
        zoom_factors, 
        order=order, 
        mode="reflect", 
        prefilter=True
    )

    return interpolated


#### multi-channel interpolation ####

def multi_channel_interpolation(
    image: np.ndarray,
    pixelsize_in: float,
    pixelsize_out: float,
    method: str = "cubic",
    parent_ui=None,
) -> np.ndarray:
    """Interpolate a multi-channel z-stack (CZYX) along the z-axis
    Args:
        image: 4D numpy array (CZYX)
        pixelsize_in: original pixel size in z-axis
        pixelsize_out: desired pixel size in z-axis
    Returns:
        interpolated: 4D numpy array (CZYX) with adjusted z-axis resolution
    """
    if parent_ui:
        parent_ui.progress_update.emit({"value": 0, "max": image.shape[0]})

    # QUERY: how to speed up?
    ch_interpolated = []
    for i, channel in enumerate(image):
        logging.info(f"Interpolating channel {i+1}/{image.shape[0]}")
        ch_interpolated.append(
            interpolate_z_stack(
                image=channel,
                pixelsize_in=pixelsize_in,
                pixelsize_out=pixelsize_out,
                method=method,
            )
        )
        if parent_ui:
            parent_ui.progress_update.emit({"value": i + 1, "max": image.shape[0]})
    return np.array(ch_interpolated)


def multi_channel_get_z_guass(image: np.ndarray, x: int, y: int, show: bool = False) -> List[float]:
    """Get the best fitting z-value for a 2D point in an multi-channel ZYX image using a 1D Gaussian fit:
    Args:
        x: x coordinate
        y: y coordinate
        image: 4D numpy array (CZYX)
        show: show the plot of the fit (for debugging)
    Returns:
        z: z coordinate (index)
    """
    # shortcut for single channel images
    if image.ndim == 3:
        return get_z_gauss(image, x, y, show=show)

    z_values = []
    for channel in image:
        try:
            # optimisation can fail, fallback to nothing
            zvals = get_z_gauss(channel, x, y, show=show)
        except Exception as e:
            logging.warning(f"Error in channel: {e}")
            zvals = [0, 0, 0]
        z_values.append(zvals)

    # get the channel with the maximum z-value (zval, zidx, zsigma)
    vals = np.array(z_values)
    ch_idx = np.argmax(vals[:, 0])

    return vals[ch_idx] # zval, zidx, zsigma

def hole_fitting_RL(img: np.ndarray,
    x: int,
    y: int,
    z: int,
    cutout: int = 11,
    small_cutout: int = 5,
    apply_threshold: bool = False,
    threshold_val: float = 0,
    iterations: int = 5,
    show: bool = False,
):
    import matplotlib.pyplot as plt
    """refine selection of hole in reflected light image
    Args:
        img: 3D numpy array (Z,Y,X), interpolated to isotropic pixel size
        x,y,z initial coordinates from the user click
        cutout: size of the cutout around the point in x,y. z uses 3x this value
        small_cutout: size of the cutout for the refined fit. z uses 3x this value
        apply_threshold: apply thresholding to the image
        threshold_val: does nothing
        iterations: does nothing
    Returns:
        xr, yr, zr: refined x, y, z coordinates
    """
    # cut out the box
    ROI=img[z-cutout*3:z+cutout*3,y-cutout:y+cutout,x-cutout:x+cutout]
    # # fit a gaussian to estimate the in focus plane
    # I=np.mean(ROI,axis=(1,2))
    # popt,popcov=fit_gauss1d_mod(I,show=False)
    # zi=int(popt[1])+z-cutout*3
    # if show:
    #     plt.figure()
    #     plt.plot(I)
    #     plt.scatter(popt[1],popt[2],color='r')
    #     plt.show()
    
    # fit a 2D gaussian to the in focus plane to get rough poition
    # slc=img[zi,y-cutout:y+cutout,x-cutout:x+cutout]
    slc=img[z,y-cutout:y+cutout,x-cutout:x+cutout]
    popt,popcov=fit_gauss_2d_mod(slc,show=False)
    # get the rough position in the coordinates of the original image
    xi=int(popt[1])+x-cutout
    yi=int(popt[2])+y-cutout
    if show:
        plt.figure()
        plt.imshow(slc)
        plt.scatter(popt[1],popt[2],color='r')
        plt.show()
    zi=z
    # refine the estimates
    # cut out a smaller box
    ROI=img[zi-small_cutout*3:zi+small_cutout*3,yi-small_cutout:yi+small_cutout,xi-small_cutout:xi+small_cutout]

    # fit a gaussian to estimate the in focus plane
    I=np.mean(ROI,axis=(1,2))
    popt,popcov=fit_gauss1d_mod(I,show=False)
    zr=popt[1]+zi-small_cutout*3
    if show:
        plt.figure()
        plt.plot(I)
        plt.axvline(popt[1],color='r')
        plt.show()

    # fit a 2D gaussian to the in focus plane to get rough poition
    slc=img[int(zr),yi-small_cutout:yi+small_cutout,xi-small_cutout:xi+small_cutout]
    popt,popcov=fit_gauss_2d_mod(slc,show=False)
    # get the refined positions in the coordinates of the original image
    xr=popt[1]+xi-small_cutout
    yr=popt[2]+yi-small_cutout

    if show:
        plt.figure()
        plt.imshow(slc)
        plt.scatter(popt[1],popt[2],color='r')
        plt.show()

    return xr,yr,zr

def hole_fitting_FIB(img: np.ndarray,
    x: int,
    y: int,
    cutout: int = 15,
    show: bool = False,
):
    import matplotlib.pyplot as plt
    """refine selection of hole in FIB image
    Args:
        img: 2D numpy array (X,Y)
        x,y initial coordinates from the user click
        cutout: size of the cutout around the point in x,y
    Returns:
        xr, yr: refined x, y coordinates
    """
    # cut out a box around the point
    ROI=img[y-cutout:y+cutout,x-cutout:x+cutout]
    # fit a 2D gaussian to estimate the hole position
    popt,popcov=fit_gauss_2d_mod(ROI,show=False)
    # get the refined positions in the coordinates of the original image
    xr=popt[1]+x-cutout
    yr=popt[2]+y-cutout

    if show:
        plt.figure()
        plt.imshow(ROI)
        plt.scatter(popt[1],popt[2],color='r')
        plt.show()
    
    return xr,yr

def zyx_targeting(
    img: np.ndarray,
    x: int,
    y: int,
    cutout: int = 15,
    apply_threshold: bool = False,
    threshold_val: float = 0.1,
    iterations: int = 5,
):
    """Automated ZYX targeting for a 3D image stack. Optimizes the selection of x,y,z coordinates
    for a given 2D point in a 3D image stack using an iterative gaussian fitting approach.
    Args:

        img: 3D numpy array (Z,Y,X)
        x: initial x coordinate
        y: initial y coordinate
        cutout: size of the cutout around the point
        apply_threshold: apply thresholding to the image
        threshold_val: threshold value for thresholding
        iterations: number of iterations
    Returns:
        x, y, (zval, zidx, zsigma): optimized x, y, z coordinates and z-value (max, index, sigma)
    """
    logging.info(f"Starting zyx targeting for {img.shape}, Initial x: {x}, y: {y}")

    # get the initial z position (Note: this can fail, returns None)
    zval, zidx, zsigma = multi_channel_get_z_guass(image=img, x=x, y=y)

    assert img.ndim == 3, "Image must be 3D"

    logging.info(f"Initial z: {zidx}, zval: {zval}, zsigma: {zsigma} for {x}, {y}")

    for i in range(iterations):
        data = extract_image_patch(img, x, y, zidx, cutout)
        if data is None:
            break  # Or handle error case differently

        if apply_threshold:  # apply threshold on normalized data
            data = threshold_image(data, threshold_val)

        # fit a 2d guassian to the 3d cutout
        poptXY = fitgaussian(data)
        if poptXY is None:
            break

        (height, xopt, yopt, width_x, width_y) = poptXY

        # x and y are switched when applying the offset
        x = x - cutout + yopt
        y = y - cutout + xopt
        width, height = img.shape[-1], img.shape[-2]
        if not (0 <= x < width and 0 <= y < height):
            break

        # fit a 1d guassian to the z stack
        zval, zidx, zsigma = get_z_gauss(img, x=x, y=y)
        logging.debug(
            f"iteration: {i}, x: {x}, y: {y}, z: {zidx}, zval: {zval}, zsigma: {zsigma}"
        )

    # TODO: check that xyz are in image bounds, if not return original x, y, z

    return x, y, (zval, zidx, zsigma)

def multi_channel_zyx_targeting(
    image: np.ndarray,
    xinit: int,
    yinit: int,
    zinit: int,
    apply_threshold: bool = False,
    threshold_val: float = 0.1,
    cutout: int = 15,
    iterations: int = 5,
) -> Tuple[int, Tuple[int, int, int]]:
    """ZYX targeting for multi-channel images
    Args:
        image: 4D numpy array (CZYX)
        xinit: initial x coordinate
        yinit: initial y coordinate
        apply_threshold: apply thresholding to the image
        threshold_val: threshold value for thresholding
        cutout: size of the cutout around the point
        iterations: number of iterations
    Returns:
        ch_idx: channel index with the best z-value
        x, y, z: x, y, z coordinates of the best z-value in the best channel
    """

    # shortcut for single channel images
    if image.ndim == 3:
        x1, y1, (zv, z1, zs) = zyx_targeting(
            image,
            xinit,
            yinit,
            cutout=cutout,
            apply_threshold=apply_threshold,
            threshold_val=threshold_val,
            iterations=iterations,
        )
        return 0, (x1, y1, z1)

    if image.ndim != 4:
        raise ValueError(f"image must be a 4D array (CZYX), got {image.ndim}")

    zvalues = []
    xyz_vals = []

    for i in range(image.shape[0]):
        ch_image = image[i]
        try:
            # x1, y1, (zv, z1, zs) = zyx_targeting(
            #     ch_image,
            #     xinit,
            #     yinit,
            #     cutout=cutout,
            #     apply_threshold=apply_threshold,
            #     threshold_val=threshold_val,
            #     iterations=iterations,
            # )
            x1, y1, zv = hole_fitting_RL(
                ch_image,
                xinit,
                yinit,
                zinit,
                cutout=cutout,
                apply_threshold=apply_threshold,
                threshold_val=threshold_val,
                iterations=iterations,
            )
        except Exception as e:
            logging.error(f"an error occured during channel {i}: {e}")
            x1, y1, zv, z1, zs = xinit, yinit, 0, None, None
        # zvalues.append((zv, z1, zs))
        # xyz_vals.append((x1, y1, z1))
        break

    return 0, (x1, y1, zv)

    # vals = np.array(zvalues).astype(np.float32)
    # ch_idx = np.argmax(vals[:, 0])

    # logging.info(f"solution found: Channel Index: {ch_idx}: xyz: {xyz_vals[ch_idx]}")
    # return ch_idx, xyz_vals[ch_idx]

def apply_refractive_index_correction(
    initial_poi: Tuple[float, float],
    surface_coord: Tuple[float, float],
    correction_factor: float,
) -> Tuple[float, float]:
    """Apply a refractive index correction to the point of interest (POI) coordinates.
    Initial point of interest and surface coordinate are both in image pixels
    Args:
        initial_poi: initial point of interest coordinates (x, y)
        surface_coord: surface coordinate (x, y)
        correction_factor: correction factor for the refractive index
    Returns:
        corrected_poi: corrected point of interest coordinates (x, y)"""

    # apply correction factor to poi
    depth = initial_poi[1] - surface_coord[1]  # assume poi always below surface, y-axis

    corrected_depth = depth * correction_factor
    logging.info(
        f"Correction Factor: {correction_factor}, Depth: {depth}, Corrected Depth: {corrected_depth}"
    )

    # update the poi coordinate in poi
    corrected_poi = (initial_poi[0], surface_coord[1] + corrected_depth)
    return corrected_poi