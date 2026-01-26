import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from matplotlib.colors import LinearSegmentedColormap

# RGB colors
light_colors = {
    700: (255, 0, 0),
    610: (255, 127, 0),
    580: (255, 255, 0),
    530: (0, 255, 0),
    500: (0, 255, 255),
    470: (0, 0, 255),
    420: (139, 0, 255),
}


def tf(arr):
    """
    Compute the 2D Fourier Transform of a 2D array with centering.
    
    Parameters:
    -----------
    arr : 2D numpy array
        Input array (e.g., pupil function)
    
    Returns:
    --------
    2D numpy array
        Shifted 2D Fourier transform of the input array
    """
    # fftshift centers the zero frequency in the middle of the array
    # Apply FFT to compute Fourier transform
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(arr)))


def pad(arr, pad_width):
    """
    Zero-pad a square 2D array by a given factor.
    
    Parameters:
    -----------
    arr : 2D numpy array
        Input array to pad (e.g., pupil)

    pad_width : int
        Number of pixels added on each side of the array - same padding applied on all sides.

    Returns:
    --------
    2D numpy array
        Zero-padded array
    """
    # np.pad adds zeros around the array (top, bottom, left, right)
    padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
    return padded


def make_pupil(Rpx, nPx):
    """
    Generate a 2D circular pupil function.
    
    Parameters:
    -----------
    Rpx : float
        Radius of the pupil in pixels
    nPx : int
        Size of the output array (nPx x nPx)
    
    Returns:
    --------
    2D numpy array
        Circular pupil with values 1 inside the radius and 0 outside
    """
    # Create a grid of coordinates (row: y, column: x)
    y, x = np.ogrid[:nPx, :nPx]
    
    # Center coordinates of the pupil
    cy = cx = (nPx - 1) / 2  
    
    # Circular mask: set to 1 if inside the radius, 0 outside
    pupil = ((x - cx)**2 + (y - cy)**2 <= Rpx**2).astype(float)
    
    return pupil

def spectral_to_rgb_image(data, light_colors):
    """
    Combine multispectral intensity maps into a single RGB image.

    Parameters
    ----------
    data : dict
        {wavelength_nm: NxN numpy array}
        Intensity maps (can be float or int, any positive scale).
    
    light_colors : dict
        {wavelength_nm: (R, G, B)}
        RGB values in range [0, 255].

    Returns
    -------
    image_rgb : NxNx3 uint8 numpy array
        RGB image ready for display or saving.
    """

    # Get image size from first entry
    first_key = next(iter(data))
    N, M = data[first_key].shape

    # Initialize RGB image (float for accumulation)
    image = np.zeros((N, M, 3), dtype=np.float32)

    # Combine spectral layers
    for wavelength, intensity in data.items():
        if wavelength not in light_colors:
            continue  # skip unknown wavelengths

        r, g, b = light_colors[wavelength]

        image[..., 0] += np.log1p(intensity * r)
        image[..., 1] += np.log1p(intensity * g)
        image[..., 2] += np.log1p(intensity * b)

    # Normalize each channel separately to avoid green dominance
    for c in range(3):
        max_val = image[..., c].max()
        if max_val > 0:
            image[..., c] /= max_val

    # Convert to 8-bit RGB
    image = (image * 255).astype(np.uint8)

    return image

def show_rgb(rgb_image: np.ndarray, title: str = "RGB Image"):
    """
    Display an RGB image using Matplotlib.

    Parameters
    ----------
    rgb_image : numpy.ndarray
        A 3D NumPy array of shape (height, width, 3), representing an RGB image.
        The array should have dtype uint8 or float in [0, 1].
    title : str, optional
        Title of the displayed figure. Default is "RGB Image".

    Returns
    -------
    None
        Displays the image in a Matplotlib figure.
    
    Notes
    -----
    This is convenient for inline display in notebooks. Unlike PIL,
    Matplotlib allows adding titles, axes labels, and saving figures easily.
    """
    # Convert float images in [0,1] to [0,255] if needed
    if rgb_image.dtype != np.uint8:
        rgb_image = np.clip(rgb_image, 0, 1)
    
    # Min-Max normalization for display
    vmin = rgb_image.min()
    vmax = rgb_image.max()
    rgb_image = np.clip(rgb_image, vmin, vmax)
    rgb_image = (rgb_image - vmin) / (vmax - vmin)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.axis("off")  # hide axes
    plt.title(title)
    plt.show(block=False)

def generate_power_law_phase_screen(N, alpha):
    """
    Generate a 2D random phase screen with PSD ~ f^(-alpha)
    
    Parameters:
        N (int): Size of the screen (NxN)
        alpha (float): Power-law exponent
        
    Returns:
        phase_screen (2D array): Real-space 2D random field
    """
    # Frequency grid
    kx = np.fft.fftfreq(N)
    ky = np.fft.fftfreq(N)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
    k[N//2, N//2] = 1e-10  # avoid div by zero at DC

    amplitude = k**(-alpha/2)
    phase = np.random.rand(N, N)

    F_field = amplitude * phase

    field = np.fft.ifft2(np.fft.ifftshift(F_field)).imag  # shift back before ifft
    field = (field - np.mean(field)) / np.std(field)
        
    return field

def show_spectral_psf(psfs, wvl, vmin=None):
    """
    Display a monochromatic PSF using a wavelength-dependent colormap.

    Parameters
    ----------
    psfs : dict
        {wavelength_nm: 2D numpy array}
        Spectral PSF data.
    wvl : int
        Wavelength (nm) to display.
    vmin : float, optional
        Minimum value for display (e.g. to suppress background).
    """
    # Convert RGB color from light_colors to [0,1]
    rgb = np.array(light_colors[wvl]) / 255

    # Create colormap: black -> spectral color
    cmap = LinearSegmentedColormap.from_list(
        f"black_to_{wvl}nm",
        [(0, 0, 0), rgb]
    )

    plt.figure()
    plt.imshow(psfs[wvl]**0.2, cmap=cmap, vmin=vmin)
    plt.title(f"PSF at {wvl} nm")
    plt.axis("off")
    plt.show()