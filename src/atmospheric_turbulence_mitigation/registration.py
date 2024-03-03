import numpy as np
from skimage.transform import warp
from skimage.registration import optical_flow_tvl1
import multiprocessing as mp


def registration_worker(args):
    image, reference = args

    # --- Compute optical flow
    v, u = optical_flow_tvl1(reference, image)

    # --- Use the estimated optical flow for registration
    nr, nc = image.shape
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
    reference_warp = warp(image, np.array([row_coords + v, col_coords + u]), mode='edge')

    return reference_warp


def stabilize(
        images,
        iterations: int = 1,
        reference_filter: callable = lambda seq: np.mean(seq, axis=0)
) -> np.ndarray:
    """
    Turbulence mitigation from sequence of still target

    :param images: the sequence of images of a still target
    :param iterations: the number of Bregman iterations
    :param reference_filter: function to be used before each iteration to combine sequenced images
    :returns: a numpy array of the resulting image after each iteration. (arr[-1] best)
    """
    if iterations < 1:
        raise ValueError("Number of iterations must be positive.")

    reference = reference_filter(images)
    sequence = np.copy(images)
    each_iter = []
    with mp.Pool(mp.cpu_count()) as pool:
        for _ in range(iterations):
            # Delegate
            sequence = pool.map(registration_worker, [(im, reference) for im in sequence])
            reference = reference_filter(sequence)
            each_iter.append(reference)
    return np.array(each_iter)
