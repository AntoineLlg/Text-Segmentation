import numpy as np
from tqdm import tqdm
import PIL.Image
import utils
from sklearn.linear_model import LinearRegression


def mask_orientation(theta, shape, eps=np.pi/32):
    """
    creates a mask in a cone around the direction theta
    """

    x, y = np.linspace(-1, 1, shape[0]).reshape(-1, 1).dot(np.ones(shape[1]).reshape(1, -1)), np.ones(shape[0]).reshape(
        -1, 1).dot(np.linspace(-1, 1, shape[1]).reshape(1, -1))

    complexes = y - 1j * x
    angles = np.angle(complexes)
    angles += 2*np.pi * (angles + np.pi < eps)
    mask = (angles > theta - eps) * (angles < theta + eps) * (np.abs(complexes) > .25)
    return mask


def angle_strength(theta, im):
    mask = mask_orientation(theta, im.shape)
    return ((mask * im) ** 2).sum()


def fft_shifted_abs(im):
    return np.abs(np.fft.fftshift(np.fft.fft2(im)))


def find_skew_angle_lin(im):
    numpy_image = np.asarray(im)
    TF = np.log(np.abs(np.fft.fftshift(np.fft.fft2(numpy_image))))
    n, m = TF.shape
    X, y = utils.find_brightest_pixels(TF * utils.middle_square(TF.shape, n // 16, m // 16), 20)

    reg = LinearRegression().fit(X.reshape(-1, 1) / TF.shape[0], y / TF.shape[1])

    alpha = np.arctan(reg.coef_)
    return alpha


def find_skew_angle_masks(im, n_theta=100):
    thetas = np.linspace(-np.pi/32, 33*np.pi/32, n_theta)

    L = []
    for theta in tqdm(thetas, desc='Looking for the angle'):
        L.append(angle_strength(theta, im))

    max_index = np.argmax(L)

    vois_theta = thetas[max(0, max_index - 2):min(max_index + 3, n_theta)]  # voisinage de theta
    weights = L[max(0, max_index - 2):min(max_index + 3, n_theta)]
    return np.average(vois_theta, weights=weights)


def rotation_PIL(im, theta):
    """
    Effectue la rotation de theta radiants de l'image en prenant le centre de l'image comme centre de rotation
    :param im: PIL.Image.Image image
    :param theta: float angle de rotation en radiants

    :return: PIL.Image.Image
    """
    return im.copy().rotate(theta * 180 / np.pi, resample=PIL.Image.BICUBIC, fillcolor=255, expand=True)
