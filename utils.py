import numpy as np
import matplotlib.pyplot as plt
import PIL
from sklearn.linear_model import LinearRegression
import warnings


def view_image(im, figsize=(10, 10)):
    """
    Permet de lire une image en niveaux de gris

    :param im: np.array image
    :param figsize: (float, float) taille de l'image affichée
    :return:
    """
    plt.imshow(im, 'Greys_r')
    plt.axis('off')
    plt.show()


def imread(file_name):
    """
    Ouvre l'image en la convertissant en niveaux de gris. Renvoie une instance de PIL.Image.Image
    :param file_name:
    :return img: PIL.Image.Image
    """
    img = PIL.Image.open(file_name).convert('L')  # conversion en niveaux de gris
    return img


def rotation_PIL(im, theta):
    """
    Effectue la rotation de theta radiants de l'image en prenant le centre de l'image comme centre de rotation
    :param im: PIL.Image.Image image
    :param theta: float angle de rotation en radiants

    :return: PIL.Image.Image
    """
    return im.rotate(theta * 180 / np.pi, resample=PIL.Image.BICUBIC)


def crop(im, size):
    """
    découpe un carré ou rectangle au centre de l'image, de dimension(s) size. Si size est un entier, il est utilisé
    à la fois pour la longueur et la largeur.

    :param im: PIL.Image.Image
    :param size: (int, int) or int
    :return:
    """
    M, N = im.size[0] // 2, im.size // 2
    if isinstance(size, tuple):
        L, K = size[0] // 2, size[1] // 2
        return im.crop((N - L, M - K, N + L, M + K))
    else:
        L = size // 2
        return im.crop((N - L, M - L, N + L, M + L))


def quadrants(im):
    """
    Divise l'image en quadrants
    :param im: 2D array image
    :return:
    """
    N, M = im.shape
    yield im[:N // 2, :M // 2]
    yield im[:N // 2, M // 2:]
    yield im[N // 2:, :M // 2]
    yield im[N // 2:, M // 2:]


def middle_square(shape, n, m=None):
    """
    Crée une matrice de uns à l'exception d'un rectangle de zéros de taille 2n * 2m au milieu recouvrant également
    chaque quadrant

    :param shape: (int, int) dimension de la matrice renvoyée
    :param n: int hauteur du rectangle. Si m n'est pas utilisé, n est également la largeur.
    :param m: (optionnel) largeur du rectangle
    :return: 2D array
    """
    if not m:
        m = n
    N, M = shape
    res = np.ones(shape) * 255
    res[N // 2 - n:N // 2 + n, M // 2 - m:M // 2 + m] = 0
    return res


def find_brightest_pixels(image, N):
    """
    trouve les positions des N pixels les plus brillants de l'image
    :param image: 2D array
    :param N: int
    :return:
    """
    image_1d = image.flatten()
    idx_1d = image_1d.argsort()[-N:]
    x_idx, y_idx = np.unravel_index(idx_1d, image.shape)
    return x_idx, y_idx


def find_skew_angle(im):
    IM = np.fft.fftshift(np.fft.fft2(im))
    coef = []
    for quad in quadrants(np.log(np.abs(IM) + 1) * middle_square(IM.shape, 20)):
        X, y = find_brightest_pixels(quad, 30)
        reg = LinearRegression().fit(X.reshape(-1, 1), y)
        R2 = reg.score(X.reshape(-1, 1), y)
        if R2 > 0.8:
            coef.append(reg.coef_)
    if len(coef) == 0:
        warnings.warn("Peu de confiance en l'angle de rotation")
        for quad in quadrants(np.log(np.abs(IM) + 1) * middle_square(IM.shape, 20)):
            X, y = find_brightest_pixels(quad, 30)
            reg = LinearRegression().fit(X.reshape(-1, 1), y)
            coef.append(reg.coef_)

    return np.arctan(np.mean(coef))