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
    plt.figure(figsize=figsize)
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
    return im.rotate(theta * 180 / np.pi, resample=PIL.Image.BICUBIC, fillcolor=255)


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


def find_skew_angle(a, pixel_used=30):
    """
    Trouve l'angle de rotation de l'image en fittant une droite sur les points les plus lumineux
    de la transformée de fourier
    :param a: 2D array
    :param pixel_used: int nombre de pixels utilisés pour fitter la droite
    :return:
    """
    A = np.fft.fftshift(np.fft.fft2(a))
    coef = []
    for quad in quadrants(np.log(np.abs(A) + 1) * middle_square(A.shape, 20)):
        X, y = find_brightest_pixels(quad, pixel_used)
        reg = LinearRegression().fit(X.reshape(-1, 1), y)
        R2 = reg.score(X.reshape(-1, 1), y)
        if R2 > 0.8:
            coef.append(reg.coef_)
    if len(coef) == 0:
        warnings.warn("Peu de confiance en l'angle de rotation")
        for quad in quadrants(np.log(np.abs(A) + 1) * middle_square(A.shape, 20)):
            X, y = find_brightest_pixels(quad, pixel_used)
            reg = LinearRegression().fit(X.reshape(-1, 1), y)
            coef.append(reg.coef_)

    correct = A.shape[1] / A.shape[0]  # distortion de l'angle si l'image n'est pas carrée

    return np.arctan(np.mean(coef) * correct)


def cut_lines(im, s=0.95):
    """
    Crée un générateur qui renvoie dans l'ordre les indice de départ et fin des lignes de texte dans l'image.

    :param im: 2D array
    :param s: float dans (0, 1) le seuil en dessous duquel on considère qu'il y a de l'écriture dans la ligne
    :return: generator
    """
    white_density = im.sum(axis=-1)
    threshold = s * white_density.max()

    text_line = white_density < threshold
    init = False
    for i in range(len(white_density)):
        if init:
            if text_line[i]:
                end = i
            else:
                init = False
                yield start, end + 1
        else:
            if text_line[i]:
                start = i
                end = i
                init = True


def cut_words(line, s=0.98):
    """
    Crée un générateur qui renvoie dans l'ordre les indice de départ et fin des mots de texte dans la ligne.

    :param line: 2D array
    :param s: float dans (0, 1) le seuil en dessous duquel on considère qu'il y a de l'écriture dans la ligne
    :return: generator
    """
    white_density = line.sum(axis=0)
    threshold = s * white_density.max()

    text_line = white_density < threshold
    init = False
    for i in range(len(white_density)):
        if init:
            if text_line[i]:
                end = i
            else:
                init = False
                yield start, end + 1
        else:
            if text_line[i]:
                start = i
                end = i
                init = True


def colors_generator():
    """
    Génère un cycle de couleurs
    :return: generator
    """
    color_bank = [(251, 180, 174, 140),
                  (179, 205, 227, 140),
                  (204, 235, 197, 140),
                  (222, 203, 228, 140),
                  (254, 217, 166, 140),
                  (255, 255, 204, 140),
                  (229, 216, 189, 140),
                  (253, 218, 236, 140)]
    n = 0
    while True:
        yield color_bank[n % 8]
        n += 1


def segmentation(image, crop_image=False, crop_size=None, unskew=True):
    """
    Effectue la segmentation de l'image et renvoie une nouvelle image entourant les mots
    dans des rectangles de différentes couleurs

    :param image: PIL.Image.Image
    :param crop_image: bool s'il faut découper l'image
    :param crop_size: tuple dimension du découpage
    :param unskew: bool s'il faut renvoyer une image remise droite
    :return:
    """

    if crop_image:
        if isinstance(crop_size, tuple):
            img = crop(image, crop_size)
        else:
            raise ValueError("Il faut indiquer une taille de découpage")
    else:
        img = image.copy()

    theta = find_skew_angle(np.array(img))

    img2 = rotation_PIL(img, -theta)
    matrix2 = np.array(img2)

    res = img2.convert('RGBA')  # passage en couleur avec transparence
    mask = PIL.Image.new('RGBA', res.size, color=(255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(mask)
    colors = colors_generator()
    for u, v in cut_lines(matrix2):
        for k, l in cut_words(matrix2[u:v]):
            draw.rectangle(((k, u), (l, v)), fill=next(iter(colors)))

    res.alpha_composite(mask)
    if unskew:
        return res
    else:
        return rotation_PIL(res, theta)
