import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import ImageDraw
from sklearn.linear_model import LinearRegression
import warnings
import turn
import analysis
from PIL.ImageFilter import GaussianBlur

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from scipy.stats import mode
from skimage.filters import threshold_local


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


def find_skew_angle_quadrants(a, pixel_used=30):
    """
    Trouve l'angle de rotation de l'image en fittant une droite sur les points les plus lumineux
    de la transformée de fourier
    :param a: 2D array
    :param pixel_used: int nombre de pixels utilisés pour fitter la droite
    :return:
    """
    A = np.fft.fftshift(np.fft.fft2(a))
    coef = []
    for quad in quadrants(np.log(np.abs(A) + 1) * middle_square(A.shape, 10)):
        X, y = find_brightest_pixels(quad, pixel_used)
        reg = LinearRegression().fit(X.reshape(-1, 1), y)
        R2 = reg.score(X.reshape(-1, 1), y)
        if R2 > 0.8:
            coef.append(reg.coef_)
    if len(coef) == 0:
        warnings.warn("Peu de confiance en l'angle de rotation")
        for quad in quadrants(np.log(np.abs(A) + 1) * middle_square(A.shape, 5)):
            X, y = find_brightest_pixels(quad, pixel_used)
            reg = LinearRegression().fit(X.reshape(-1, 1), y)
            coef.append(reg.coef_)

    correct = A.shape[1] / A.shape[0]  # distortion de l'angle si l'image n'est pas carrée

    return np.arctan(np.mean(coef) / correct)


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


def segmentation(image, r=1, sl=0.95, sw=0.98):
    """
    Effectue la segmentation de l'image et renvoie une nouvelle image entourant les mots
    dans des rectangles de différentes couleurs

    :param sw: seuil de segmentation des mots
    :param sl: seuil de segmentation des lignes
    :param r: rayon du noyau gaussien utilisé pour segmenter les mots
    :param image: PIL.Image.Image
    :return:
    """
    numpy_image = np.array(image)

    binary = otsu(numpy_image)
    binary_PIL = PIL.Image.fromarray(binary)

    blurred = np.array(binary_PIL.filter(GaussianBlur(radius=r)))

    res = image.copy().convert('RGBA')
    mask = PIL.Image.new('RGBA', res.size, color=(255, 255, 255, 0))
    draw = PIL.ImageDraw.Draw(mask)
    colors = colors_generator()

    n, m = binary.shape

    for u, v in analysis.cut_lines(binary, sl):
        for k, l in analysis.cut_words(blurred[u:v], sw):
            if u > 0 and v < n and k > 0 and l < m:
                draw.rectangle(((k, u), (l, v)), fill=next(iter(colors)))
    res.alpha_composite(mask)
    return res



def histogram(im):
    nl, nc = im.shape

    hist = np.zeros(256)

    for i in range(nl):
        for j in range(nc):
            hist[im[i][j]] = hist[im[i][j]] + 1

    for i in range(256):
        hist[i] = hist[i] / (nc * nl)

    return hist


def otsu(im):
    h = histogram(im)

    m = 0
    for i in range(256):
        m = m + i * h[i]

    maxt = 0
    maxk = 0

    for t in range(256):
        w0 = 0
        w1 = 0
        m0 = 0
        m1 = 0
        for i in range(t):
            w0 = w0 + h[i]
            m0 = m0 + i * h[i]
        if w0 > 0:
            m0 = m0 / w0

        for i in range(t, 256):
            w1 = w1 + h[i]
            m1 = m1 + i * h[i]
        if w1 > 0:
            m1 = m1 / w1

        k = w0 * w1 * (m0 - m1) * (m0 - m1)

        if k > maxk:
            maxk = k
            maxt = t

    thresh = maxt

    return ((im > thresh) * 255).astype(np.uint8)


def find_skew_angle_quadrants(a, pixel_used=30):
    """
    Trouve l'angle de rotation de l'image en fittant une droite sur les points les plus lumineux
    de la transformée de fourier
    :param a: 2D array
    :param pixel_used: int nombre de pixels utilisés pour fitter la droite
    :return:
    """
    A = np.fft.fftshift(np.fft.fft2(a))
    coef = []
    for quad in quadrants(np.log(np.abs(A) + 1) * middle_square(A.shape, 10)):
        X, y = find_brightest_pixels(quad, pixel_used)
        reg = LinearRegression().fit(X.reshape(-1, 1), y)
        R2 = reg.score(X.reshape(-1, 1), y)
        if R2 > 0.8:
            coef.append(reg.coef_)
    if len(coef) == 0:
        warnings.warn("Peu de confiance en l'angle de rotation")
        for quad in quadrants(np.log(np.abs(A) + 1) * middle_square(A.shape, 5)):
            X, y = find_brightest_pixels(quad, pixel_used)
            reg = LinearRegression().fit(X.reshape(-1, 1), y)
            coef.append(reg.coef_)

    correct = A.shape[1] / A.shape[0]  # distortion de l'angle si l'image n'est pas carrée

    return np.arctan(np.mean(coef) / correct)

def find_skew_angle_no_quadrants(a, pixel_used=30 , square = 10):
    """
    Trouve l'angle de rotation de l'image en fittant une droite sur les points les plus lumineux
    de la transformée de fourier, sans utiliser la méthode des quadrants
    :param a: 2D array
    :param pixel_used: int nombre de pixels utilisés pour fitter la droite
    :return:
    """
    A = np.fft.fftshift(np.fft.fft2(a))
    coef = 0
    X, y = find_brightest_pixels(np.log(np.abs(A) + 1) * middle_square(A.shape, square), pixel_used)
    reg = LinearRegression().fit(X.reshape(-1, 1), y)
    R2 = reg.score(X.reshape(-1, 1), y)
    if R2 > 0.8:
        coef = reg.coef_
    if coef == 0:
        warnings.warn("Peu de confiance en l'angle de rotation")
        X, y = find_brightest_pixels(np.log(np.abs(A) + 1) * middle_square(A.shape, square), pixel_used)
        reg = LinearRegression().fit(X.reshape(-1, 1), y)
        coef = reg.coef_

    correct = A.shape[1] / A.shape[0]  # distortion de l'angle si l'image n'est pas carrée

    return np.arctan(coef / correct)

def find_skew_angle_hough(im, plot = False):
    edges = canny(im)
    angles = np.deg2rad(np.arange(0.1, 180.0)) #radians
    h, theta, d = hough_line(edges, theta=angles)
    accum, angles, dists = hough_line_peaks(h, theta, d)

    #to plot the lines on the image
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 16))
        ax = axes.ravel()
        ax[0].imshow(im, cmap="gray")
        ax[0].set_title('Image')
        ax[0].set_axis_off()
        ax[1].imshow(edges, cmap="gray")
        origin = np.array((0, im.shape[1]))
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[1].plot(origin, (y0, y1), '-r')
        ax[1].set_xlim(origin)
        ax[1].set_ylim((edges.shape[0], 0))
        ax[1].set_axis_off()
        ax[1].set_title('Lignes de Hough')

    return mode(np.around(angles, decimals=2))[0]


def find_skew_angle(im, method = 'fourier', pixel_used = 30, square = 10, plot = False ):
    """
    Trouve l'angle de rotation de l'image du texte

    :param im: l'image pré-traitée
    :param method: la méthode à utiliser : fourier, quadrants, hough
    :param pixel_used: pour les méthodes utilisant fourier
    :param square: pour les méthodes utilisant fourier
    :param plot: pour tracer les lignes de Hough
    :return: angle
    """
    if method == 'fourier':
        out = find_skew_angle_no_quadrants(im, pixel_used, square)
    elif method == 'quadrants':
        out = find_skew_angle_quadrants(im, pixel_used)
    elif method == 'hough':
        out = find_skew_angle_hough(im, plot)
    return out

def ostu_local(im, size = 9, offset=0):
    local_thresh = threshold_local(im, size, offset)
    out = im > local_thresh
    return out


def median_filter(im, typ=1, r=1, xy=None):
    """
    renvoie le median de l'image im.
    si typ==1 (defaut) le median est calcule sur un carre de cote 2r+1
    si typ==2 : disque de rayon r
    si typ==3 alors xy est un couple de liste de x et liste de y
         ([-1,0,1] , [0,0,0]) donne un median sur un segment horizontql de taille trois.
    """
    lx = []
    ly = []
    (ty, tx) = im.shape
    if typ == 1:  # carre

        for k in range(-r, r + 1):
            for l in range(-r, r + 1):
                lx.append(k)
                ly.append(l)

    elif typ == 2:
        for k in range(-r, r + 1):
            for l in range(-r, r + 1):
                if k ** 2 + l ** 2 <= r ** 2:
                    lx.append(k)
                    ly.append(l)
    else:  #freeshape
        lx, ly = xy

    debx = -min(lx)  # min is supposed negatif
    deby = -min(ly)
    finx = tx - max(lx)  # max is supposed positif
    finy = ty - max(ly)
    ttx = finx - debx
    tty = finy - deby
    tab = np.zeros((len(lx), ttx * tty))
    # print (lx,ly)
    # print(ttx,tty)
    # print(im[deby+ly[k]:tty+ly[k]+deby,debx+lx[k]:debx+ttx+lx[k]].reshape(-1).shape)
    for k in range(len(lx)):
        tab[k, :] = im[deby + ly[k]:deby + tty + ly[k], debx + lx[k]:debx + ttx + lx[k]].reshape(-1)
    out = im.copy()
    out[deby:finy, debx:finx] = np.median(tab, axis=0).reshape((tty, ttx))
    return out

