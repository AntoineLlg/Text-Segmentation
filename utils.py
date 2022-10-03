import numpy as np
import matplotlib.pyplot as plt
import PIL


def view_image(im, figsize=(10, 10)):
    """
    Permet de lire une image en niveaux de gris

    :param im: np.array image
    :param figsize: (float, float) taille de l'image affichée
    :return:
    """
    fig = plt.figure(figsize=figsize)
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
    return im.rotate(theta*180/np.pi, resample=PIL.Image.BICUBIC)


def crop(im, size):
    """
    découpe un carré ou rectangle au centre de l'image, de dimension(s) size. Si size est un entier, il est utilisé
    à la fois pour la longueur et la largeur.

    :param im: PIL.Image.Image
    :param size: (int, int) or int
    :return:
    """
    M, N = im.size[0]//2, im.size//2
    if isinstance(size, tuple):
        L, K = size[0]//2, size[1]//2
        return im.crop((N-L, M-K, N+L, M+K))
    else:
        L = size//2
        return im.crop((N - L, M - L, N + L, M + L))