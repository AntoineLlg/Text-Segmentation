import numpy as np
import PIL
from PIL.ImageFilter import GaussianBlur
import utils
import turn


def density(im, axis=-1):
    white_density = im.sum(axis=axis)
    return white_density


def hist(dens):
    """transforme la densité de blancs en un histogramme en vue de trouver un seuil de segmentation optimal"""


def cut_lines(im, sl=0.95):
    """
    Crée un générateur qui renvoie dans l'ordre les indice de départ et fin des lignes de texte dans l'image.

    :param im: 2D array
    :param s: float dans (0, 1) le seuil en dessous duquel on considère qu'il y a de l'écriture dans la ligne
    :return: generator
    """
    white_density = im.sum(axis=-1)

    text_line = white_density < sl * white_density.max()
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


def cut_words(line, sw=0.98):
    """
    Crée un générateur qui renvoie dans l'ordre les indice de départ et fin des mots de texte dans la ligne.

    :param line: 2D array
    :return: generator
    """
    white_density = line.sum(axis=0)

    threshold = sw * white_density.max()

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
    if init:
        yield start, end + 1


def regular_cuts(image: PIL.Image.Image):
    n, m = image.size
    for i in range(7):
        for j in range(7):
            yield image.crop((j*n//9, i*m//9, j*n//9 + n//3, i*m//9 + m//3))


def segmentation_mask(image, r=1, sl=0.95, sw=0.98):
    """
    Effectue la segmentation de l'image et renvoie une nouvelle image correspondant
    à des rectangles encadrant les mots segmentés

    :param sw: seuil de segmentation des mots
    :param sl: seuil de segmentation des lignes
    :param r: rayon du noyau gaussien utilisé pour segmenter les mots
    :param image: PIL.Image.Image
    :return:
    """
    numpy_image = np.array(image)

    blurred = np.array(image.filter(GaussianBlur(radius=r)))

    mask = PIL.Image.new('L', image.size, color=0)
    draw = PIL.ImageDraw.Draw(mask)

    n, m = numpy_image.shape

    for u, v in cut_lines(numpy_image, sl):
        for k, l in cut_words(blurred[u:v], sw):
            if u > 0 and v < n and k > 0 and l < m:
                draw.rectangle(((k, u), (l, v)), fill=255)

    return mask


def segmentation_coordinates(image, r=1, sl=0.95, sw=0.98):
    """
    Effectue la segmentation de l'image et renvoie une nouvelle image correspondant
    à des rectangles encadrant les mots segmentés

    :param sw: seuil de segmentation des mots
    :param sl: seuil de segmentation des lignes
    :param r: rayon du noyau gaussien utilisé pour segmenter les mots
    :param image: PIL.Image.Image
    :return:
    """
    numpy_image = np.array(image)

    blurred = np.array(image.filter(GaussianBlur(radius=r)))


    n, m = numpy_image.shape

    for u, v in cut_lines(numpy_image, sl):
        for k, l in cut_words(blurred[u:v], sw):
            if u > 0 and v < n and k > 0 and l < m:
                yield (k, u), (l, v)
