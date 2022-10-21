import numpy as np


def density(im, axis=-1):
    white_density = im.sum(axis=axis)
    return white_density


def hist(dens):
    """transforme la densité de blancs en un histogramme en vue de trouver un seuil de segmentation optimal"""


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
                yield im[start: end + 1]
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
                yield line[:, start: end + 1]
        else:
            if text_line[i]:
                start = i
                end = i
                init = True
