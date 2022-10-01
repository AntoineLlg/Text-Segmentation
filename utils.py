import numpy as np


def Get_values_without_error(im, XX, YY):
    """ retourne une image de la taille de XX et YY
     qui vaut im[XX,YY] mais en faisant attention a ce que XX et YY ne debordent
     pas """
    sh = XX.shape
    defaultval = 255;
    if len(im.shape) > 2:  # color image !
        defaultval = np.asarray([255, 255, 255])
        sh = [*sh, im.shape[2]]
    imout = np.zeros(sh)
    (ty, tx) = XX.shape[0:2]
    for k in range(ty):
        for l in range(tx):
            posx = int(XX[k, l] - 0.5)
            posy = int(YY[k, l] - 0.5)
            if posx < 0 or posx >= im.shape[1] or posy < 0 or posy >= im.shape[0]:
                valtmp = defaultval
            else:
                valtmp = im[posy, posx]
            imout[k, l] = valtmp

    return imout


def rotation(im, theta, alpha=1.0, x0=None, y0=None, ech=1, clip=True):
    """
   %
%Effectue la transformation geometrique d'une image par
%une rotation + homothetie
%
% x' = alpha*cos(theta)*(x-x0) - alpha*sin(theta)*(y-y0) + x0
% y' = alpha*sin(theta)*(x-x0) + alpha*cos(theta)*(y-y0) + y0
%
% theta : angle de rotation en degres
% alpha : facteur d'homothetie (defaut=1)
% x0, y0 : centre de la rotation (defaut=centre de l'image)
% ech : plus proche voisin (defaut=0) ou bilineaire (1)
% clip : format de l'image originale (defaut=True), image complete (False)
%

    """
    dy = im.shape[0]
    dx = im.shape[1]

    if x0 is None:
        x0 = dx / 2.0
    if y0 is None:
        y0 = dy / 2.0
    v0 = np.asarray([x0, y0]).reshape((2, 1))
    ct = alpha * np.cos(theta)
    st = alpha * np.sin(theta)
    matdirect = np.asarray([[ct, -st], [st, ct]])
    if not clip:
        # ON CALCULE exactement la transformee des positions de l'image
        # on cree un tableau des quatre points extremes
        tabextreme = np.asarray([[0, 0, dx, dx], [0, dy, 0, dy]])
        tabextreme_trans = matdirect @ (tabextreme - v0) + v0
        xmin = np.floor(tabextreme_trans[0].min())
        xmax = np.ceil(tabextreme_trans[0].max())
        ymin = np.floor(tabextreme_trans[1].min())
        ymax = np.ceil(tabextreme_trans[1].max())

    else:
        xmin = 0
        xmax = dx
        ymin = 0
        ymax = dy
    if len(im.shape) > 2:
        shout = (int(ymax - ymin), int(xmax - xmin), im.shape[2])  # image couleur
    else:
        shout = (int(ymax - ymin), int(xmax - xmin))
    dyout = shout[0]
    dxout = shout[1]
    eps = 0.0001
    Xout = np.arange(xmin + 0.5, xmax - 0.5 + eps)
    Xout = np.ones((dyout, 1)) @ Xout.reshape((1, -1))

    Yout = np.arange(ymin + 0.5, ymax - 0.5 + eps)
    Yout = Yout.reshape((-1, 1)) @ np.ones((1, dxout))

    XY = np.concatenate((Xout.reshape((1, -1)), Yout.reshape((1, -1))), axis=0)
    XY = np.linalg.inv(matdirect) @ (XY - v0) + v0
    Xout = XY[0, :].reshape(shout)
    Yout = XY[1, :].reshape(shout)
    if ech == 0:  # plus proche voisin
        out = Get_values_without_error(im, Xout, Yout)
    else:  # bilineaire
        assert ech == 1, "Vous avez choisi un echantillonnage inconnu"
        Y0 = np.floor(Yout - 0.5) + 0.5  # on va au entier+0.5 inferieur
        X0 = np.floor(Xout - 0.5) + 0.5
        Y1 = np.ceil(Yout - 0.5) + 0.5
        X1 = np.ceil(Xout - 0.5) + 0.5
        PoidsX = Xout - X0
        PoidsY = Yout - Y0
        PoidsX[X0 == X1] = 1  # points entiers
        PoidsY[Y0 == Y1] = 1  # points entiers
        I00 = Get_values_without_error(im, X0, Y0)
        I01 = Get_values_without_error(im, X0, Y1)
        I10 = Get_values_without_error(im, X1, Y0)
        I11 = Get_values_without_error(im, X1, Y1)
        out = I00 * (1.0 - PoidsX) * (1.0 - PoidsY) + I01 * (1 - PoidsX) * PoidsY + I10 * PoidsX * (
                1 - PoidsY) + I11 * PoidsX * PoidsY
    return out
