import numpy as np
import matplotlib.pyplot as plt

import cv2

import skimage.io, skimage.feature

import os, shutil
import random

from PIL import Image

def _relleno_(Imagen):
    """
    Esta función genera un marco de ceros alrededor de una imagen con la finalidad de considerar los pixeles
    en los bordes y esquinas.
    """
    
    # Creamos las tiras de ceros laterales:
    ceros_laterales = np.zeros(shape=(Imagen.shape[0],1))

    # Comenzamos a realizar los marcos laterales:
    Imagen = np.hstack((ceros_laterales,Imagen))
    Imagen = np.hstack((Imagen,ceros_laterales))
        
    # Creamos las tiras de los ceros superiores e inferiores:
    ceros_superiores_e_inferiores = np.zeros(shape=(1,Imagen.shape[1]))

    # Comenzamos a realizar el marco superior e inferior:
    Imagen = np.vstack((ceros_superiores_e_inferiores,Imagen))
    Imagen = np.vstack((Imagen,ceros_superiores_e_inferiores))
    
    return Imagen
    
def convolucion(img, filtros):
    """Esta función convoluciona los filtros ingresados con la imagen deseada."""
    
    ReLU = lambda x: (x > 0)*x
    #bordes = lambda x: (x >= 0.1)*1
    
    altura = img.shape[0]
    anchura = img.shape[1]
    
    img = _relleno_(img.copy())
    
    filas = 0
    
    i = 0
    
    patches = list()
    
    while (filas < altura):
        
        columnas = 0
        
        while (columnas < anchura):
            
            patches.append(img[filas:filas+3, columnas:columnas+3])
            
            columnas += 1
            
        filas += 1
        
    patches = np.asarray(patches).reshape((altura*anchura,-1))
    convoluciones = patches@filtros # Aquí se lleva acabo la convolución.
        
    resultados = ReLU(convoluciones)
        
    bordes_horizontales = resultados[:,0].reshape((altura,anchura))
    bordes_verticales = resultados[:,1].reshape((altura,anchura))
    
    Imagen_resultante = bordes_horizontales + bordes_verticales
    Imagen_resultante = (Imagen_resultante/Imagen_resultante.max()).astype("float64")
    
    return Imagen_resultante