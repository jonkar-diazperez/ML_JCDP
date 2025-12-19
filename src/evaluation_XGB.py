import pandas as pd
import numpy as np

import pickle

import xgboost
from sklearn.decomposition import PCA
import sys
import os
sys.path.append(os.path.abspath("../src"))

def pred_XGB(X_t):
    filename = '../models/trained_model_XGB.pkl'
    with open(filename, 'rb') as archivo_entrada:
        modelo_importado = pickle.load(archivo_entrada)
        
    pred_t = modelo_importado.predict(X_t) 
    
    return pred_t
