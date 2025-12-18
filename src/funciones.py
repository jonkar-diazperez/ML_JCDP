import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import *



def train_test(X,y, size, semilla:int):
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=size, random_state=semilla)
    return X_tr, X_ts, y_tr, y_ts

def escalar_X(Tr, Ts, tipoEscala=None):
    if tipoEscala != None:
        if tipoEscala == "StandardScaler":
            sc = StandardScaler()
            sc.fit(Tr)
            X_tr_sc = sc.transform(Tr)
            X_ts_sc = sc.transform(Ts)
            return X_tr_sc, X_ts_sc
        elif tipoEscala == "MinMaxScaler":
            mmc = MinMaxScaler()
            mmc.fit(Tr)
            X_tr_mmc = mmc.transform(Tr)
            X_ts_mmc = mmc.transform(Ts)
            return X_tr_mmc, X_ts_mmc
    else: "Función no encontrada"
    
def graficos(df):
    plt.figure(figsize=(8,8))
    sns.heatmap(df.corr(numeric_only = True), cmap="coolwarm", vmin = -1, annot=True)
    plt.title("Matriz de correlación")
 
    sns.pairplot(df)
    plt.title("Pairplot")
    plt.show()
    
def metricas_R(y_test,y_pred):
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("r2_score:", r2_score(y_test, y_pred))

    print("______________________________________________")
    return "OK"
