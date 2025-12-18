import pandas as pd
import numpy as np

import sklearn.datasets as skds
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error, roc_auc_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.decomposition import PCA

X_train_n = pd.read_csv("../data/train/pred_sol_large_train.csv",index_col=0)
X_test_n = pd.read_csv("../data/test/pred_sol_large_test.csv")
y_train_n = pd.read_csv("../data/train/target_sol_large_train.csv")
y_test_n = pd.read_csv("../data/test/target_sol_large_test.csv")

X_train_m = pd.read_csv("../data/train/pred_sol_nulls_train.csv",index_col=0)
X_test_m = pd.read_csv("../data/test/pred_sol_nulls_test.csv")
y_train_m = pd.read_csv("../data/train/target_sol_nulls_train.csv")
y_test_m = pd.read_csv("../data/test/target_sol_nulls_test.csv")

X_train_c = pd.read_csv("../data/train/pred_clean_train.csv",index_col=0)
X_test_c = pd.read_csv("../data/test/pred_clean_test.csv")
y_train_c = pd.read_csv("../data/train/target_clean_train.csv")
y_test_c = pd.read_csv("../data/test/target_clean_test.csv")

# XGBOOST Con X_max
xgb_reg = xgboost.XGBRegressor(random_state=42)

xgb_reg.fit(X_train_m, y_train_m)
y_pred_m = xgb_reg.predict(X_test_m)

print("MAE:", mean_absolute_error(y_test_m, y_pred_m))
print("MAPE:", mean_absolute_percentage_error(y_test_m, y_pred_m))
print("MSE:", mean_squared_error(y_test_m, y_pred_m))
print("RMSE:", np.sqrt(mean_squared_error(y_test_m, y_pred_m)))
print("r2_score:", r2_score(y_test_m, y_pred_m))

print("______________________________________________")

filename = 'models/trained_model_XGB.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(xgb_reg, archivo_salida)
    
# PCA con Linear Regression
mmc = MinMaxScaler()
X_mmc = mmc.fit_transform(X_train_c)

pca_pipe = make_pipeline(PCA(n_components=9))
pca_pipe.fit(X_mmc)
modelo_pca = pca_pipe['pca']

# Porcentaje de varianza explicada acumulada
# ==============================================================================
prop_varianza_acum = modelo_pca.explained_variance_ratio_.cumsum()
print('------------------------------------------')
print('Porcentaje de varianza explicada acumulada')
print('------------------------------------------')
print(prop_varianza_acum)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
ax.plot(
    np.arange(modelo_pca.n_components_) + 1,
    prop_varianza_acum,
    marker = 'o'
)

for x, y in zip(np.arange(modelo_pca.n_components_) + 1, prop_varianza_acum):
    label = round(y, 2)
    ax.annotate(
        label,
        (x,y),
        textcoords="offset points",
        xytext=(0,10),
        ha='center'
    )
    
ax.set_ylim(0, 1.1)
ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
ax.set_title('Porcentaje de varianza explicada acumulada')
ax.set_xlabel('Componente principal')
ax.set_ylabel('Por. varianza acumulada');
plt.show;

# Proyección variables PCA
pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=8))
modelo_pca = pca_pipe['pca']
proyecciones = pca_pipe.fit_transform(X_train_c)
proyecciones = pd.DataFrame(
    proyecciones,
    columns = ['PC1', 'PC2','PC3', 'PC4','PC5', 'PC6','PC7', 'PC8'],
    index   = X_train_c.index
)
proyecciones

# Hacemos LR con los PCA
lin_reg = LinearRegression()
lin_reg.fit(proyecciones, y_train_c)

proyecciones_t_c = pca_pipe.fit_transform(X_test_c)
predictions = lin_reg.predict(proyecciones_t_c)

#f.metricas_R(y_test_c,predictions)
print("MAE:", mean_absolute_error(y_test_c, predictions))
print("MAPE:", mean_absolute_percentage_error(y_test_c, predictions))
print("MSE:", mean_squared_error(y_test_c, predictions))
print("RMSE:", np.sqrt(mean_squared_error(y_test_c, predictions)))
print("r2_score:", r2_score(y_test_c, predictions))

print("______________________________________________")
filename = 'models/trained_model_LR+PCA.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(lin_reg, archivo_salida)
    
# Polynomial Regression
poly_feats = PolynomialFeatures(degree = 2)
poly_feats.fit(X_c)
X_poly = poly_feats.transform(X_c)
print(X_poly.shape)
print(y_train_c.shape)

X_train_p, X_test_p , y_train_p, y_test_p = train_test_split(X_poly,y_c, test_size = 0.2, random_state=42)

lin_r = LinearRegression()
lin_r.fit(X_train_p, y_train_p)

pred_lr = lin_r.predict(X_test_p)

#f.metricas_R(y_test_c,predictions)
print("MAE:", mean_absolute_error(y_test_p, pred_lr))
print("MAPE:", mean_absolute_percentage_error(y_test_p, pred_lr))
print("MSE:", mean_squared_error(y_test_p, pred_lr))
print("RMSE:", np.sqrt(mean_squared_error(y_test_p, pred_lr)))
print("r2_score:", r2_score(y_test_p, pred_lr))

print("______________________________________________")

filename = 'models/trained_model_pol.pkl'

with open(filename, 'wb') as archivo_salida:
    pickle.dump(lin_r, archivo_salida)