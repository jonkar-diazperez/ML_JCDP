import pandas as pd
import numpy as np
import API_AEMET as aemet
import API_REE as ree
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RandomizedSearchCV


df_solar = ree.csv_REE()
df_aemet = aemet.csv_AEMET()

df_aemet_solar = pd.merge(df_aemet,df_solar, how='left', on=["geo_id","Año","Mes"])

#Exportamos el dataset con los datos de AEMET procesados

df_aemet_solar.to_csv("../data/raw/pred_solar.csv")

# Tratamiento y transformación dataset para ejecutar los modelos

df_pred_sol = pd.read_csv("../data/raw/pred_solar.csv", index_col=0)

# Eliminamos columnas con pocos registros
df_pred_sol.drop(columns=["ts_10", "ts_20", "ts_50"],inplace=True)
df_pred_sol.drop(columns=["evap"],inplace=True)
# Eliminamos columnas de min y max
df_pred_sol.drop(columns=["q_min","q_max","ta_min","ta_max", "w_racha","p_max"],inplace=True)

plt.figure(figsize=(20,20))
sns.heatmap(df_pred_sol.corr(numeric_only=True), annot=True, annot_kws={"size": 'small'}, cmap="coolwarm", vmin=-1);
plt.show()

# Dataset con nulos y columnas principales
df_pred_sol.to_csv("../data/processed/pred_sol_large.csv")
X = df_pred_sol[["glo","hr","inso","q_med","p_mes","p_sol","tm_mes"]]
y = df_pred_sol["Energía Solar"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

X_train.to_csv("../data/train/pred_sol_large_train.csv")
X_test.to_csv("../data/test/pred_sol_large_test.csv")
y_train.to_csv("../data/train/target_sol_large_train.csv")
y_test.to_csv("../data/test/target_sol_large_test.csv")

# Dataset con nulos y columnas ampliadas
X_max = df_pred_sol[["glo","hr","inso","q_med","p_mes","p_sol","tm_mes",
                     "nt_30","n_des","ti_max","tm_max"]]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_max,y, test_size = 0.2, random_state=42)

X_train_m.to_csv("../data/train/pred_sol_nulls_train.csv")
X_test_m.to_csv("../data/test/pred_sol_nulls_test.csv")
y_train_m.to_csv("../data/train/target_sol_nulls_train.csv")
y_test_m.to_csv("../data/test/target_sol_nulls_test.csv")

# Dataset sin nulos
# Eliminamos columnas con menos de 1000 no-null
df_pred = df_pred_sol.drop(columns=["n_cub","n_gra","n_fog","inso", "nv_0050", "n_des", "n_nub", "p_sol",
                                    "nv_1000","n_llu", "n_tor","n_nie","nv_0100"],inplace=True)

df_pred = df_pred_sol.copy()
df_pred.drop(columns=["w_rec"],inplace=True)
df_pred.drop(columns=["q_mar"],inplace=True)

df_pred.drop(df_pred[df_pred['indicativo'].isin(["4478X","6156X","9434","3260B"])].index,inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_pred['estacion_cod'] = le.fit_transform(df_pred['indicativo'])
df_pred.info()

df_pred.sort_values(by=['geo_id','estacion_cod','Año', 'Mes'],inplace=True)

# Tratamiento nulos restantes

df_pred.dropna(thresh=9,inplace=True)
df_pred['glo'].bfill(inplace=True)
    
df_pred["q_med"].fillna(df_pred["q_med"].mean(),inplace=True)
for c in df_pred.columns:
    print(c)
    #print(df_pred[df_pred[c].isnull()].empty)
    if not df_pred[df_pred[c].isnull()].empty: df_pred[c].fillna(df_pred[c].mean(),inplace=True)
    
df_pred.to_csv("../data/processed/pred_clean.csv")
X = df_pred.drop(columns=['indicativo', 'fecha', 'Comunidad', 'Energía Solar'])
y = df_pred["Energía Solar"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)

X_train.to_csv("../data/train/pred_clean_train.csv")
X_test.to_csv("../data/test/pred_clean_test.csv")

y_train.to_csv("../data/train/target_clean_train.csv")
y_test.to_csv("../data/test/target_clean_test.csv")
