import pandas as pd
import numpy as np
import API_AEMET as aemet
import API_REE as ree

df_solar = ree.csv_REE()
df_aemet = aemet.csv_AEMET()

df_aemet_solar = pd.merge(df_aemet,df_solar, how='left', on=["geo_id","Año","Mes"])

#Exportamos el dataset con los datos de AEMET procesados

df_aemet_solar.to_csv("../data/raw/pred_solar.csv")