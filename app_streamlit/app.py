import streamlit as st
import pickle as pic
import base64
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath("../src"))
import evaluation_XGB as eval


def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
      background-image: url("data:image/png;base64,{bin_str}");
      background-size: cover;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
set_background("static/Fondo_paneles_app.jpg")

st.title("ESTIMACIÓN ENERGÍA SOLAR")

st.text("Introduce los datos de la zona de la que quieres calcular la estimación:",width="stretch")
indicativo = st.selectbox(
    "Estación referencia",
    ("Coruña", "Albacete", "Almería Aer","Barcelona Port"),
)
mes_pred = st.selectbox(
    "Mes estimación",
    (1,2,3,4,5,6,7,8,9,10,11,12),
)

glo = st.number_input("Radiación global mensual (decenas de kJ/m2)",
                      min_value=0.0,value="min",format="%0.2f")
st.write("The current number is ", glo)

hr = st.number_input("Humedad relativa media mensual (%)",
                      min_value=0.0,value="min",format="%0.2f")
st.write("The current number is ", hr)
inso = st.number_input("Media mensual/anual de la insolación diaria (horas)",
                      min_value=0,value="min")
st.write("The current number is ", inso)
q_med = st.number_input("Presión media mensual (hPa)",
                      min_value=0.0,value="min",format="%0.2f")
st.write("The current number is ", q_med)
tm_mes = st.number_input("Temperatura media mensual (ºC)",
                      min_value=0.0,value="min",format="%0.2f")
st.write("The current number is ", tm_mes)
nt_30 = st.number_input("Nº de días de temperatura máxima mayor o igual que 30 °C",
                      min_value=0,value="min")
st.write("The current number is ", nt_30)
n_des = st.number_input("Nº de días despejados en el mes",
                      min_value=0,value="min")
st.write("The current number is ", n_des)

tm_max = st.number_input("Temperatura media de las máximas (ºC)",
                      min_value=0.0,value="min",format="%0.2f")
st.write("The current number is ", tm_max)
p_sol = st.number_input("Porcentaje medio mensual de la insolación diaria (%)",
                      min_value=0.0,value="min",format="%0.2f")
st.write("The current number is ", p_sol)
p_mes = st.number_input("Precipitación total mensual (mm)",
                      min_value=0.0,value="min",format="%0.2f")
st.write("The current number is ", p_mes)

# Transformación inputs en X_test
match indicativo:
    case "Coruña": 
        estacion_cod = 4
        geo_id = 17
    case "Albacete": 
        estacion_cod = 21
        geo_id = 7        
    case "Almería Aer": 
        estacion_cod = 19
        geo_id = 4        
    case "Barcelona Port": 
        estacion_cod = 0
        geo_id = 9        
mes_pred = int(mes_pred)


if st.button("Entrenar"):
    X_app_t = pd.DataFrame({'glo':glo,
                            'hr':hr,
                            'inso':inso,
                            'q_med':q_med,
                            #'tm_min': tm_mes-5,
                            #'ts_min': tm_mes-5,
                            #'np_100':np_100,
                            #'nw_91':0,
                            #'np_001':np_100,
                            #'e':e,
                            #'np_300':0,
                            'p_mes':p_mes,
                            'p_sol':p_sol,
                            'tm_mes':tm_mes,
                            'nt_30':nt_30,
                            'n_des':n_des,
                            #'nt_00':0,
                            'ti_max':tm_max-5,
                            'tm_max':tm_max,
                            #'np_010':np_100,
                            #'Año':2025,
                            #'Mes':mes_pred,
                            #'geo_id':geo_id,
                            #'estacion_cod':estacion_cod
                            },index=[0])
    #y_app_t = eval.pred_XGB(X_app_t)
    filename = '../models/trained_model_XGB.pkl'
    with open(filename, 'rb') as archivo_entrada:
            modelo_importado = pic.load(archivo_entrada)
        
    pred_t = modelo_importado.predict(X_app_t) 

 
    st.write(pred_t)