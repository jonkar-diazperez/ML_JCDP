import pandas as pd
import numpy as np
import requests
import json

def csv_AEMET():
    df_aemet_raw = pd.read_json('../data/raw/AEMET/AEMET Estaciones DATOS Anuales.json', orient='records')
    df_aemet_fechas = df_aemet_raw["fecha"].str.split('-', expand=True)
    df_aemet_raw["Año"] = df_aemet_fechas[0]
    df_aemet_raw["Mes"] = df_aemet_fechas[1]

    list_geo_id = []
    for i in df_aemet_raw.index:
        match df_aemet_raw.loc[i]["indicativo"]:
            case "1387": list_geo_id.append(17)
            case "8178D": list_geo_id .append(7)
            case "6325O": list_geo_id .append(4)
            case "4478X": list_geo_id .append(16)
            case "0201D": list_geo_id .append(9)
            case "3469A": list_geo_id .append(16)
            case "5973": list_geo_id .append(4)
            case "4121": list_geo_id .append(7)
            case "5402": list_geo_id .append(4)
            case "5514": list_geo_id .append(4)
            case "4642E": list_geo_id .append(4)
            case "C430E": list_geo_id .append(19)
            case "2661": list_geo_id .append(8)
            case "9771C": list_geo_id .append(9)
            case "9170": list_geo_id .append(20)
            case "3194U": list_geo_id .append(13)
            case "6156X": list_geo_id .append(4)
            case "C639M": list_geo_id .append(19)
            case "5860E": list_geo_id .append(4)
            case "7178I": list_geo_id .append(21)
            case "1249X": list_geo_id .append(11)
            case "B228": list_geo_id .append(18)
            case "1549": list_geo_id .append(8)
            case "2462": list_geo_id .append(13)
            case "2867": list_geo_id .append(8)
            case "1024E": list_geo_id .append(10)
            case "1111": list_geo_id .append(6)        
            case "1111X": list_geo_id .append(6)
            case "2030": list_geo_id .append(8)
            case "C449C": list_geo_id .append(19)
            case "8368U": list_geo_id .append(5)
            case "3260B": list_geo_id .append(7)
            case "9981A": list_geo_id .append(9)
            case "8414A": list_geo_id .append(15)
            case "2422": list_geo_id .append(8)
            case "9434": list_geo_id .append(5)

    df_aemet_raw["geo_id"] = list_geo_id
    df_aemet_raw["Mes"] = df_aemet_raw["Mes"].astype(int)
    df_aemet_raw.to_csv("../data/raw/AEMET/AEMET Datos anuales.csv")
    return df_aemet_raw


