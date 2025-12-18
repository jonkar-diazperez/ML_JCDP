import pandas as pd
import numpy as np
import requests
import json

# Carga ficheros REE con datos mensuales de energías renovables
def csv_REE():
    df_solar_w = pd.DataFrame()
    for i in [2023,2024,2025]:
        ree_url ="https://apidatos.ree.es"
        path_renov = "/es/datos/generacion/estructura-renovables"
        ree_api_param = f"?start_date={i}-01-01T00:00&end_date={i}-12-31T23:59&time_trunc=month&geo_limit=ccaa&all_ccaa=allCcaa&tecno_select=1458&widget=estructura-renovables"

        api_ree_url = ree_url + path_renov + ree_api_param

        print(api_ree_url)

        response = requests.get(api_ree_url)

        if response.status_code == 200:  # Código 200 indica una respuesta exitosa.
            data = response.json()
            print(data)
            df_solar_w = pd.concat([df_solar_w, pd.DataFrame(data["included"])[:-1]],ignore_index=True)

            
        else:
            print("Error en la solicitud: ", response.status_code)
        
#print(df_solar_w)

# Procesamiento de diccionario con datos energía para formatear fichero final REE
# con datos mensuales de energía solar fotovoltaica

    reg = []
    for i in df_solar_w.index:
        print(i, "geo_id: ", df_solar_w.loc[i]['geo_id'], "Comunidad: ", df_solar_w.loc[i]['community_name'])
        content = df_solar_w.loc[i]["content"]
        for c in content:
            if c['type'] == 'Solar fotovoltaica':
                for v in c['attributes']['values']:
                    print(v['value'], v['datetime'][0:4], v['datetime'][5:7])
                    r = [df_solar_w.loc[i]['geo_id'],
                        df_solar_w.loc[i]['community_name'],
                        v['datetime'][0:4],
                        v['datetime'][5:7],
                        round(v['value'],2)
                        ]
                    print(r)
                    reg.append(r)
                    
        df_solar = pd.DataFrame(reg,columns=['geo_id', 'Comunidad', 'Año', 'Mes', 'Energía Solar'])

    df_solar["Mes"] = df_solar["Mes"].astype(int)

    df_solar.to_csv("../data/raw/REE/Solar fotovoltaica.csv")
    return df_solar
