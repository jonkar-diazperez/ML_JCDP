# PREDICCIÓN GENERACION ENERGÍA SOLAR ☀️
# (C) by Juan Carlos Díaz Pérez

Este proyecto utiliza técnicas de **Machine Learning** para predecir la producción de energía de plantas solares basándose en datos históricos y variables meteorológicas. El objetivo es proporcionar una herramienta que permitiera mejorar la gestión de la red eléctrica y calcular la producción y eficiencia de fuentes de energía renovables.

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](https://www.google.com/search?q=%23descripci%C3%B3n-del-proyecto)
2. [Estructura del Repositorio](https://www.google.com/search?q=%23estructura-del-repositorio)
3. [Instalación y Configuración](https://www.google.com/search?q=%23instalaci%C3%B3n-y-configuraci%C3%B3n)
4. [Flujo de Trabajo](https://www.google.com/search?q=%23flujo-de-trabajo)
5. [Visualización y Despliegue](https://www.google.com/search?q=%23visualizaci%C3%B3n-y-despliegue)
6. [Métricas de Evaluación](https://www.google.com/search?q=%23m%C3%A9tricas-de-evaluaci%C3%B3n)

---

## 🚀 Descripción del Proyecto

El proyecto pretende predecir la producción de energía solar fotovoltaica de una zona a partir de la información meteorológica y de radiación solar de la misma. Mediante el análisis de datos meteorológicos recogidos en las estaciones de AEMET que son publicados mendiante API, como la radiación global, la temperatura y la nubosidad, se entrena un modelo capaz de estimar la generación en kilovatios (MWh).

**Tecnologías utilizadas:**

* **Lenguaje:** Python 3.x
* **Librerías:** Pandas, Numpy, Scikit-learn, Matplotlib, Seaborn.
* **Visualización:** Streamlit.

---

## 📂 Estructura del Repositorio

```bash
|-- data
|   |-- raw                # Datos originales sin procesar.
|   |-- processed          # Datos tras la limpieza y feature engineering.
|   |-- train / test       # Splits utilizados para modelado.
|
|-- notebooks              # Experimentos y análisis exploratorio (EDA).
|   |-- 01_Fuentes.ipynb
|   |-- 02_LimpiezaEDA.ipynb
|   |-- 03_Entrenamiento_Evaluacion.ipynb
|
|-- src                    # Código fuente modular (scripts .py).
|   |-- data_processing.py # Transformaciones de datos.
|   |-- training.py        # Scripts de entrenamiento de modelos.
|   |-- evaluation.py      # Funciones de métricas y validación.
|
|-- models                 # Artefactos del modelo.
|   |-- trained_model.pkl  # Modelo final exportado.
|   |-- model_config.yaml  # Hiperparámetros y configuración.
|
|-- app_streamlit          # Aplicación web interactiva.
|   |-- app.py             # Interfaz de usuario.
|   |-- requirements.txt   # Dependencias específicas de la app.
|
|-- docs                   # Documentación y presentaciones.
    |-- negocio.ppt        # Enfoque de negocio.
    |-- ds.ppt             # Enfoque técnico (Data Science).
    |-- memoria.md         # Documentación detallada del proceso.

```

---

## 🛠️ Instalación y Configuración

1. **Clonar el repositorio:**
```bash
git clone https://github.com/jonkar-diazperez/ML_JCDP.git
cd ML_JCDP

```

2. **Instalar dependencias:**
```bash
pip install -r app_streamlit/requirements.txt

```


---

## 🔄 Flujo de Trabajo

Para reproducir los resultados, sigue este orden en los scripts de la carpeta `src`:

1. **Procesamiento:** Ejecuta `src/data_processing.py` para importar los datasets con los datos de AEMET y REE de `data/raw` y limpiar los datos para crear los datasets de entrenamiento en `data/processed`.
2. **Entrenamiento:** el script `src/training.py` contiene las instrucciones para importar los datos de entrenamiento y ejecutar las pruebas de los distintos modelos ML para almacenarlos en la carpeta `models/`.
3. **Evaluación:** contiene el script `src/evaluation_XGB.py` para ejecutar el modelo XGBoost, importando los datos de prueba.

---

## 📊 Visualización y Despliegue

El proyecto incluye una app interactiva construida con **Streamlit** que permite ejecutar el mejor modelo entrenado para realizar predicciones online con los datos introducidos por el usuario.

**Para lanzar la aplicación:**

```bash
cd app_streamlit
streamlit run app.py

```

---

## 📝 Notas Adicionales

* Los archivos dentro de `docs/` contienen las presentaciones solicitadas en el proyecto, orientadas hacia usuarios de negocio y del equipo técnico respectivamente.

