# 🏠 House Prices Prediction - Advanced Regression Techniques

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-House%20Prices-20BEFF.svg?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

Proyecto de ciencia de datos y aprendizaje automático enfocado en la predicción del precio de venta residencial (`SalePrice`) en Ames, Iowa. Este desarrollo se enmarca en la competencia oficial de Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/).

---

## 🎯 Objetivo del Proyecto

Desarrollar un modelo predictivo robusto, generalizable y explicable capaz de tasar viviendas residenciales a partir de 79 características físicas, arquitectónicas, de ubicación y de calidad, superando de manera estadísticamente consistente el rendimiento de un modelo lineal base (*baseline*).

* **Métrica Oficial de Evaluación (Kaggle):** **RMSLE** (*Root Mean Squared Logarithmic Error*), evaluada como la raíz del error cuadrático medio sobre la variable objetivo transformada:
  $$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2}$$
  *Esta métrica penaliza de forma equitativa los errores porcentuales y evita que las propiedades de mayor valor distorsionen la función de pérdida del modelo.*

---

## 🔄 Metodología del Proyecto (CRISP-DM)

El flujo de trabajo se articuló rigurosamente en las 6 fases de CRISP-DM:

```text
┌───────────────────────────┐      ┌───────────────────────────┐
│ 1. Business Understanding │ ───> │   2. Data Understanding   │
└───────────────────────────┘      └─────────────┬─────────────┘
                                                 │
                                                 ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│       4. Modeling         │ <──> │    3. Data Preparation    │
└─────────────┬─────────────┘      └───────────────────────────┘
              │
              ▼
┌───────────────────────────┐      ┌───────────────────────────┐
│       5. Evaluation       │ ───> │       6. Deployment       │
└───────────────────────────┘      └───────────────────────────┘
```


## 📁 Estructura del Repositorio

```text
House-Prices-Prediction/
├── Docs/                                          # Documentación técnica formal
│   ├── Solución del Reto.docx.pdf                 # Reporte técnico completo en formato CRISP-DM
│   └── Documento de correcciones.pdf              # Justificación metodológica de cambios iterativos
├── data_description.txt                           # Diccionario original de variables (Ames, Iowa)
├── train_clean.csv                                # Dataset de entrenamiento procesado (1,458 filas)
├── test_clean.csv                                 # Dataset de prueba procesado (1,459 filas)
├── preparacion_de_los_datos(corregido).ipynb      # Notebook de Fase 2 y 3 (EDA y Preprocesamiento)
├── Solución_del_Reto.ipynb                        # Notebook de Fase 4 y 5 (Modelado y Evaluación)
├── submission.csv                                 # Archivo de predicciones finales generado para Kaggle
└── README.md                                      # Descripción general del proyecto
```
