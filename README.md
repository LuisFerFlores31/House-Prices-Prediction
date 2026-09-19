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

### 1. Entendimiento del Negocio (*Business Understanding*)
* **Problema:** La valuación inmobiliaria tradicional depende de comparables manuales propensos a sesgo humano. Un modelo cuantitativo aporta certidumbre objetiva para compra, venta y tasación.
* **Criterio de Éxito Técnico:** Superar al modelo lineal base con una mejora significativa en RMSE y $R^2$ en datos no vistos, garantizando baja varianza (sin sobreajuste).

### 2. Entendimiento de los Datos (*Data Understanding / EDA*)
* **Dimensiones:** 1,460 viviendas y 81 columnas en `train.csv`; 1,459 viviendas y 80 columnas en `test.csv`.
* **Tratamiento de Valores Faltantes Semánticos:** Se descubrió que la mayoría de los nulos (`PoolQC`, `MiscFeature`, `Alley`, `Fence`, `FireplaceQu`, `Garage*`, `Bsmt*`) representaban la **ausencia física de la característica** y no errores de recolección. Se reclasificaron con la etiqueta explícita `"SinCaracteristica"` y `0` en sus campos numéricos asociados.
* **Depuración de Valores Atípicos (*Outliers*):** Con base en las recomendaciones del autor del dataset (Dean De Cock), se eliminaron 2 observaciones con `GrLivArea > 4000 sqft` y precios inusualmente bajos (`SalePrice < $200,000`), reduciendo el conjunto de entrenamiento de 1,460 a **1,458 registros**.
* **Corrección de Errores de Captura:** Se detectó en `test.csv` un año imposible en `GarageYrBlt = 2207`, corregido automáticamente mediante el año de construcción de la vivienda (`YearBuilt`).

### 3. Preparación de los Datos (*Data Preparation*)
* **Ingeniería de Características (*Feature Engineering*):**
  * `Edad_Casa` = $\text{YrSold} - \text{YearBuilt}$
  * `Area_Total_Construida` = $\text{TotalBsmtSF} + \text{1stFlrSF} + \text{2ndFlrSF}$
  * `Total_Banos` = $\text{FullBath} + 0.5(\text{HalfBath}) + \text{BsmtFullBath} + 0.5(\text{BsmtHalfBath})$
  * `Es_Remodelada` = $1$ si $\text{YearRemodAdd} \neq \text{YearBuilt}$, $0$ en caso contrario.
* **Transformación Logarítmica:** Aplicación de $\log(1+x)$ a `SalePrice` y a variables de superficie con sesgo extremo (`LotArea`, `GrLivArea`, `TotalBsmtSF`, `MasVnrArea`) para mitigar el apalancamiento (*leverage*).
* **Prevención de Fuga de Datos (*Data Leakage*):** El preprocesamiento (escalado con `StandardScaler` y codificación con `OneHotEncoder(handle_unknown="ignore")`) se encapsuló en un **Pipeline de scikit-learn**, asegurando que las transformaciones se ajusten **exclusivamente con los datos de entrenamiento**.
* **Validación Empírica del Periodo de Crisis (2008–2010):** Se evaluó mediante 5-Fold Cross-Validation si excluir los años de la crisis inmobiliaria mejoraba el modelo. El error aumentó un 8.4% al removerlos, demostrando empíricamente que retener todo el histórico previene pérdida de información.

### 4. Modelado (*Modeling*)
Se implementó un diseño de prueba riguroso con partición **80% entrenamiento (1,166 casas)** y **20% prueba interna (292 casas)**, combinada con **5-Fold Cross-Validation** sobre el subconjunto de entrenamiento:
* **Modelos comparados:**
  1. *Regresión Lineal Simple* (Baseline de referencia)
  2. *Ridge Regression* (Regularización $L_2$ mediante `GridSearchCV`)
  3. *Lasso Regression* (Regularización $L_1$ mediante `GridSearchCV`)
  4. *Random Forest Regressor* (Bagging optimizado mediante `RandomizedSearchCV`)
  5. *Gradient Boosting Regressor* (Boosting secuencial optimizado mediante `RandomizedSearchCV`)

### 5. Evaluación (*Evaluation*)

Los modelos fueron evaluados sobre el conjunto de prueba independiente (`X_test`):

| Modelo | RMSE Test (Log) | $R^2$ Test | MAE Test | Diagnóstico Sesgo / Varianza |
| :--- | :---: | :---: | :---: | :--- |
| 🥇 **Gradient Boosting** | **0.1429** | **0.8789** | **0.0996** | Excelente equilibrio; brecha train-test mínima (0.0289). |
| 🥈 **Lasso** | 0.1486 | 0.8691 | 0.1069 | Muy interpretable; selección natural de variables ($\\alpha=0.001$). |
| 🥉 **Ridge** | 0.1491 | 0.8680 | 0.1067 | Control efectivo de multicolinealidad ($\\alpha=50$). |
| 4. **Random Forest** | 0.1515 | 0.8639 | 0.1007 | Sobreajuste evidente (error train 0.0522 vs test 0.1515). |
| 5. **Regresión Lineal** | 0.1526 | 0.8618 | 0.1099 | Baseline superado exitosamente (+6.4% de mejora en RMSE). |

* **Variables más influyentes (*Feature Importance*):** `OverallQual` (Calidad General), `GrLivArea` (Superficie habitable), `TotalBsmtSF` (Área de sótano), `YearBuilt` y ubicación geográfica (`Neighborhood`), corroborando de forma cuantitativa los hallazgos del negocio.

### 6. Despliegue (*Deployment*)
* Debido al alcance académico y competitivo de Kaggle, el despliegue se implementó como un **proceso por lotes (*batch*)**.
* El pipeline final procesa las 1,459 observaciones de `test.csv`, predice el precio en logaritmo y aplica $\text{exp}(\\hat{y}) - 1$ para generar el archivo oficial:
  📄 `submission.csv` con columnas `Id` y `SalePrice`.

---

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

---

## 🚀 Cómo Reproducir el Proyecto

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/LuisFerFlores31/House-Prices-Prediction.git
   cd House-Prices-Prediction
   ```

2. **Instalar dependencias necesarias:**
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

3. **Ejecutar los cuadernos de Jupyter:**
   * Abrir y ejecutar `preparacion_de_los_datos(corregido).ipynb` para replicar el EDA, la limpieza y la ingeniería de variables.
   * Abrir y ejecutar `Solución_del_Reto.ipynb` para entrenar los modelos, generar la validación cruzada y crear `submission.csv`.
