# 🧠 Análisis de Sentimientos en Reseñas de Amazon (Deep Learning – NLP)

Este proyecto utiliza Deep Learning con modelos de procesamiento de lenguaje natural (NLP) para clasificar reseñas de productos electrónicos de Amazon como **positivas** o **negativas**, basado en su contenido textual.

---

## 📌 Objetivo del Proyecto

Construir un clasificador robusto de sentimientos usando redes neuronales profundas (Keras + TensorFlow) para detectar automáticamente la polaridad (positiva/negativa) en reseñas de productos electrónicos, mejorando la capacidad de análisis automático de opiniones.

---

## 📁 Estructura del Proyecto
```bash
📦 sentiment-analysis-nlp/
│
├── data/ # Datos crudos o limpios (ignorados por .gitignore)
│ ├── clean_amazon_reviews_100k.csv # Dataset original limpiado
│ ├── processed_data.npz # Features y labels tokenizados
│ ├── processed_split.npz # Split de datos en train/test
│ ├── tokenizer.pkl # Tokenizer serializado
│ ├── train.ft.txt / test.ft.txt # Datos en formato FastText
│
├── models/ # Modelos entrenados (pendiente)
│ └── sentiment_model.keras # Modelo entrenado (.keras)
│
├── notebooks/
│ ├── 01_eda_feature_pipeline.ipynb # Análisis exploratorio y estadísticas
│ ├── 02_preprocessing.ipynb # Tokenización y limpieza
│ ├── 03_training_pipeline.ipynb # Entrenamiento del modelo
│ └── 04_evaluation_export.ipynb # Evaluación final + exportación
│
├── reports/
│ └── predictions_results.csv # Resultados de predicción
│
├── README.md # Este archivo
├── requirements.txt # Requisitos del proyecto
└── .gitignore # Archivos a ignorar por Git
```
---

## 🔍 Descripción del Dataset

- Fuente: [Amazon Reviews – Electronics](https://registry.opendata.aws/amazon-reviews/)
- Tamaño: 100,000 reseñas en inglés
- Variables principales:
  - `review_body`: Texto libre con la opinión del usuario.
  - `star_rating`: Calificación de 1 a 5 estrellas.
  - `sentiment`: Etiqueta binaria generada (`0 = Negativa`, `1 = Positiva`).

---

## 🔄 Flujo del Proyecto

El flujo se basa en la metodología de **Pau Labarta** con tres pipelines clave:

### 📘 1. Feature Pipeline (01_eda_feature_pipeline.ipynb)
- Limpieza y balanceo de clases.
- Estadísticas descriptivas y visuales (longitud de reseñas, frecuencia de palabras).
- Gráficos: histogramas, densidades, boxplots, nubes de palabras.

### 📙 2. Training Pipeline (03_training_pipeline.ipynb)
- Tokenización con Keras y padding.
- Arquitectura: Red neuronal con Embedding + capas Dense.
- Entrenamiento del modelo con EarlyStopping.
- Registro de experimentos con MLflow.

### 📒 3. Batch Inference Pipeline (04_evaluation_export.ipynb)
- Predicción sobre el conjunto de prueba.
- Exportación de resultados (`predictions_results.csv`).
- Evaluación con métricas de clasificación.

---

## 📊 Resultados y Métricas

- **Accuracy final alcanzada**: `~91.97%`
- **F1-score promedio ponderado**: `91.97%`
- **Precisión (Precision)**: 91.12%
- **Sensibilidad (Recall)**: 92.07%

### 🔹 Matriz de Confusión

|                 | Predicción Negativa | Predicción Positiva |
|-----------------|---------------------|----------------------|
| **Real Negativa** |       181,788        |        18,212         |
| **Real Positiva** |       15,854         |        184,146        |

El modelo muestra un **rendimiento equilibrado**, sin un sesgo fuerte hacia una de las clases, y un **buen nivel de recuperación y precisión** para ambas etiquetas. Este desempeño es adecuado para tareas de moderación automatizada o análisis de reputación de marca.

---

## 🛠️ Tecnologías Utilizadas

- Python 3.10
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
- Scikit-learn
- MLflow (para seguimiento de experimentos)
- Jupyter Notebooks
- WSL + VSCode

---

## 📦 Requisitos

Instala las dependencias:

```bash
pip install -r requirements.txt
```
## 🧪 Ejecución

```
# Entrenar desde notebook principal
jupyter notebook notebooks/03_training_pipeline.ipynb
```

---

## 📝 Créditos

Desarrollado por [@alexormx](https://github.com/alexormx) como parte del portafolio final de **AiLab** – Proyecto de clasificación binaria.
