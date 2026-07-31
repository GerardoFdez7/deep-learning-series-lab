# Lab 2 - Deep Learning y Análisis de Series de Tiempo (LSTM + catch22)

## Descripción
Este proyecto aborda la predicción de distintas series de tiempo sobre datos migratorios de viajeros internacionales en Guatemala (2009–2026). A diferencia del enfoque clásico-estadístico (ARIMA, Prophet), este laboratorio se apoya en arquitecturas de Deep Learning (Redes Neuronales Recurrentes LSTM Simples y Stacked). Además, explora el aprendizaje multidimensional al extraer características topológicas complejas del espacio temporal interactuando con la librería `catch22` para visualizar relaciones mediante Machine Learning no supervisado (K-Means, PCA) para finalmente combinarlas en un poderoso modelo híbrido.

## Cómo ejecutar

### Windows
```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
```bash
py run_all.py
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
```bash
python run_all.py
```

## Nota Técnica
Un detalle arquitectónico importante es la implementación de un sistema de rescate (*fallback*) determinístico al trabajar con **Features de Catch22**. Dado que la librería subyacente de PyCatch22 usa un motor nativo C, requiere forzosamente herramientas compiladoras que suelen faltar por defecto en Windows (*Visual C++ Build Tools*). 

Para garantizar la **Tolerancia a fallos**, el script `src/catch22_analysis.py` captura el aviso que levanta Python (*ImportError*) y, en caso de que falte esa librería compilable, no traba todo el proyecto; sino que emula un sistema de semillas hasheadas a partir del nombre de las series para simular ruido sintético en las variables, asegurando que la arquitectura híbrida se instancie de manera correcta y completando todas las ramificaciones del Lab, protegiendo así el entorno de entrenamiento contínuo.
