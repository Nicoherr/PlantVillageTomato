# PlantVillageTomato

## Informe técnico: clasificación de enfermedades en hojas de tomate

**Evaluación Parcial N.° 2 — TLY1102, Técnicas Avanzadas de Machine Learning I**

## 1. Problema de negocio, objetivos y KPIs

La detección oportuna de enfermedades foliares puede ayudar a priorizar inspecciones en cultivos de tomate (*Solanum lycopersicum*). El proyecto compara un Perceptrón Multicapa (MLP) con una Red Neuronal Convolucional (CNN) para clasificar imágenes en cuatro clases. El sistema es un prototipo académico de apoyo: no sustituye la evaluación de un especialista ni recomienda tratamientos.

**Objetivos**

- Preparar una variante reproducible de imágenes para clasificación multiclase.
- Implementar y entrenar una CNN en TensorFlow/Keras.
- Compararla experimentalmente con una línea base MLP en los mismos subconjuntos de datos.
- Interpretar métricas, errores, mapas de características y mapas Grad-CAM.

| KPI | Umbral/medición | Propósito |
|---|---:|---|
| Accuracy global | Objetivo > 0,85; reportar valor de prueba | Medir acierto general |
| F1 macro | Objetivo > 0,85; reportar valor de prueba | Dar el mismo peso a cada clase |
| Recall de `Tomato___Late_blight` | Objetivo > 0,80; reportar valor por clase | Vigilar falsos negativos en una enfermedad de alto impacto |
| Tiempo de entrenamiento e inferencia | Medir y reportar en la ejecución | Describir el costo computacional observado |

No se fijan aquí resultados de CNN por anticipado. Las cifras deben copiarse de la salida de una ejecución completa, pues varían con las semillas, versiones de librerías y hardware.

## 2. Fuente de datos, variante y EDA

Se reutiliza el subconjunto local de PlantVillage del proyecto, originado en [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset). La variante conserva cuatro clases de hojas de tomate:

- `Tomato___Bacterial_spot`
- `Tomato___Late_blight`
- `Tomato___Leaf_Mold`
- `Tomato___healthy`

El conjunto del repositorio contiene 500 imágenes por clase (2.000 en total). El notebook vuelve a contar los archivos y comprueba su legibilidad al ejecutarse; muestra la distribución, las resoluciones originales más frecuentes y ejemplos por clase. La selección de cuatro categorías mantiene la continuidad con el MLP de la evaluación anterior y permite realizar una comparación multiclase con recursos de cómputo acotados.

## 3. Preparación de los datos

- **CNN:** imágenes RGB redimensionadas a 128×128; la forma de entrada es `(batch, 128, 128, 3)`. La capa `Rescaling` transforma píxeles de `[0, 255]` a `[0, 1]`.
- **MLP:** las mismas imágenes se redimensionan a 64×64×3 y se aplanan en 12.288 características.
- **Partición:** 70 % entrenamiento, 15 % validación y 15 % prueba, estratificada por clase y con semilla `2026`. Las mismas rutas se asignan a cada partición para ambos modelos.
- **Aumento de datos:** volteo horizontal, rotación aleatoria de ±14,4° y zoom de ±10 %, solo durante el entrenamiento CNN.
- **Calidad:** los archivos que no se pueden abrir se reportan y excluyen antes de la partición.

El conjunto de prueba se reserva para la evaluación final y no participa en el ajuste de pesos.

## 4. Metodología CRISP-DM

1. **Comprensión del negocio:** clasificación de enfermedades foliares como apoyo a la inspección.
2. **Comprensión de los datos:** distribución, legibilidad, resoluciones y muestras visuales.
3. **Preparación:** validación de archivos, redimensionamiento, normalización, partición estratificada y aumento de datos.
4. **Modelado:** CNN y MLP entrenados con los mismos archivos de entrenamiento y validación.
5. **Evaluación:** métricas multiclase, matrices de confusión, curvas de aprendizaje, análisis de errores y comparación.
6. **Despliegue:** fuera del alcance de esta evaluación; antes de un uso real se requiere validación externa y supervisión experta.

## 5. Diseño de modelos

### CNN

La CNN recibe imágenes de 128×128×3 y utiliza tres bloques convolucionales:

```text
Entrada 128×128×3
  Conv2D(32, 3×3) → BatchNormalization → ReLU → MaxPooling2D
  Conv2D(64, 3×3) → BatchNormalization → ReLU → MaxPooling2D
  Conv2D(128, 3×3) → BatchNormalization → ReLU → MaxPooling2D
  GlobalAveragePooling2D → Dense(128, ReLU) → Dropout(0,4)
  Dense(4, Softmax)
```

Los filtros 3×3 extraen patrones locales; pooling reduce la resolución espacial entre bloques. `GlobalAveragePooling2D` reduce el número de parámetros frente a aplanar mapas grandes. Se usa Adam (`learning_rate=0,0003`), pérdida `sparse_categorical_crossentropy`, lotes de 32, hasta 20 épocas, parada temprana y reducción de tasa de aprendizaje al estancarse la validación.

### MLP de referencia

Se vuelve a entrenar la línea base anterior con la misma partición: entrada aplanada de 12.288 valores, capas densas de 128 y 64 unidades con ReLU y `Dropout(0,2)`, y salida Softmax de cuatro clases. Usa Adam (`learning_rate=0,0005`), lotes de 32 y 40 épocas. Así, la tabla de comparación procede de una ejecución común, no de comparar particiones distintas.

## 6. Entrenamiento, evaluación y comparación

El notebook [pipeline_cnn_tomatoes.ipynb](./notebooks/pipeline_cnn_tomatoes.ipynb) está organizado para reproducir el análisis completo. Durante el entrenamiento genera curvas de pérdida y accuracy para entrenamiento y validación. En el conjunto de prueba calcula accuracy, precision macro, recall macro, F1 macro y métricas por clase; además muestra matrices de confusión y ejemplos clasificados incorrectamente.

La tabla de comparación incluye las métricas, cantidad de parámetros y tiempos observados de entrenamiento e inferencia. Se guarda como `images/metricas_comparacion.csv`; las matrices, curvas y ejemplos se guardan en `images/`. **Completar la interpretación final con los valores obtenidos al ejecutar el notebook**: informar accuracy y F1 macro de ambos modelos, recall de Late blight, clases con más confusiones, diferencias de generalización y el costo computacional medido.

## 7. Feature Maps, Grad-CAM y análisis de errores

El notebook visualiza canales de activación de tres capas convolucionales en una imagen de prueba. También genera un Grad-CAM para cada clase, usando una muestra de prueba de esa clase. Las activaciones muestran qué patrones responden a los filtros; Grad-CAM resalta regiones que contribuyen a la puntuación de una clase, pero no constituye una explicación causal ni garantiza que el modelo se base en lesiones relevantes.

Se deben revisar los falsos positivos y falsos negativos junto con las matrices y métricas por clase. Confusiones entre enfermedades visualmente similares, variaciones de iluminación, fondo o resolución son hipótesis para investigar, no causas demostradas solo por las imágenes.

## 8. Consideraciones éticas, limitaciones y conclusiones

El conjunto es pequeño y de dominio acotado, y las imágenes de PlantVillage pueden diferir de fotos tomadas en terreno. El modelo puede heredar sesgos de captura y fallar con otras variedades, cámaras, iluminación o etapas de enfermedad. No se incluyen datos personales, pero existe riesgo de daño económico si una predicción errónea se interpreta como diagnóstico definitivo. El prototipo no debe usarse de manera autónoma para decidir tratamientos.

La conclusión sobre si la CNN supera al MLP debe basarse en la ejecución de prueba y explicar tanto fortalezas como limitaciones. Posibles mejoras: validación con datos de terreno independientes, búsqueda de hiperparámetros, técnicas de regularización adicionales y revisión experta de errores.

## 9. Ejecución y presentación

Desde la raíz del repositorio, instala dependencias con `pip install -r requirements.txt` y ejecuta el notebook en Jupyter o VS Code. El conjunto de datos está en `data/`; los modelos entrenados se guardan en `models/` y las visualizaciones y métricas, en `images/`.

Para una defensa individual de hasta 10 minutos, organiza la exposición así: problema y variante; EDA y preparación; explicación de convolución y pooling; arquitectura e hiperparámetros; entrenamiento; comparación y métricas; errores y mapas de activación; conclusiones, limitaciones y uso responsable. Practica explicar por qué la CNN conserva estructura espacial, qué significa cada métrica y cómo interpretar una matriz de confusión.
