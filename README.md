# PlantVillageTomato
Evaluación 1 de Machine Learning

Informe Técnico - Proyecto de Clasificación de Enfermedades en Hojas de Tomate mediante Perceptrón Multicapa (MLP)

## 1. Descripción del Problema de Negocio
La detección temprana de enfermedades fitosanitarias en la agricultura es crítica para prevenir la pérdida masiva de cultivos y garantizar la seguridad alimentaria. En la producción de tomate, afecciones patógenas como la mancha bacteriana, el molde de la hoja y el tizón tardío representan amenazas severas que afectan el rendimiento agrícola. Este proyecto busca automatizar el diagnóstico visual mediante técnicas de Deep Learning, implementando un pipeline reproducible de clasificación de imágenes basado en un Perceptrón Multicapa (MLP).

## 2. Objetivos del Proyecto y KPIs

### Objetivos del Proyecto
- Desarrollar un flujo end-to-end de ciencia de datos aplicando la metodología CRISP-DM para el procesamiento de imágenes de hojas de tomate.
- Diseñar e implementar una arquitectura de red neuronal tipo Perceptrón Multicapa (MLP) en TensorFlow/Keras.
- Evaluar críticamente las capacidades y limitaciones de los modelos densos frente a datos no estructurados de visión por computadora.

### KPIs del Proyecto y resultado obtenido

| Métrica / KPI | Objetivo Esperado | Resultado Obtenido | ¿Se cumple? | Propósito en el Negocio |
|---|---|---|---|---|
| Accuracy Global | > 70% | **79%** | ✅ Sí | Garantizar un nivel general de aciertos en la clasificación multiclase. |
| Macro F1-Score | > 0.70 | **0.79** | ✅ Sí | Asegurar un desempeño balanceado entre las clases verdaderas y falsos positivos. |
| Tiempo de Inferencia | < 50 ms por imagen | ~2-3 ms por imagen (predicción en lote sobre CPU) | ✅ Sí | Permitir diagnósticos rápidos y eficientes en entornos agrícolas. |

*(Los valores exactos de precision/recall/F1 por clase están en la Sección 6 y son reproducibles ejecutando `notebooks/pipeline_mlp_tomatoes.ipynb`; por el uso de Dropout y operaciones no completamente deterministas de TensorFlow en CPU, una nueva ejecución puede variar en ±3-5 puntos porcentuales respecto a los valores aquí reportados, manteniendo el mismo orden de magnitud y las mismas conclusiones cualitativas.)*

## 3. Descripción de Datos y Variante del Proyecto
El proyecto utiliza una variante específica del dataset público [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset). Para garantizar la reproducibilidad y evitar trabajos idénticos, se acotó el alcance del problema de la siguiente forma:
- **Cultivo Seleccionado:** Hojas de Tomate (*Solanum lycopersicum*).
- **Clases Seleccionadas (4):** `Tomato___healthy`, `Tomato___Late_blight`, `Tomato___Bacterial_spot`, `Tomato___Leaf_Mold`.
- **Muestreo por Clase:** 500 imágenes aleatorias por categoría (2,000 imágenes en total).
- **Trazabilidad y Semilla:** Semilla global fija `SEED = 2026` para la selección y división de particiones.
- **Fuente de datos incluida en el repositorio:** a diferencia de una entrega que dependa de Google Drive o de una descarga externa al momento de ejecutar, este repositorio incluye directamente las 2,000 imágenes ya muestreadas en `data/`, para que el proyecto sea ejecutable sin modificaciones adicionales.

## 4. Análisis Exploratorio de Datos (EDA) y Preprocesamiento

### Análisis Exploratorio (EDA)
Se constató un perfecto balance de clases (500 imágenes por categoría). Las imágenes de esta variante presentan una resolución **uniforme de 256x256 píxeles con 3 canales de color (RGB)**. Se identificó que las distintas enfermedades comparten características cromáticas (variaciones de verde y marrón) y fondos similares, lo que supone un reto significativo para clasificadores vectoriales — hipótesis que se confirma más adelante en la Sección 7, donde la clase con lesiones de forma más irregular (`Late_blight`) resulta ser la más difícil de clasificar.

### Pipeline de Preprocesamiento
- **Redimensionamiento (Resize):** escala de imágenes a dimensiones fijas de 64x64 píxeles (reduce la dimensionalidad de entrada de forma manejable para un MLP; a resolución original se generaría un vector de ~196,608 features).
- **Normalización:** escalamiento de valores de píxeles del rango [0, 255] al intervalo [0.0, 1.0] para facilitar la convergencia de la red.
- **Aplanado (Flattening):** transformación de las matrices RGB de 64x64x3 a un vector unidimensional de 12,288 características por imagen.
- **Partición de Datos:** división reproducible en 70% Entrenamiento (1,400 muestras), 15% Validación (300 muestras) y 15% Prueba (300 muestras), estratificada por clase.

## 5. Metodología CRISP-DM y Arquitectura MLP
La implementación sigue las fases de la metodología CRISP-DM (Comprensión del Negocio, Comprensión de Datos, Preparación de Datos, Modelado, Evaluación y Despliegue).

### Arquitectura del Perceptrón Multicapa (MLP)

| Capa | Tipo / Configuración | Unidades / Neuronas | Función de Activación / Parámetros |
|---|---|---|---|
| Entrada | Vector aplanado | 12,288 | Normalizado [0, 1] |
| Oculta 1 | Dense + Dropout (0.2) | 128 | ReLU |
| Oculta 2 | Dense + Dropout (0.2) | 64 | ReLU |
| Salida | Dense | 4 | Softmax |

**Hiperparámetros de Entrenamiento:** Optimizador Adam (learning rate = 0.0005), función de pérdida `sparse_categorical_crossentropy`, 40 épocas y tamaño de batch de 32.

### Justificación técnica de las decisiones de arquitectura
Estas decisiones se tomaron en función de las características específicas de este problema (dataset pequeño y vector de entrada muy grande), no solo por replicar un ejemplo estándar de clase:
- **Dos capas ocultas con reducción progresiva (128 → 64):** con un vector de entrada de 12,288 features y solo 1,400 muestras de entrenamiento, una red más ancha o profunda tendería a memorizar ruido de fondo/color en vez de patrones generalizables. La reducción progresiva fuerza al modelo a comprimir la información hacia representaciones más compactas.
- **ReLU** en capas ocultas: evita la saturación de gradientes que tendrían `sigmoid`/`tanh` con más de una capa oculta.
- **Dropout(0.2):** dado el alto ratio de parámetros del modelo (~1.58 millones) respecto al tamaño del dataset (1,400 imágenes de entrenamiento), el riesgo de sobreajuste es alto; Dropout apaga aleatoriamente el 20% de las neuronas en cada paso para mitigarlo.
- **Softmax + `sparse_categorical_crossentropy`:** estándar para clasificación multiclase excluyente con etiquetas enteras.
- **Adam con `learning_rate = 0.0005`** (más conservador que el 0.001 por defecto): con un vector de entrada tan grande, un learning rate más bajo ayuda a evitar oscilaciones bruscas en la pérdida durante las primeras épocas.
- **40 épocas, batch size 32:** un batch pequeño introduce ruido estocástico que actúa como regularización adicional; 40 épocas fueron suficientes para observar estabilización de la pérdida de validación (ver curvas en `images/curvas_entrenamiento.png`) sin más costo de cómputo del necesario.

## 6. Resultados, Auditoría de Errores y Limitaciones de MLP

### Evaluación del Desempeño (conjunto de Test, 300 imágenes)

| Clase | Precision | Recall | F1-Score |
|---|---|---|---|
| Tomato___Bacterial_spot | 0.80 | 0.91 | 0.85 |
| Tomato___Late_blight | 0.65 | 0.68 | 0.67 |
| Tomato___Leaf_Mold | 0.83 | 0.83 | 0.83 |
| Tomato___healthy | 0.92 | 0.76 | 0.83 |
| **Accuracy global** | | | **0.79** |
| **Macro F1-Score** | | | **0.79** |

El modelo fue evaluado en el conjunto de prueba utilizando Accuracy, Precision, Recall y F1-Score, e inspeccionado mediante la matriz de confusión guardada en `images/matriz_confusion.png`.

### Auditoría de Errores y Limitaciones Técnicas
- **Clase con mayor nivel de confusión:** `Tomato___Late_blight` es, con diferencia, la clase más débil (F1 = 0.67). De sus 75 imágenes de prueba, 51 se clasifican correctamente; el resto se confunde principalmente con `Leaf_Mold` (11 casos) y `Bacterial_spot` (8 casos). También existe una confusión bidireccional relevante con `healthy` (11 casos en cada sentido).
- **Posible causa:** el tizón tardío (*Late blight*) presenta lesiones de forma y extensión muy variables de una hoja a otra, a diferencia de `Leaf_Mold` o `Bacterial_spot`, cuyas manchas son más consistentes en tamaño y distribución. Al aplanar la imagen en un vector 1D, esa variabilidad espacial se traduce en vectores muy distintos para la misma clase, dificultando que el MLP aprenda un patrón estable. Además, en etapas tempranas del tizón, buena parte de la hoja aún se ve verde, lo que explica la confusión con `healthy`.
- **Pérdida de Información Espacial:** al aplanar la imagen de 2D/3D a un vector 1D, el MLP pierde la relación de vecindad entre píxeles adyacentes, destruyendo patrones geométricos clave como bordes, manchas y texturas.
- **Invariancia a Traslaciones:** el MLP no posee invariancia espacial; una ligera rotación o desplazamiento de la hoja altera completamente la entrada vectorial.
- **Sensibilidad al Fondo y Color:** la red densa tiende a memorizar distribuciones generales de color en lugar de aprender características intrínsecas de las patologías de la hoja — esto se evidencia empíricamente en que las clases con manchas de color más definido (`Leaf_Mold`, `Bacterial_spot`) obtienen el mejor desempeño, mientras que la clase con lesiones más variables e irregulares (`Late_blight`) obtiene el peor.

## 7. Conclusiones y Trabajo Futuro
El proyecto permitió construir un pipeline completo y reproducible para la clasificación de imágenes agrícolas utilizando un Perceptrón Multicapa, alcanzando **79% de accuracy y 0.79 de Macro F1-Score**, superando ambos KPIs definidos. Sin embargo, el desempeño es desigual entre clases: el modelo funciona bien para `Bacterial_spot`, `Leaf_Mold` y `healthy`, pero tiene dificultades consistentes con `Late_blight`, cuyas lesiones irregulares no pueden ser bien capturadas por un modelo que ignora la estructura espacial de la imagen. Esto confirma la necesidad de adoptar arquitecturas especializadas como Redes Neuronales Convolucionales (CNN) en iteraciones futuras, junto con técnicas de aumento de datos (*data augmentation*) que expongan al modelo a mayor variabilidad de iluminación y ángulo.

## 8. Estructura del Proyecto

```
PlantVillageTomato/
├── README.md
├── data/               # 2,000 imágenes (4 clases x 500), ya incluidas para reproducibilidad
├── notebooks/
│   └── pipeline_mlp_tomatoes.ipynb   # Pipeline completo y documentado (secciones a-h), ejecutable localmente
├── models/
│   └── modelo_mlp_plantdisease.h5    # Modelo entrenado, listo para inferencia
└── images/
    ├── eda_muestras_clases.png
    ├── curvas_entrenamiento.png
    ├── matriz_confusion.png
    └── ejemplos_errores.png
```

El notebook está escrito para ejecutarse directamente desde `notebooks/` usando rutas relativas al repositorio (no depende de Google Drive ni de configuraciones externas), por lo que cualquier persona puede clonarlo y correrlo sin modificaciones adicionales.
