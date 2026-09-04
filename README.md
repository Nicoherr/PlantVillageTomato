# PlantVillageTomato
Evaluación 1 de Machine Learning

Informe Técnico - Proyecto de Clasificación de Enfermedades en Hojas de Tomate mediante Perceptrón Multicapa (MLP)
1. Descripción del Problema de Negocio
La detección temprana de enfermedades fitosanitarias en la agricultura es crítica para prevenir la pérdida masiva de cultivos y garantizar la seguridad alimentaria. En la producción de tomate, afecciones patógenas como la mancha bacteriana, el molde de la hoja y el tizón tardío representan amenazas severas que afectan el rendimiento agrícola. Este proyecto busca automatizar el diagnóstico visual mediante técnicas de Deep Learning, implementando un pipeline reproducible de clasificación de imágenes basado en un Perceptrón Multicapa (MLP).
2. Objetivos del Proyecto y KPIs
Objetivos del Proyecto
Desarrollar un flujo end-to-end de ciencia de datos aplicando la metodología CRISP-DM para el procesamiento de imágenes de hojas de tomate.
Diseñar e implementar una arquitectura de red neuronal tipo Perceptrón Multicapa (MLP) en TensorFlow/Keras.
Evaluar críticamente las capacidades y limitaciones de los modelos densos frente a datos no estructurados de visión por computadora.
KPIs del Proyecto
Métrica / KPI
Objetivo Esperado
Propósito en el Negocio
 
Accuracy Global
> 70%
Garantizar un nivel general de aciertos en la clasificación multiclase.
Macro F1-Score
> 0.70
Asegurar un desempeño balanceado entre las clases verdaderas y falsos positivos.
Tiempo de Inferencia
< 50 ms por imagen
Permitir diagnósticos rápidos y eficientes en entornos agrícolas.

3. Descripción de Datos y Variante del Proyecto
El proyecto utiliza una variante específica del dataset público PlantVillage. Para garantizar la reproducibilidad y evitar trabajos idénticos, se acotó el alcance del problema de la siguiente forma:
Cultivo Seleccionado: Hojas de Tomate (Solanum lycopersicum).
Clases Seleccionadas (4): Tomato___healthy, Tomato___Late_blight, Tomato___Bacterial_spot, Tomato___Leaf_Mold.
Muestreo por Clase: 500 imágenes aleatorias por categoría (2,000 imágenes en total).
Trazabilidad y Semilla: Semilla global fija SEED = 2026 para la selección y división de particiones.
4. Análisis Exploratorio de Datos (EDA) y Preprocesamiento
Análisis Exploratorio (EDA)
Se constató un perfecto balance de clases (500 imágenes por categoría). Las imágenes originales presentan una resolución variable con 3 canales de color (RGB). Se identificó que las distintas enfermedades comparten características cromáticas (variaciones de verde y marrón) y fondos similares, lo que supone un reto significativo para clasificadores vectoriales.
Pipeline de Preprocesamiento
Redimensionamiento (Resize): Escala de imágenes a dimensiones fijas de 64x64 píxeles.
Normalización: Escalamiento de valores de píxeles del rango [0, 255] al intervalo [0.0, 1.0] para facilitar la convergencia de la red.
Aplanado (Flattening): Transformación de las matrices RGB de 64x64x3 a un vector unidimensional de 12,288 características por imagen.
Partición de Datos: División reproducible en 70% Entrenamiento (1,400 muestras), 15% Validación (300 muestras) y 15% Prueba (300 muestras).
5. Metodología CRISP-DM y Arquitectura MLP
La implementación sigue las fases de la metodología CRISP-DM (Comprensión del Negocio, Comprensión de Datos, Preparación de Datos, Modelado, Evaluación y Despliegue).
Arquitectura del Perceptrón Multicapa (MLP)
Capa
Tipo / Configuración
Unidades / Neuronas
Función de Activación / Parámetros
 
Entrada
Vector aplanado
12,288
Normalizado [0, 1]
Oculta 1
Dense + Dropout (0.2)
128
ReLU
Oculta 2
Dense + Dropout (0.2)
64
ReLU
Salida
Dense
4
Softmax

Hiperparámetros de Entrenamiento: Optimizador Adam (learning rate = 0.0005), función de pérdida sparse_categorical_crossentropy, 40 épocas y tamaño de batch de 32.
6. Resultados, Auditoría de Errores y Limitaciones de MLP
Evaluación del Desempeño
El modelo fue evaluado en el conjunto de prueba utilizando métricas clave (Accuracy, Precision, Recall y F1-Score) e inspeccionado mediante la matriz de confusión guardada en la carpeta de imágenes del proyecto.
Auditoría de Errores y Limitaciones Técnicas
Pérdida de Información Espacial: Al aplanar la imagen de 2D/3D a un vector 1D, el MLP pierde la relación de vecindad entre píxeles adyacentes, destruyendo patrones geométricos clave como bordes, manchas y texturas.
Invariancia a Traslaciones: El MLP no posee invariancia espacial; una ligera rotación o desplazamiento de la hoja altera completamente la entrada vectorial.
Sensibilidad al Fondo y Color: La red densa tiende a memorizar distribuciones generales de color en lugar de aprender características intrínsecas de las patologías de la hoja.
7. Conclusiones y Trabajo Futuro
El proyecto permitió construir un pipeline completo y reproducible para la clasificación de imágenes agrícolas utilizando un Perceptrón Multicapa. Si bien el modelo MLP sirve como una sólida línea base (baseline), sus limitaciones estructurales en el procesamiento de datos bidimensionales demuestran la necesidad de adoptar arquitecturas especializadas como Redes Neuronales Convolucionales (CNN) en iteraciones futuras.
