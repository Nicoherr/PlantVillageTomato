# PlantVillageTomato

Informe Técnico - Clasificación de Enfermedades en Hojas de Tomate mediante Redes Neuronales Convolucionales (CNN) y Comparación con Perceptrón Multicapa (MLP)

---

## 1. Descripción del Problema de Negocio

La detección temprana de enfermedades fitosanitarias en la agricultura es crítica para prevenir la pérdida masiva de cultivos de tomate (*Solanum lycopersicum*). Mientras que en la Evaluación Parcial 1 se abordó este problema utilizando un Perceptrón Multicapa (MLP), dicho modelo evidenció severas limitaciones estructurales al aplanar los datos espaciales (1D), resultando en pérdidas significativamente elevadas de información geométrica y textural.

Esta Evaluación Parcial 2 evoluciona la solución mediante el diseño e implementación de una **Red Neuronal Convolucional (CNN)**. Las CNN aplican filtros convolucionales capaces de preservar la dimensionalidad y estructura espacial en 2D/3D, permitiendo la extracción automática e invariante de características jerárquicas (bordes, manchas, texturas) para elevar la precisión diagnóstica y mejorar la toma de decisiones agrícolas.

---

## 2. Objetivos del Proyecto y KPIs

### Objetivos del Proyecto
- Desarrollar un pipeline reproducible de Deep Learning adaptado para datos tensoriales de visión por computadora.
- Diseñar e implementar una arquitectura CNN personalizada en Keras/TensorFlow.
- Evaluar, comparar y justificar técnicamente el desempeño cuantitativo y cualitativo de la arquitectura CNN frente al modelo MLP previo.
- Analizar e interpretar los mapas de características (*Feature Maps*) y mapas de activación para comprender el proceso interno de extracción de características de la red.

### KPIs del Proyecto y Comparativa de Resultados

| Métrica / KPI | Objetivo Esperado | Resultado MLP (EP1) | Resultado CNN (EP2) | ¿Se cumple el KPI? | Propósito en el Negocio |
|---|---|---|---|---|---|
| **Accuracy Global** | > 85% | 79.0% | **92.3%** | ✅ Sí | Maximizar el acierto diagnóstico general en el cultivo. |
| **Macro F1-Score** | > 0.85 | 0.79 | **0.92** | ✅ Sí | Garantizar un rendimiento equilibrado en todas las patologías. |
| **Recall en *Late blight*** | > 0.80 | 0.68 | **0.88** | ✅ Sí | Minimizar falsos negativos en la enfermedad más destructiva. |
| **Tiempo de Inferencia** | < 50 ms / imagen | ~2-3 ms | **~5-8 ms** | ✅ Sí | Permitir el procesamiento en tiempo real en dispositivos de campo. |

---

## 3. Continuidad del Problema y Trazabilidad de la Variante

Para asegurar la trazabilidad experimental y comparar objetivamente ambas arquitecturas, se mantuvo estrictamente la **variante de datos original** de la Evaluación Parcial 1:

- **Cultivo Seleccionado:** Hojas de Tomate (*Solanum lycopersicum*).
- **Clases Seleccionadas (4):** `Tomato___healthy`, `Tomato___Late_blight`, `Tomato___Bacterial_spot`, `Tomato___Leaf_Mold`.
- **Volumen de Datos:** 2,000 imágenes en total (500 imágenes por clase, perfectamente balanceado).
- **Semilla de Reproducibilidad:** `SEED = 2026` para la partición estratificada de datos.
- **Estructura de Partición:** 70% Entrenamiento (1,400 muestras), 15% Validación (300 muestras) y 15% Prueba (300 muestras).

---

## 4. Preprocesamiento y Preparación de Datos para CNN

A diferencia del pipeline MLP donde las imágenes se redimensionaron a 64x64x3 y se aplanaron a vectores de 12,288 dimensiones, la CNN requiere preservar el formato tensorial en 2D/3D:

1. **Reescalado de Resolución:** Las imágenes se procesan a una resolución de **128x128x3 píxeles** (RGB). Esto proporciona una resolución superior a la usada en MLP (64x64), permitiendo capturar micro-lesiones y bordes finos de las manchas.
2. **Conservación Tensorial:** Se mantiene la forma del tensor (B, 128, 128, 3), preservando la adjacencia espacial entre píxeles adyacentes.
3. **Normalización:** Los valores de los píxeles se reescalan del rango [0, 255] al intervalo [0.0, 1.0] dividiendo por 255.0 para estabilizar y acelerar la convergencia del gradiente.
4. **Data Augmentation:** Se aplica aumento de datos en tiempo de ejecución (rotaciones aleatorias hasta ±15°, cambios de zoom de ±10% y giros horizontales) para mejorar la capacidad de generalización del modelo frente a variaciones de captura en terreno.

---

## 5. Arquitectura de la Red Neuronal Convolucional (CNN)

### Configuración del Modelo

```text
Tensor de Entrada (128, 128, 3)
   │
   ├── Conv2D (32 filtros 3x3, ReLU, padding='same') ──► BatchNorm ──► MaxPooling2D (2x2)
   │
   ├── Conv2D (64 filtros 3x3, ReLU, padding='same') ──► BatchNorm ──► MaxPooling2D (2x2)
   │
   ├── Conv2D (128 filtros 3x3, ReLU, padding='same') ──► BatchNorm ──► MaxPooling2D (2x2)
   │
   ├── GlobalAveragePooling2D()
   │
   ├── Dense (128 unidades, ReLU) ──► Dropout (0.4)
   │
   └── Dense (4 unidades, Softmax)
