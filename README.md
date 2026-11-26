# Inspección automática de defectos en acero laminado
**Visión Artificial – ITM · Entregas 4 y 5**
Docente: Cristian Mateo Hernández González · Integrantes: Santiago Garcia Gomez y Tomas Lopez Perez

Clasificación de 6 tipos de defectos superficiales en bandas de acero laminado en caliente (NEU Surface Defect Database) con:
- **Entrega 4:** MobileNetV2 congelada como extractor de características + **AutoGluon** como cabeza de decisión.
- **Entrega 5:** **Destilación de conocimiento por logits** hacia una CNN liviana, exportada a TensorFlow Lite para una Raspberry Pi 5, con demo en tiempo real.

## Resultados (270 imágenes de prueba)
| Modelo | Accuracy | F1 macro | Tamaño | Latencia (CPU) |
|---|---|---|---|---|
| HOG + SVM (clásico) | 88,5 % | 0,883 | – | – |
| MobileNetV2 + AutoGluon (E4) | 98,5 % | 0,985 | 32 MB | ~32 ms |
| Profesor TL (TFLite) | 98,1 % | 0,982 | 8,9 MB | 2,8 ms |
| Alumno sin KD (TFLite) | 97,0 % | 0,971 | 0,32 MB | 0,59 ms |
| Alumno con KD (TFLite int8) | 95,6 % | 0,956 | 89 KB | 0,27 ms |

En este dataset la destilación no superó al mismo alumno entrenado solo con etiquetas; el análisis está en el notebook de la Entrega 5.

## Estructura
- `Entrega4_TL_AutoGluon.ipynb`: tf.data, extractor MobileNetV2, DataFrame f_0…f_1279, AutoGluon, leaderboard, matriz de confusión, reporte, ROC.
- `Entrega5_Destilacion_KD.ipynb`: Profesor, Alumno, KD por logits, comparativa TL vs KD, TFLite, hardware y cámara.
- `demo_tiempo_real.py`: demo con webcam o flujo simulado (clase, confianza, latencia y FPS).
- `dataset_NEU/`: 1.800 imágenes PNG, una carpeta por clase.
- `salidas_E4/`, `salidas_E5/`: figuras, métricas y modelos (.keras, .tflite).

## Cómo ejecutar (Anaconda)
```
conda create -n vision python=3.11 -y
conda activate vision
pip install -r requirements.txt
jupyter notebook
```
Ejecutar primero la Entrega 4 y luego la Entrega 5. Demo: `python demo_tiempo_real.py` (o `--fuente carpeta` para flujo simulado).

## Dataset
NEU Surface Defect Database — K. Song y Y. Yan, Northeastern University. 1.800 imágenes en gris de 200×200 px, 300 por clase. Copia pública usada: https://github.com/mahesh-sudhakar/NEU_XAI

## Licencia
Código bajo licencia MIT (ver `LICENSE`). Las imágenes del dataset pertenecen a sus autores originales y se incluyen solo con fines académicos.
