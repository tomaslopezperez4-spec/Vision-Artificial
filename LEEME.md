# Visión Artificial ITM – Entregas 4 y 5
**Inspección de defectos en acero laminado en caliente (dataset NEU)**

## Contenido
| Archivo | Para qué |
|---|---|
| `Entrega4_TL_AutoGluon.ipynb` | MobileNetV2 congelada + AutoGluon, leaderboard, matriz de confusión, reporte, ROC, comparación con HOG+SVM |
| `Entrega5_Destilacion_KD.ipynb` | Profesor TL → Alumno por destilación de logits, comparativa TL vs KD, TFLite, hardware y cámara |
| `demo_tiempo_real.py` | Demo en vivo con webcam (o flujo simulado) mostrando clase, latencia y FPS |
| `dataset_NEU/` | 1.800 imágenes, una carpeta por clase (300 c/u) |

## 1. Crear el entorno en Anaconda (una sola vez)
Abrir **Anaconda Prompt** y ejecutar:
```bash
conda create -n vision python=3.11 -y
conda activate vision
pip install tensorflow autogluon.tabular[all] opencv-python scikit-learn scikit-image seaborn matplotlib pandas jupyter ipykernel
python -m ipykernel install --user --name vision --display-name "Python (vision)"
```
> AutoGluon instala PyTorch (~1 GB); la instalación puede tardar 10–15 minutos.
> En Windows, si `autogluon.tabular[all]` falla, use `pip install autogluon.tabular[lightgbm,catboost,xgboost,fastai]`.

## 2. Ejecutar
```bash
conda activate vision
cd ruta\a\esta\carpeta
jupyter notebook
```
1. Abrir `Entrega4_TL_AutoGluon.ipynb`, elegir el kernel **Python (vision)** y ejecutar todo (Kernel → Restart & Run All). ~5–8 min.
   Genera `splits.csv`, `clases.json` y la carpeta `salidas_E4/`.
2. Abrir `Entrega5_Destilacion_KD.ipynb` y ejecutar todo. ~10–20 min en CPU.
   Genera `salidas_E5/` con los modelos `.keras` y `.tflite`.
3. Demo en vivo:
```bash
python demo_tiempo_real.py                   # webcam
python demo_tiempo_real.py --fuente carpeta  # si la cámara falla: flujo simulado
```
Teclas: `q` salir, `espacio` pausa, `g` guarda captura.

## Dataset
NEU Surface Defect Database (Song & Yan, Northeastern University). Copia pública usada: https://github.com/mahesh-sudhakar/NEU_XAI (carpeta "NEU surface defect database"), convertida a PNG y organizada por clase. También disponible en Kaggle ("NEU Metal Surface Defects Data").

## Para la rúbrica
- **Enlace de GitHub:** subir esta carpeta a un repositorio (sin `salidas_*/autogluon_modelo` si pesa mucho).
- **Enlace del dataset:** el de arriba, o la carpeta `dataset_NEU` en Drive.
- Los resultados pueden variar levemente en cada ejecución (AutoGluon y el entrenamiento no son 100 % deterministas); actualizar los números de la presentación con los suyos.
