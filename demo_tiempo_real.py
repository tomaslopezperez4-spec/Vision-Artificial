"""
Demo en tiempo real – Entrega 5 Visión Artificial (ITM)
Clasificación de defectos en acero (NEU) con el Alumno destilado (TFLite).

Uso:
    python demo_tiempo_real.py                         # webcam 0
    python demo_tiempo_real.py --camara 1              # otra webcam
    python demo_tiempo_real.py --fuente carpeta        # flujo simulado con imágenes de prueba
    python demo_tiempo_real.py --fuente video --video mi_video.mp4
    python demo_tiempo_real.py --modelo salidas_E5/profesor_tl.tflite   # comparar con el Profesor

Teclas: q = salir | espacio = pausar | g = guardar captura
"""
import argparse
import json
import os
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd

try:                                   # en Raspberry Pi: pip install ai-edge-litert (o tflite-runtime)
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

try:                                   # Pillow permite dibujar tildes y ñ (OpenCV no puede)
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None

COLORES = {  # BGR
    "crazing": (0, 165, 255), "inclusion": (0, 0, 255), "patches": (255, 0, 255),
    "pitted_surface": (0, 255, 255), "rolled-in_scale": (255, 128, 0), "scratches": (0, 255, 0),
}

# Nombres en español que se muestran en pantalla (las claves internas siguen siendo las del dataset)
NOMBRES_ES = {
    "crazing": "Agrietamiento", "inclusion": "Inclusión", "patches": "Manchas",
    "pitted_surface": "Superficie picada", "rolled-in_scale": "Cascarilla laminada", "scratches": "Rayones",
}

MODELOS_ES = {
    "alumno_kd_int8.tflite": "Alumno con KD (int8)", "alumno_kd.tflite": "Alumno con KD",
    "alumno_sin_kd.tflite": "Alumno sin KD", "profesor_tl.tflite": "Profesor TL",
}


def _fuente(tam):
    for f in ("arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, tam)
        except OSError:
            continue
    return None


_CACHE_FUENTES = {}


def poner_texto(frame, textos):
    """Dibuja una lista de (texto, (x, y_base), tamaño_px, color_BGR) admitiendo tildes y ñ."""
    fuentes = {}
    if Image is not None:
        for _, _, tam, _ in textos:
            if tam not in _CACHE_FUENTES:
                _CACHE_FUENTES[tam] = _fuente(tam)
            fuentes[tam] = _CACHE_FUENTES[tam]
    if Image is None or any(f is None for f in fuentes.values()):
        import unicodedata               # respaldo: OpenCV sin tildes
        for t, (x, y), tam, color in textos:
            t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
            cv2.putText(frame, t, (x, y), cv2.FONT_HERSHEY_SIMPLEX, tam / 32, color, max(1, tam // 16))
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for t, (x, y), tam, color in textos:
        d.text((x, y), t, font=fuentes[tam], fill=color[::-1], anchor="ls")
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return frame


def argumentos():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo", default="salidas_E5/alumno_kd_int8.tflite")
    p.add_argument("--fuente", choices=["camara", "carpeta", "video"], default="camara")
    p.add_argument("--camara", type=int, default=0)
    p.add_argument("--video", default="")
    p.add_argument("--fps_carpeta", type=float, default=4.0, help="velocidad del flujo simulado")
    p.add_argument("--umbral", type=float, default=0.60, help="confianza mínima para mostrar la clase")
    return p.parse_args()


class Clasificador:
    def __init__(self, ruta_modelo, clases):
        self.interp = Interpreter(model_path=ruta_modelo, num_threads=4)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]
        _, self.h, self.w, self.c = self.inp["shape"]
        self.clases = clases
        self.es_profesor = self.c == 3   # el Profesor espera 224x224x3 en [-1,1]

    def preprocesar(self, gris):
        img = cv2.resize(gris, (self.w, self.h), interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.es_profesor:
            img = np.repeat(img[..., None], 3, axis=-1) / 127.5 - 1.0
        else:
            img = img[..., None] / 255.0
        return img[None]

    def predecir(self, gris):
        t0 = time.perf_counter()
        self.interp.set_tensor(self.inp["index"], self.preprocesar(gris))
        self.interp.invoke()
        logits = self.interp.get_tensor(self.out["index"])[0]
        ms = (time.perf_counter() - t0) * 1000
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        return p, ms


def fuente_frames(args):
    if args.fuente == "carpeta":
        rutas = pd.read_csv("splits.csv").query("split == 'test'").ruta.sample(frac=1, random_state=0).tolist()
        while True:
            for r in rutas:
                img = cv2.imread(r)
                yield cv2.resize(img, (480, 480), interpolation=cv2.INTER_NEAREST), os.path.basename(os.path.dirname(r))
    else:
        cap = cv2.VideoCapture(args.video if args.fuente == "video" else args.camara)
        if not cap.isOpened():
            raise SystemExit("No se pudo abrir la cámara/video. Pruebe --camara 1 o --fuente carpeta")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.fuente == "video":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            yield frame, None


def main():
    args = argumentos()
    clases = json.load(open("clases.json"))
    clf = Clasificador(args.modelo, clases)
    nombre_modelo = MODELOS_ES.get(os.path.basename(args.modelo), os.path.basename(args.modelo))
    print(f"Modelo: {nombre_modelo} | entrada {clf.h}x{clf.w}x{clf.c}")

    hist_fps, hist_p = deque(maxlen=30), deque(maxlen=5)   # suavizado de FPS y de probabilidades
    t_prev, pausa, n_guardadas = time.perf_counter(), False, 0
    aciertos = total = 0
    frame = None

    for frame, real in fuente_frames(args):
        h, w = frame.shape[:2]
        lado = int(min(h, w) * 0.6) if real is None else min(h, w)   # ROI cuadrada central
        y0, x0 = (h - lado) // 2, (w - lado) // 2
        roi = cv2.cvtColor(frame[y0:y0 + lado, x0:x0 + lado], cv2.COLOR_BGR2GRAY)

        p, ms = clf.predecir(roi)
        hist_p.append(p)
        p_suave = np.mean(hist_p, axis=0) if real is None else p
        k = int(p_suave.argmax())
        clase, conf = clases[k], float(p_suave[k])

        ahora = time.perf_counter()
        hist_fps.append(1.0 / max(ahora - t_prev, 1e-6))
        t_prev = ahora

        color = COLORES.get(clase, (255, 255, 255)) if conf >= args.umbral else (180, 180, 180)
        texto = f"{NOMBRES_ES.get(clase, clase)}  {conf * 100:.0f} %" if conf >= args.umbral else "Incierto"
        cv2.rectangle(frame, (x0, y0), (x0 + lado, y0 + lado), color, 3)
        cv2.rectangle(frame, (0, 0), (w, 100), (30, 30, 30), -1)
        textos = [
            (texto, (15, 40), 32, color),
            (f"Inferencia: {ms:.1f} ms | Cuadros/s: {np.mean(hist_fps):.1f}", (15, 68), 17, (255, 255, 255)),
            (f"Modelo: {nombre_modelo}", (15, 92), 17, (200, 200, 200)),
        ]
        if real is not None:
            total += 1
            aciertos += int(real == clase)
            textos.append((f"Clase real: {NOMBRES_ES.get(real, real)} | Aciertos acumulados: {aciertos}/{total}",
                           (15, h - 15), 18, (255, 255, 255)))
        # barras de probabilidad (nombre a la izquierda, barra a la derecha)
        for i, (c, pi) in enumerate(zip(clases, p_suave)):
            yb = 115 + i * 24
            cv2.rectangle(frame, (w - 300, yb - 2), (w - 5, yb + 20), (30, 30, 30), -1)
            cv2.rectangle(frame, (w - 130, yb), (w - 130 + int(120 * pi), yb + 16), COLORES.get(c, (200, 200, 200)), -1)
            textos.append((NOMBRES_ES.get(c, c), (w - 292, yb + 14), 15, (255, 255, 255)))
        poner_texto(frame, textos)

        cv2.imshow("Inspeccion de acero laminado", frame)
        espera = int(1000 / args.fps_carpeta) if real is not None else 1
        tecla = cv2.waitKey(0 if pausa else espera) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord(" "):
            pausa = not pausa
        if tecla == ord("g"):
            n_guardadas += 1
            cv2.imwrite(f"captura_{n_guardadas}.png", frame)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
