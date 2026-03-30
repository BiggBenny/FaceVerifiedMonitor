"""
Gesichtsvergleich mit ArcFace-Embeddings.
- Gesichtsdetektion: YuNet (OpenCV)
- Embeddings:        ArcFace w600k_r50 (ONNX via onnxruntime)
- Vergleich:         Eigene Kosinus-Ähnlichkeit (kein Framework-Vergleich)
- GUI:               Tkinter
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import threading
import numpy as np
import cv2
import onnxruntime as ort
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Modell-Download-Pfade
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).parent / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
ARCFACE_PATH = MODEL_DIR / "w600k_r50.onnx"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
# buffalo_l-Zip enthält w600k_r50.onnx (ArcFace ResNet50)
BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)


# ---------------------------------------------------------------------------
# Eigene Vergleichsfunktionen (keine Framework-Methode verwendet)
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Kosinus-Ähnlichkeit zweier Vektoren.
    cos(θ) = (a · b) / (||a|| · ||b||)
    Rückgabe: Wert in [-1, 1], je höher desto ähnlicher.
    """
    dot = float(np.dot(emb1, emb2))
    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return dot / denom if denom > 0 else 0.0


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Euklidischer Abstand (kleiner = ähnlicher)."""
    return float(np.sqrt(np.sum((emb1 - emb2) ** 2)))


def similarity_to_percent(cos_sim: float) -> float:
    """Kosinus-Ähnlichkeit [-1, 1] → Prozentwert [0, 100]."""
    return (cos_sim + 1.0) / 2.0 * 100.0


def interpret(cos_sim: float) -> tuple[str, str]:
    """
    Schwellwert basiert auf ArcFace LFW-Benchmark (EER ≈ 0.28).
    Gibt (Text, Farbe) zurück.
    """
    if cos_sim >= 0.50:
        return "Sehr wahrscheinlich dieselbe Person", "#2ecc71"
    if cos_sim >= 0.30:
        return "Möglicherweise dieselbe Person", "#f39c12"
    if cos_sim >= 0.10:
        return "Geringe Ähnlichkeit / unsicher", "#e67e22"
    return "Wahrscheinlich verschiedene Personen", "#e74c3c"


# ---------------------------------------------------------------------------
# Modell-Verwaltung
# ---------------------------------------------------------------------------

def ensure_yunet() -> str:
    MODEL_DIR.mkdir(exist_ok=True)
    if not YUNET_PATH.exists():
        print("Lade YuNet herunter…")
        urlretrieve(YUNET_URL, YUNET_PATH)
    return str(YUNET_PATH)


def ensure_arcface(progress_cb=None) -> str:
    MODEL_DIR.mkdir(exist_ok=True)
    if ARCFACE_PATH.exists():
        return str(ARCFACE_PATH)

    if progress_cb:
        progress_cb("ArcFace-Modell wird heruntergeladen (~330 MB)…")

    print("Lade buffalo_l.zip herunter…")
    zip_path = MODEL_DIR / "buffalo_l.zip"
    urlretrieve(BUFFALO_L_URL, zip_path)

    print("Entpacke w600k_r50.onnx…")
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith("w600k_r50.onnx"):
                data = z.read(name)
                ARCFACE_PATH.write_bytes(data)
                break
    zip_path.unlink(missing_ok=True)

    if not ARCFACE_PATH.exists():
        raise FileNotFoundError("w600k_r50.onnx nicht in buffalo_l.zip gefunden.")
    return str(ARCFACE_PATH)


# ---------------------------------------------------------------------------
# ArcFace-Inferenz via ONNX (ohne insightface-Vergleichsmethode)
# ---------------------------------------------------------------------------

class ArcFaceONNX:
    """Minimaler Wrapper um das ArcFace ONNX-Modell."""

    # Referenz-Landmarks für die Gesichts-Ausrichtung (112×112)
    _REFERENCE_LANDMARKS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    def __init__(self, model_path: str):
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def _align_face(self, img_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        """Affine Transformation anhand der 5 Landmark-Punkte auf 112×112."""
        src = landmarks5.astype(np.float32)
        dst = self._REFERENCE_LANDMARKS
        transform = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(img_bgr, transform, (112, 112))
        return aligned

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """BGR → RGB, normalisiert auf [-1, 1], NCHW-Format."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_f = (face_rgb.astype(np.float32) - 127.5) / 128.0
        return face_f.transpose(2, 0, 1)[np.newaxis]  # (1, 3, 112, 112)

    def get_embedding(self, img_bgr: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
        """Gibt L2-normierten 512-dim Embedding-Vektor zurück."""
        aligned = self._align_face(img_bgr, landmarks5)
        blob = self._preprocess(aligned)
        output = self._session.run(None, {self._input_name: blob})[0][0]
        norm = np.linalg.norm(output)
        return output / norm if norm > 0 else output


# ---------------------------------------------------------------------------
# Gesichtserkennung (YuNet) + Embedding-Extraktion
# ---------------------------------------------------------------------------

def detect_and_embed(
    yunet: cv2.FaceDetectorYN,
    arcface: ArcFaceONNX,
    image_path: str,
) -> np.ndarray | None:
    """
    Erkennt das größte Gesicht im Bild und gibt sein ArcFace-Embedding zurück.
    Gibt None zurück, wenn kein Gesicht gefunden wurde.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    h, w = img.shape[:2]
    yunet.setInputSize((w, h))
    _, faces = yunet.detect(img)

    if faces is None or len(faces) == 0:
        return None

    # Größtes Gesicht (höchste Bounding-Box-Fläche)
    best = max(faces, key=lambda f: f[2] * f[3])

    # Landmarks: Spalten 4–13 → 5 Punkte (x,y)
    landmarks = best[4:14].reshape(5, 2)
    return arcface.get_embedding(img, landmarks)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

PREVIEW_W, PREVIEW_H = 300, 300
BG   = "#1e1e2e"
PANEL = "#2a2a3e"
ACCENT = "#7c3aed"
TEXT = "#cdd6f4"
GRAY = "#585b70"


class FaceCompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ArcFace Gesichtsvergleich")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.yunet: cv2.FaceDetectorYN | None = None
        self.arcface: ArcFaceONNX | None = None
        self.paths = [None, None]
        self.embeddings = [None, None]
        self._photos = [None, None]  # tk.PhotoImage-Referenzen

        self._build_ui()
        self.after(150, self._load_models_threaded)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        tk.Label(
            self, text="ArcFace Gesichtsvergleich",
            font=("Segoe UI", 16, "bold"), bg=BG, fg=TEXT,
        ).pack(pady=(18, 2))

        tk.Label(
            self, text="Lade zwei Bilder und klicke »Vergleichen«",
            font=("Segoe UI", 10), bg=BG, fg=GRAY,
        ).pack(pady=(0, 14))

        # ---- Bildpanels ----
        row = tk.Frame(self, bg=BG)
        row.pack(padx=20)

        self._canvases: list[tk.Canvas] = []
        self._status_lbls: list[tk.Label] = []

        for i in range(2):
            col = tk.Frame(row, bg=PANEL)
            col.grid(row=0, column=i * 2, padx=10, pady=4)

            c = tk.Canvas(col, width=PREVIEW_W, height=PREVIEW_H,
                          bg=PANEL, highlightthickness=0)
            c.pack(padx=2, pady=2)
            c.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                          text=f"Bild {i+1}", fill=GRAY,
                          font=("Segoe UI", 14), tags="placeholder")
            self._canvases.append(c)

            tk.Button(
                col, text=f"  Bild {i+1} laden  ",
                font=("Segoe UI", 10, "bold"),
                bg=ACCENT, fg="white",
                activebackground="#6d28d9", activeforeground="white",
                relief="flat", cursor="hand2", padx=8, pady=6,
                command=lambda idx=i: self._load_image(idx),
            ).pack(pady=(0, 4))

            lbl = tk.Label(col, text="–", font=("Segoe UI", 9),
                           bg=PANEL, fg=GRAY)
            lbl.pack(pady=(0, 8))
            self._status_lbls.append(lbl)

            if i == 0:
                tk.Label(row, text="vs.", font=("Segoe UI", 20, "bold"),
                         bg=BG, fg=GRAY).grid(row=0, column=1, padx=6)

        # ---- Vergleiche-Button ----
        self._cmp_btn = tk.Button(
            self, text="  Gesichter vergleichen  ",
            font=("Segoe UI", 12, "bold"),
            bg=ACCENT, fg="white",
            activebackground="#6d28d9", activeforeground="white",
            relief="flat", cursor="hand2", padx=16, pady=10,
            state="disabled",
            command=self._compare,
        )
        self._cmp_btn.pack(pady=(18, 8))

        # ---- Ergebnis-Box ----
        res_frame = tk.Frame(self, bg=PANEL)
        res_frame.pack(padx=20, pady=(0, 16), fill="x")

        self._result_lbl = tk.Label(
            res_frame, text="", font=("Segoe UI", 13, "bold"),
            bg=PANEL, fg=TEXT, pady=8,
        )
        self._result_lbl.pack()

        self._detail_lbl = tk.Label(
            res_frame, text="", font=("Segoe UI", 10),
            bg=PANEL, fg=GRAY, pady=4,
        )
        self._detail_lbl.pack()

        # ---- Footer-Status ----
        self._footer = tk.Label(
            self, text="Modelle werden geladen…",
            font=("Segoe UI", 9), bg=BG, fg=GRAY,
        )
        self._footer.pack(pady=(0, 10))

    # ------------------------------------------------------------------
    # Modelle laden
    # ------------------------------------------------------------------

    def _load_models_threaded(self):
        t = threading.Thread(target=self._load_models, daemon=True)
        t.start()

    def _load_models(self):
        try:
            self._footer.config(text="Lade YuNet…")
            self.update()
            yunet_path = ensure_yunet()
            self.yunet = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320),
                score_threshold=0.7,
                nms_threshold=0.3,
                top_k=5000,
            )

            self._footer.config(
                text="Lade ArcFace ONNX-Modell… (1x Download ~330 MB)"
            )
            self.update()
            arcface_path = ensure_arcface(
                progress_cb=lambda msg: self._footer.config(text=msg)
            )
            self.arcface = ArcFaceONNX(arcface_path)
            self._footer.config(text="Bereit.")
        except Exception as e:
            self._footer.config(text=f"Fehler beim Laden: {e}", fg="#e74c3c")
            messagebox.showerror("Modellfehler", str(e))

    # ------------------------------------------------------------------
    # Bild laden
    # ------------------------------------------------------------------

    def _load_image(self, idx: int):
        path = filedialog.askopenfilename(
            title=f"Bild {idx+1} auswählen",
            filetypes=[
                ("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("Alle", "*.*"),
            ],
        )
        if not path:
            return

        self.paths[idx] = path
        self.embeddings[idx] = None
        self._status_lbls[idx].config(text="Verarbeite…", fg=GRAY)
        self._show_preview(idx, path)
        self._clear_result()
        self.update()

        if self.arcface is None or self.yunet is None:
            self._status_lbls[idx].config(text="Modell noch nicht bereit", fg="#e74c3c")
            return

        try:
            emb = detect_and_embed(self.yunet, self.arcface, path)
            if emb is None:
                self._status_lbls[idx].config(text="Kein Gesicht gefunden", fg="#e74c3c")
            else:
                self.embeddings[idx] = emb
                self._status_lbls[idx].config(text="Gesicht erkannt  ✓", fg="#2ecc71")
        except Exception as e:
            self._status_lbls[idx].config(text=f"Fehler: {e}", fg="#e74c3c")

        self._refresh_cmp_btn()

    def _show_preview(self, idx: int, path: str):
        img = Image.open(path)
        img.thumbnail((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._photos[idx] = photo
        c = self._canvases[idx]
        c.delete("all")
        c.create_image(PREVIEW_W // 2, PREVIEW_H // 2, anchor="center", image=photo)

    def _refresh_cmp_btn(self):
        ready = all(e is not None for e in self.embeddings)
        self._cmp_btn.config(state="normal" if ready else "disabled")

    def _clear_result(self):
        self._result_lbl.config(text="", fg=TEXT)
        self._detail_lbl.config(text="")

    # ------------------------------------------------------------------
    # Vergleich — eigene Implementierung, kein Framework-Vergleich
    # ------------------------------------------------------------------

    def _compare(self):
        emb1, emb2 = self.embeddings
        if emb1 is None or emb2 is None:
            return

        cos_sim  = cosine_similarity(emb1, emb2)
        euc_dist = euclidean_distance(emb1, emb2)
        pct      = similarity_to_percent(cos_sim)
        verdict, color = interpret(cos_sim)

        self._result_lbl.config(
            text=f"{verdict}  —  {pct:.1f} %",
            fg=color,
        )
        self._detail_lbl.config(
            text=(
                f"Kosinus-Ähnlichkeit: {cos_sim:+.4f}   |   "
                f"Euklidischer Abstand: {euc_dist:.4f}   |   "
                f"Embedding-Dim: {len(emb1)}"
            ),
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceCompareApp()
    app.mainloop()
