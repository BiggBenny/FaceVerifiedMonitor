"""
Gesichtsvergleich mit ArcFace-Embeddings.
- Gesichtsdetektion: SCRFD det_10g (insightface, ONNX via onnxruntime)
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
# Modell-Pfade
# ---------------------------------------------------------------------------

MODEL_DIR    = Path(__file__).parent / "models"
SCRFD_PATH   = MODEL_DIR / "det_10g.onnx"
ARCFACE_PATH = MODEL_DIR / "w600k_r50.onnx"

BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)


# ---------------------------------------------------------------------------
# Download: beide Modelle aus buffalo_l.zip
# ---------------------------------------------------------------------------

def ensure_models(progress_cb=None):
    MODEL_DIR.mkdir(exist_ok=True)

    if SCRFD_PATH.exists() and ARCFACE_PATH.exists():
        return

    zip_path = MODEL_DIR / "buffalo_l.zip"

    if not zip_path.exists():
        if progress_cb:
            progress_cb("Lade buffalo_l.zip herunter (~330 MB)…")
        urlretrieve(BUFFALO_L_URL, zip_path)

    if progress_cb:
        progress_cb("Entpacke Modelle…")

    need = {
        "det_10g.onnx":   SCRFD_PATH,
        "w600k_r50.onnx": ARCFACE_PATH,
    }
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            filename = Path(entry).name
            if filename in need and not need[filename].exists():
                need[filename].write_bytes(z.read(entry))

    zip_path.unlink(missing_ok=True)

    missing = [k for k, v in need.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Nicht in buffalo_l.zip gefunden: {missing}")


# ---------------------------------------------------------------------------
# Eigene Vergleichsfunktionen (kein Framework-Vergleich)
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """cos(θ) = (a · b) / (‖a‖ · ‖b‖)  →  [-1, 1]"""
    dot   = float(np.dot(emb1, emb2))
    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return dot / denom if denom > 0 else 0.0


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.linalg.norm(emb1 - emb2))


def similarity_to_percent(cos_sim: float) -> float:
    return (cos_sim + 1.0) / 2.0 * 100.0


def interpret(cos_sim: float) -> tuple[str, str]:
    """Schwellwerte auf Basis des ArcFace-LFW-Benchmarks (EER ≈ 0.28)."""
    if cos_sim >= 0.50:
        return "Sehr wahrscheinlich dieselbe Person", "#2ecc71"
    if cos_sim >= 0.30:
        return "Möglicherweise dieselbe Person", "#f39c12"
    if cos_sim >= 0.10:
        return "Geringe Ähnlichkeit / unsicher", "#e67e22"
    return "Wahrscheinlich verschiedene Personen", "#e74c3c"


# ---------------------------------------------------------------------------
# SCRFD-Detektor (det_10g.onnx)
# Strides [8, 16, 32], 2 Anchors/Location, 5 Landmarks
# Ausgabe-Reihenfolge (insightface-Standard):
#   [0-2] Score  je Stride  (1, N, 1)
#   [3-5] BBox   je Stride  (1, N, 4)
#   [6-8] Kps    je Stride  (1, N, 10)
# ---------------------------------------------------------------------------

class SCRFDDetector:
    _STRIDES      = [8, 16, 32]
    _NUM_ANCHORS  = 2
    _INPUT_SIZE   = 640

    def __init__(self, model_path: str):
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name  = self._sess.get_inputs()[0].name
        self._output_names = [o.name for o in self._sess.get_outputs()]

    # ---- Vorverarbeitung ------------------------------------------------

    def _letterbox(self, img: np.ndarray):
        """Skaliert proportional auf INPUT_SIZE×INPUT_SIZE (oben-links ausgerichtet)."""
        h, w = img.shape[:2]
        scale = self._INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE, 3), dtype=np.float32)
        canvas[:nh, :nw] = resized.astype(np.float32)
        return canvas, scale

    def _preprocess(self, img_bgr: np.ndarray):
        canvas, scale = self._letterbox(img_bgr)
        # BGR → RGB, [-1, 1], NCHW
        blob = (canvas[:, :, ::-1] - 127.5) / 128.0
        blob = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        return blob, scale

    # ---- Anchor-Erzeugung -----------------------------------------------

    @staticmethod
    def _anchor_centers(stride: int, num_anchors: int) -> np.ndarray:
        """Gitter-Ankerpunkte für einen Stride (kein Offset)."""
        size  = SCRFDDetector._INPUT_SIZE // stride
        grid  = np.stack(np.mgrid[:size, :size][::-1], axis=-1).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)          # (size², 2)
        return np.tile(centers, (1, num_anchors)).reshape(-1, 2)  # interleaved

    # ---- Dekodierung ----------------------------------------------------

    @staticmethod
    def _decode_bbox(centers: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        BBox-Dekodierung: Ankerpunkt ± Abstandswert → [x1, y1, x2, y2].
        """
        x1 = centers[:, 0] - deltas[:, 0]
        y1 = centers[:, 1] - deltas[:, 1]
        x2 = centers[:, 0] + deltas[:, 2]
        y2 = centers[:, 1] + deltas[:, 3]
        return np.stack([x1, y1, x2, y2], axis=1)

    @staticmethod
    def _decode_kps(centers: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """
        Landmark-Dekodierung: 5 Punkte (x, y) relativ zum Ankerpunkt.
        """
        pts = []
        for i in range(0, 10, 2):
            pts.append(centers[:, 0] + deltas[:, i])
            pts.append(centers[:, 1] + deltas[:, i + 1])
        return np.stack(pts, axis=1).reshape(-1, 5, 2)

    # ---- NMS ------------------------------------------------------------

    @staticmethod
    def _nms(bboxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
        """IoU-basiertes Greedy-NMS, gibt beibehaltene Indizes zurück."""
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep  = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            ix1 = np.maximum(x1[i], x1[order[1:]])
            iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]])
            iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            order = order[1:][iou <= iou_thresh]
        return np.array(keep, dtype=np.int64)

    # ---- Haupt-Interface ------------------------------------------------

    def detect(
        self,
        img_bgr: np.ndarray,
        score_thresh: float = 0.5,
        nms_thresh:   float = 0.4,
    ) -> list[dict]:
        """
        Gibt eine Liste von Dicts zurück:
          {'bbox': [x1,y1,x2,y2], 'score': float, 'kps': np.ndarray(5,2)}
        Koordinaten sind auf das Original-Bild skaliert.
        """
        blob, scale = self._preprocess(img_bgr)
        outs = self._sess.run(self._output_names, {self._input_name: blob})

        fmc = len(self._STRIDES)          # 3
        all_scores, all_bboxes, all_kps = [], [], []

        for i, stride in enumerate(self._STRIDES):
            scores  = outs[i].reshape(-1)           # (N,)
            bdeltas = outs[i + fmc].reshape(-1, 4) * stride
            kdeltas = outs[i + fmc * 2].reshape(-1, 10) * stride

            centers = self._anchor_centers(stride, self._NUM_ANCHORS)

            mask    = scores >= score_thresh
            if not mask.any():
                continue

            all_scores.append(scores[mask])
            all_bboxes.append(self._decode_bbox(centers[mask], bdeltas[mask]))
            all_kps.append(self._decode_kps(centers[mask], kdeltas[mask]))

        if not all_scores:
            return []

        scores  = np.concatenate(all_scores)
        bboxes  = np.concatenate(all_bboxes)
        kps_all = np.concatenate(all_kps)

        keep = self._nms(bboxes, scores, nms_thresh)

        # Skalierung zurück auf Original-Bild
        inv = 1.0 / scale
        results = []
        for k in keep:
            results.append({
                "bbox":  (bboxes[k] * inv).tolist(),
                "score": float(scores[k]),
                "kps":   kps_all[k] * inv,
            })
        return results


# ---------------------------------------------------------------------------
# ArcFace-Inferenz (w600k_r50.onnx)
# ---------------------------------------------------------------------------

class ArcFaceONNX:
    # Standard-Referenz-Landmarks für 112×112 (insightface arcface_dst)
    _REF_KPS = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    def __init__(self, model_path: str):
        self._sess = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._sess.get_inputs()[0].name

    def _align(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Affine Ausrichtung auf 112×112 anhand der 5 Landmarks."""
        M, _ = cv2.estimateAffinePartial2D(
            kps.astype(np.float32), self._REF_KPS, method=cv2.LMEDS
        )
        return cv2.warpAffine(img_bgr, M, (112, 112))

    def get_embedding(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """L2-normierter 512-dim ArcFace-Vektor."""
        face   = self._align(img_bgr, kps)
        face_f = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob   = ((face_f - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]
        emb    = self._sess.run(None, {self._input_name: blob})[0][0]
        norm   = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


# ---------------------------------------------------------------------------
# Pipeline: Bild → Embedding
# ---------------------------------------------------------------------------

def detect_and_embed(
    detector: SCRFDDetector,
    arcface:  ArcFaceONNX,
    image_path: str,
) -> np.ndarray | None:
    """Größtes Gesicht im Bild → ArcFace-Embedding. None wenn kein Gesicht."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    faces = detector.detect(img, score_thresh=0.5)
    if not faces:
        return None

    best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) *
                                     (f["bbox"][3] - f["bbox"][1]))
    return arcface.get_embedding(img, best["kps"])


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

PREVIEW_W, PREVIEW_H = 300, 300
BG    = "#1e1e2e"
PANEL = "#2a2a3e"
ACCENT = "#7c3aed"
TEXT  = "#cdd6f4"
GRAY  = "#585b70"


class FaceCompareApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ArcFace Gesichtsvergleich")
        self.resizable(False, False)
        self.configure(bg=BG)

        self.detector: SCRFDDetector | None = None
        self.arcface:  ArcFaceONNX  | None = None
        self.paths      = [None, None]
        self.embeddings = [None, None]
        self._photos    = [None, None]

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

        row = tk.Frame(self, bg=BG)
        row.pack(padx=20)

        self._canvases:    list[tk.Canvas] = []
        self._status_lbls: list[tk.Label]  = []
        self._load_btns:   list[tk.Button] = []

        for i in range(2):
            col = tk.Frame(row, bg=PANEL)
            col.grid(row=0, column=i * 2, padx=10, pady=4)

            c = tk.Canvas(col, width=PREVIEW_W, height=PREVIEW_H,
                          bg=PANEL, highlightthickness=0)
            c.pack(padx=2, pady=2)
            c.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                          text=f"Bild {i+1}", fill=GRAY,
                          font=("Segoe UI", 14))
            self._canvases.append(c)

            btn = tk.Button(
                col, text=f"  Bild {i+1} laden  ",
                font=("Segoe UI", 10, "bold"),
                bg=ACCENT, fg="white",
                activebackground="#6d28d9", activeforeground="white",
                relief="flat", cursor="hand2", padx=8, pady=6,
                state="disabled",
                command=lambda idx=i: self._load_image(idx),
            )
            btn.pack(pady=(0, 4))
            self._load_btns.append(btn)

            lbl = tk.Label(col, text="–", font=("Segoe UI", 9),
                           bg=PANEL, fg=GRAY)
            lbl.pack(pady=(0, 8))
            self._status_lbls.append(lbl)

            if i == 0:
                tk.Label(row, text="vs.", font=("Segoe UI", 20, "bold"),
                         bg=BG, fg=GRAY).grid(row=0, column=1, padx=6)

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

        res = tk.Frame(self, bg=PANEL)
        res.pack(padx=20, pady=(0, 16), fill="x")

        self._result_lbl = tk.Label(
            res, text="", font=("Segoe UI", 13, "bold"),
            bg=PANEL, fg=TEXT, pady=8,
        )
        self._result_lbl.pack()

        self._detail_lbl = tk.Label(
            res, text="", font=("Segoe UI", 10),
            bg=PANEL, fg=GRAY, pady=4,
        )
        self._detail_lbl.pack()

        self._footer = tk.Label(
            self, text="Modelle werden geladen…",
            font=("Segoe UI", 9), bg=BG, fg=GRAY,
        )
        self._footer.pack(pady=(0, 10))

    # ------------------------------------------------------------------
    # Modelle laden (Hintergrund-Thread)
    # ------------------------------------------------------------------

    def _load_models_threaded(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _enable_load_buttons(self):
        for btn in self._load_btns:
            btn.config(state="normal")

    def _set_footer(self, text, color=GRAY):
        self._footer.config(text=text, fg=color)

    def _load_models(self):
        try:
            self.after(0, self._set_footer, "Lade Modelle… (1x Download ~330 MB)")
            ensure_models(progress_cb=lambda m: self.after(0, self._set_footer, m))

            self.after(0, self._set_footer, "Initialisiere SCRFD-Detektor…")
            self.detector = SCRFDDetector(str(SCRFD_PATH))

            self.after(0, self._set_footer, "Initialisiere ArcFace…")
            self.arcface = ArcFaceONNX(str(ARCFACE_PATH))

            self.after(0, self._set_footer, "Bereit.")
            self.after(0, self._enable_load_buttons)
        except Exception as e:
            self.after(0, self._set_footer, f"Fehler: {e}", "#e74c3c")
            self.after(0, messagebox.showerror, "Modellfehler", str(e))

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

        self.paths[idx]      = path
        self.embeddings[idx] = None
        self._status_lbls[idx].config(text="Verarbeite…", fg=GRAY)
        self._show_preview(idx, path)
        self._clear_result()
        self.update()

        if self.detector is None or self.arcface is None:
            self._status_lbls[idx].config(text="Modell noch nicht bereit", fg="#e74c3c")
            return

        try:
            emb = detect_and_embed(self.detector, self.arcface, path)
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
    # Vergleich
    # ------------------------------------------------------------------

    def _compare(self):
        emb1, emb2 = self.embeddings
        if emb1 is None or emb2 is None:
            return

        cos_sim  = cosine_similarity(emb1, emb2)
        euc_dist = euclidean_distance(emb1, emb2)
        pct      = similarity_to_percent(cos_sim)
        verdict, color = interpret(cos_sim)

        self._result_lbl.config(text=f"{verdict}  —  {pct:.1f} %", fg=color)
        self._detail_lbl.config(
            text=(
                f"Kosinus-Ähnlichkeit: {cos_sim:+.4f}   |   "
                f"Euklidischer Abstand: {euc_dist:.4f}   |   "
                f"Dim: {len(emb1)}"
            ),
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceCompareApp()
    app.mainloop()
