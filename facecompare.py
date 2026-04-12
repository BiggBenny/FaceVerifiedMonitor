"""
ArcFace Live-Gesichtserkennung mit Personendatenbank.
- Gesichtsdetektion: SCRFD det_10g (insightface, ONNX via onnxruntime)
- Embeddings:        ArcFace w600k_r50 (ONNX via onnxruntime)
- Vergleich:         Eigene Kosinus-Ähnlichkeit
- GUI:               Tkinter – ein Fenster mit Live-Feed, DB-Panel und ID-Abschnitt
"""

import pickle
import threading
from collections import deque, Counter
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Modell-Pfade & Konstanten
# ---------------------------------------------------------------------------

MODEL_DIR    = Path(__file__).parent / "models"
SCRFD_PATH   = MODEL_DIR / "det_10g.onnx"
ARCFACE_PATH = MODEL_DIR / "w600k_r50.onnx"
DB_PATH      = Path(__file__).parent / "face_db.pkl"

BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)

FEED_W, FEED_H = 640, 480
ID_CROP        = 130   # Gesichts-Crop-Größe im ID-Abschnitt

BG     = "#1e1e2e"
PANEL  = "#2a2a3e"
CARD   = "#16162a"
ACCENT = "#7c3aed"
TEXT   = "#cdd6f4"
GRAY   = "#585b70"
GREEN  = "#2ecc71"
RED    = "#e74c3c"
ORANGE = "#f39c12"


# ---------------------------------------------------------------------------
# Download
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
    need = {"det_10g.onnx": SCRFD_PATH, "w600k_r50.onnx": ARCFACE_PATH}
    with zipfile.ZipFile(zip_path) as z:
        for entry in z.namelist():
            fname = Path(entry).name
            if fname in need and not need[fname].exists():
                need[fname].write_bytes(z.read(entry))
    zip_path.unlink(missing_ok=True)
    missing = [k for k, v in need.items() if not v.exists()]
    if missing:
        raise FileNotFoundError(f"Nicht in buffalo_l.zip gefunden: {missing}")


# ---------------------------------------------------------------------------
# Vergleich
# ---------------------------------------------------------------------------

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    dot   = float(np.dot(emb1, emb2))
    denom = float(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return dot / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# SCRFD-Detektor
# ---------------------------------------------------------------------------

class SCRFDDetector:
    _STRIDES     = [8, 16, 32]
    _NUM_ANCHORS = 2
    _INPUT_SIZE  = 640

    def __init__(self, model_path: str):
        self._sess         = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name   = self._sess.get_inputs()[0].name
        self._output_names = [o.name for o in self._sess.get_outputs()]
        scores_idx, bbox_idx, kps_idx = [], [], []
        for i, o in enumerate(self._sess.get_outputs()):
            last = o.shape[-1]
            if last == 1:   scores_idx.append(i)
            elif last == 4: bbox_idx.append(i)
            elif last == 10: kps_idx.append(i)
        self._scores_idx, self._bbox_idx, self._kps_idx = scores_idx, bbox_idx, kps_idx

    def _letterbox(self, img):
        h, w   = img.shape[:2]
        scale  = self._INPUT_SIZE / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas  = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE, 3), dtype=np.float32)
        canvas[:nh, :nw] = resized.astype(np.float32)
        return canvas, scale

    def _preprocess(self, img_bgr):
        canvas, scale = self._letterbox(img_bgr)
        blob = (canvas[:, :, ::-1] - 127.5) / 128.0
        return blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32), scale

    @staticmethod
    def _anchor_centers(stride, num_anchors):
        size    = SCRFDDetector._INPUT_SIZE // stride
        grid    = np.stack(np.mgrid[:size, :size][::-1], axis=-1).astype(np.float32)
        centers = (grid * stride).reshape(-1, 2)
        return np.tile(centers, (1, num_anchors)).reshape(-1, 2)

    @staticmethod
    def _decode_bbox(centers, deltas):
        return np.stack([
            centers[:, 0] - deltas[:, 0], centers[:, 1] - deltas[:, 1],
            centers[:, 0] + deltas[:, 2], centers[:, 1] + deltas[:, 3],
        ], axis=1)

    @staticmethod
    def _decode_kps(centers, deltas):
        pts = []
        for i in range(0, 10, 2):
            pts += [centers[:, 0] + deltas[:, i], centers[:, 1] + deltas[:, i + 1]]
        return np.stack(pts, axis=1).reshape(-1, 5, 2)

    @staticmethod
    def _nms(bboxes, scores, iou_thresh):
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order, keep = scores.argsort()[::-1], []
        while order.size > 0:
            i = order[0]; keep.append(i)
            ix1 = np.maximum(x1[i], x1[order[1:]]); iy1 = np.maximum(y1[i], y1[order[1:]])
            ix2 = np.minimum(x2[i], x2[order[1:]]); iy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
            iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
            order = order[1:][iou <= iou_thresh]
        return np.array(keep, dtype=np.int64)

    def detect(self, img_bgr, score_thresh=0.35, nms_thresh=0.4):
        blob, scale = self._preprocess(img_bgr)
        outs = self._sess.run(self._output_names, {self._input_name: blob})
        all_scores, all_bboxes, all_kps = [], [], []
        for i, stride in enumerate(self._STRIDES):
            scores  = outs[self._scores_idx[i]].reshape(-1)
            bdeltas = outs[self._bbox_idx[i]].reshape(-1, 4) * stride
            kdeltas = outs[self._kps_idx[i]].reshape(-1, 10) * stride
            centers = self._anchor_centers(stride, self._NUM_ANCHORS)
            mask    = scores >= score_thresh
            if not mask.any(): continue
            all_scores.append(scores[mask])
            all_bboxes.append(self._decode_bbox(centers[mask], bdeltas[mask]))
            all_kps.append(self._decode_kps(centers[mask], kdeltas[mask]))
        if not all_scores: return []
        scores  = np.concatenate(all_scores)
        bboxes  = np.concatenate(all_bboxes)
        kps_all = np.concatenate(all_kps)
        keep    = self._nms(bboxes, scores, nms_thresh)
        inv     = 1.0 / scale
        return [{"bbox": (bboxes[k] * inv).tolist(), "score": float(scores[k]), "kps": kps_all[k] * inv}
                for k in keep]


# ---------------------------------------------------------------------------
# ArcFace
# ---------------------------------------------------------------------------

class ArcFaceONNX:
    _REF_KPS = np.array([
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041],
    ], dtype=np.float32)

    def __init__(self, model_path: str):
        self._sess       = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self._sess.get_inputs()[0].name

    def get_embedding(self, img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
        M, _   = cv2.estimateAffinePartial2D(kps.astype(np.float32), self._REF_KPS, method=cv2.LMEDS)
        face   = cv2.warpAffine(img_bgr, M, (112, 112))
        face_f = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob   = ((face_f - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]
        emb    = self._sess.run(None, {self._input_name: blob})[0][0]
        norm   = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


# ---------------------------------------------------------------------------
# Datenbank
# ---------------------------------------------------------------------------

class FaceDatabase:
    def __init__(self, db_path: Path):
        self._path = db_path
        self._data: dict[str, list[np.ndarray]] = {}
        self._lock = threading.Lock()
        if db_path.exists():
            try:
                with open(db_path, "rb") as f:
                    self._data = pickle.load(f)
            except Exception:
                pass

    def add(self, name: str, embedding: np.ndarray):
        with self._lock:
            self._data.setdefault(name, []).append(embedding)
            self._save()

    def remove(self, name: str):
        with self._lock:
            self._data.pop(name, None)
            self._save()

    def identify(self, embedding: np.ndarray, threshold: float) -> tuple[str | None, float]:
        with self._lock:
            best_name, best_sim = None, -1.0
            for name, embs in self._data.items():
                for emb in embs:
                    s = cosine_similarity(embedding, emb)
                    if s > best_sim:
                        best_sim, best_name = s, name
        return (best_name, best_sim) if best_name and best_sim >= threshold else (None, best_sim)

    def names(self) -> list[str]:
        with self._lock: return sorted(self._data.keys())

    def count(self, name: str) -> int:
        with self._lock: return len(self._data.get(name, []))

    def _save(self):
        with open(self._path, "wb") as f: pickle.dump(self._data, f)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class FaceCheckApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ArcFace Live-Erkennung")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.detector: SCRFDDetector | None = None
        self.arcface:  ArcFaceONNX  | None = None
        self._db           = FaceDatabase(DB_PATH)
        self._threshold    = 0.45
        self._stop_event   = threading.Event()
        self._models_ready = False
        self._feed_photo   = None
        self._id_photos    = []
        # Speichert die letzten 5 ArcFace-Ergebnisse für stabiles Voting
        self._vote_history: deque = deque(maxlen=5)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(150, self._load_models_threaded)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        # Titelzeile
        tk.Label(self, text="ArcFace Live-Erkennung",
                 font=("Segoe UI", 14, "bold"), bg=BG, fg=TEXT).pack(pady=(12, 6))

        main = tk.Frame(self, bg=BG)
        main.pack(padx=16, pady=(0, 4))

        # ── Links: Live Feed ──────────────────────────────────────────
        left = tk.Frame(main, bg=PANEL)
        left.grid(row=0, column=0, padx=(0, 10), sticky="nsew")

        tk.Label(left, text="Live Feed", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=TEXT).pack(pady=(8, 4))

        self._feed_canvas = tk.Canvas(left, width=FEED_W, height=FEED_H,
                                       bg=CARD, highlightthickness=0)
        self._feed_canvas.pack(padx=8, pady=(0, 8))
        self._feed_canvas.create_text(FEED_W // 2, FEED_H // 2,
                                       text="Kamera wird gestartet…", fill=GRAY,
                                       font=("Segoe UI", 14))

        # ── Rechts: DB-Panel + ID-Abschnitt ──────────────────────────
        right = tk.Frame(main, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")

        # --- DB-Panel ---
        db_panel = tk.Frame(right, bg=PANEL)
        db_panel.pack(fill="x", pady=(0, 8))

        tk.Label(db_panel, text="Datenbank", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=TEXT).pack(pady=(8, 2), padx=12, anchor="w")

        self._db_count_lbl = tk.Label(db_panel, text=self._db_count_text(),
                                       font=("Segoe UI", 9), bg=PANEL, fg=GRAY)
        self._db_count_lbl.pack(padx=12, anchor="w")

        list_frame = tk.Frame(db_panel, bg=PANEL)
        list_frame.pack(padx=12, pady=(6, 4), fill="x")

        sb = tk.Scrollbar(list_frame, bg=PANEL)
        sb.pack(side="right", fill="y")

        self._listbox = tk.Listbox(list_frame, font=("Segoe UI", 10),
                                    bg=CARD, fg=TEXT, selectbackground=ACCENT,
                                    selectforeground="white", activestyle="none",
                                    relief="flat", bd=0, yscrollcommand=sb.set,
                                    width=28, height=8)
        self._listbox.pack(side="left", fill="x", expand=True)
        sb.config(command=self._listbox.yview)
        self._listbox.bind("<<ListboxSelect>>", self._on_listbox_select)

        btn_frame = tk.Frame(db_panel, bg=PANEL)
        btn_frame.pack(padx=12, pady=(4, 6), fill="x")

        self._add_btn = self._btn(btn_frame, "+ Person hinzufügen", self._add_person, ACCENT)
        self._add_btn.pack(fill="x", pady=(0, 3))
        self._add_btn.config(state="disabled")

        self._add_img_btn = self._btn(btn_frame, "+ Bild zu Person", self._add_image_to_person, "#16a085")
        self._add_img_btn.pack(fill="x", pady=(0, 3))
        self._add_img_btn.config(state="disabled")

        self._remove_btn = self._btn(btn_frame, "Person entfernen", self._remove_person, "#c0392b")
        self._remove_btn.pack(fill="x")
        self._remove_btn.config(state="disabled")

        # Schwelle
        thr = tk.Frame(db_panel, bg=PANEL)
        thr.pack(padx=12, pady=(8, 4), fill="x")
        tk.Label(thr, text="Erkennungsschwelle:", font=("Segoe UI", 9),
                 bg=PANEL, fg=GRAY).pack(anchor="w")
        row = tk.Frame(thr, bg=PANEL)
        row.pack(fill="x")
        self._thresh_var = tk.DoubleVar(value=self._threshold)
        tk.Scale(row, from_=0.10, to=0.90, resolution=0.05, orient="horizontal",
                 variable=self._thresh_var, command=self._on_threshold_change,
                 bg=PANEL, fg=TEXT, troughcolor=CARD, highlightthickness=0,
                 showvalue=False).pack(side="left", fill="x", expand=True)
        self._thresh_lbl = tk.Label(row, text=f"{self._threshold:.0%}",
                                     font=("Segoe UI", 9, "bold"), bg=PANEL, fg=TEXT, width=5)
        self._thresh_lbl.pack(side="left")

        self._db_status = tk.Label(db_panel, text="", font=("Segoe UI", 9),
                                    bg=PANEL, fg=GRAY, wraplength=230, justify="left")
        self._db_status.pack(padx=12, pady=(2, 8), anchor="w")

        # --- ID-Abschnitt ---
        id_panel = tk.Frame(right, bg=PANEL)
        id_panel.pack(fill="x")

        id_header = tk.Frame(id_panel, bg=PANEL)
        id_header.pack(fill="x", padx=12, pady=(8, 4))
        tk.Label(id_header, text="Erkannte Personen", font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=TEXT).pack(side="left")
        self._id_status_lbl = tk.Label(id_header, text="(alle 10 Frames)",
                                        font=("Segoe UI", 9), bg=PANEL, fg=GRAY)
        self._id_status_lbl.pack(side="left", padx=(8, 0))

        self._id_cards_frame = tk.Frame(id_panel, bg=PANEL)
        self._id_cards_frame.pack(padx=12, pady=(0, 10), fill="x")

        # Platzhalter bis erste Erkennung
        tk.Label(self._id_cards_frame, text="Warte auf Erkennung…",
                 font=("Segoe UI", 10), bg=PANEL, fg=GRAY, pady=12).pack()

        # Footer
        self._footer = tk.Label(self, text="Modelle werden geladen…",
                                 font=("Segoe UI", 9), bg=BG, fg=GRAY)
        self._footer.pack(pady=(4, 8))

        self._refresh_listbox()

    def _btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, font=("Segoe UI", 9, "bold"),
                         bg=color, fg="white", activebackground=color,
                         activeforeground="white", relief="flat",
                         cursor="hand2", pady=6, command=cmd)

    # ------------------------------------------------------------------
    # Modelle laden
    # ------------------------------------------------------------------

    def _load_models_threaded(self):
        threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self):
        try:
            self.after(0, self._set_footer, "Lade Modelle… (1x Download ~330 MB)")
            ensure_models(progress_cb=lambda m: self.after(0, self._set_footer, m))
            self.after(0, self._set_footer, "Initialisiere SCRFD…")
            self.detector = SCRFDDetector(str(SCRFD_PATH))
            self.after(0, self._set_footer, "Initialisiere ArcFace…")
            self.arcface = ArcFaceONNX(str(ARCFACE_PATH))
            self._models_ready = True
            self.after(0, self._set_footer, "Bereit.")
            self.after(0, self._on_models_ready)
        except Exception as e:
            self.after(0, self._set_footer, f"Fehler: {e}", RED)

    def _on_models_ready(self):
        self._add_btn.config(state="normal")
        self._on_listbox_select()
        threading.Thread(target=self._webcam_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # Webcam-Loop
    #   - SCRFD  alle 2 Frames  → Live Feed (Boxen)
    #   - ArcFace alle 10 Frames → ID-Abschnitt (Crops + Namen)
    # ------------------------------------------------------------------

    def _webcam_loop(self):
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            self.after(0, self._show_cam_error)
            return

        frame_n    = 0
        live_boxes = []

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            frame_n += 1

            # SCRFD alle 2 Frames
            if frame_n % 2 == 0 and self._models_ready:
                try:
                    live_boxes = [f["bbox"] for f in self.detector.detect(frame)]
                except Exception:
                    pass

            # ArcFace alle 10 Frames
            if frame_n % 10 == 0 and self._models_ready:
                try:
                    faces      = self.detector.detect(frame)
                    id_results = []
                    for face in faces:
                        emb       = self.arcface.get_embedding(frame, face["kps"])
                        name, sim = self._db.identify(emb, self._threshold)
                        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame.shape[1], x2);  y2 = min(frame.shape[0], y2)
                        crop = frame[y1:y2, x1:x2]
                        id_results.append({
                            "crop":  cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size > 0 else None,
                            "name":  name or "Unbekannt",
                            "sim":   sim,
                            "known": name is not None,
                        })
                    id_results = self._stabilize_results(id_results)
                    self.after(0, self._update_id_section, id_results, frame_n)
                except Exception:
                    pass

            # Live Feed zeichnen
            display = frame.copy()
            for bbox in live_boxes:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(display, (max(0, x1), max(0, y1)), (x2, y2), (50, 210, 80), 2)

            h, w   = display.shape[:2]
            scale  = min(FEED_W / w, FEED_H / h)
            display = cv2.resize(display, (int(w * scale), int(h * scale)))
            rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            self.after(0, self._update_feed, rgb)

        cap.release()

    def _update_feed(self, rgb: np.ndarray):
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._feed_photo = photo
        self._feed_canvas.delete("all")
        self._feed_canvas.create_image(FEED_W // 2, FEED_H // 2, anchor="center", image=photo)

    def _stabilize_results(self, new_results: list[dict]) -> list[dict]:
        """
        Voting über die letzten N ArcFace-Runs.
        Jedes Gesicht (gleicher Index) bekommt den Namen der am häufigsten
        erkannt wurde — "Unbekannt" gewinnt nur bei echter Mehrheit.
        """
        self._vote_history.append(new_results)
        stable = []
        for i, r in enumerate(new_results):
            # Alle Namen für diesen Gesichts-Slot aus der History sammeln
            namen = [run[i]["name"] for run in self._vote_history if i < len(run)]
            # Häufigsten Namen nehmen
            bester_name = Counter(namen).most_common(1)[0][0]
            stable.append({**r, "name": bester_name, "known": bester_name != "Unbekannt"})
        return stable

    def _update_id_section(self, results: list[dict], frame_n: int):
        """Aktualisiert den ID-Abschnitt mit Gesichts-Crops und Namen."""
        self._id_status_lbl.config(text=f"Frame {frame_n}")

        for w in self._id_cards_frame.winfo_children():
            w.destroy()
        self._id_photos.clear()

        if not results:
            tk.Label(self._id_cards_frame, text="Kein Gesicht erkannt",
                     font=("Segoe UI", 10), bg=PANEL, fg=GRAY, pady=12).pack()
            return

        # Karten horizontal nebeneinander
        for r in results:
            card = tk.Frame(self._id_cards_frame, bg=CARD, padx=6, pady=6)
            card.pack(side="left", padx=(0, 8))

            if r["crop"] is not None:
                img = Image.fromarray(r["crop"])
                img.thumbnail((ID_CROP, ID_CROP), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._id_photos.append(photo)
                tk.Label(card, image=photo, bg=CARD).pack()

            col = GREEN if r["known"] else RED
            tk.Label(card, text=r["name"], font=("Segoe UI", 10, "bold"),
                     bg=CARD, fg=col).pack(pady=(4, 0))
            tk.Label(card, text=f"{r['sim'] * 100:.0f}%", font=("Segoe UI", 9),
                     bg=CARD, fg=GRAY).pack()

    def _show_cam_error(self):
        self._feed_canvas.delete("all")
        self._feed_canvas.create_text(FEED_W // 2, FEED_H // 2,
                                       text="Kamera nicht gefunden", fill=RED,
                                       font=("Segoe UI", 14))
        self._set_footer("Kamera nicht verfügbar", RED)

    # ------------------------------------------------------------------
    # Datenbank-Aktionen
    # ------------------------------------------------------------------

    def _add_person(self):
        path = filedialog.askopenfilename(
            title="Bild der Person auswählen",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Alle", "*.*")],
        )
        if not path: return
        name = simpledialog.askstring("Name eingeben", "Name der Person:", parent=self)
        if not name or not name.strip(): return
        self._process_and_store(path, name.strip())

    def _add_image_to_person(self):
        sel = self._listbox.curselection()
        if not sel: return
        name = self._listbox.get(sel[0]).split("  (")[0]
        path = filedialog.askopenfilename(
            title=f"Weiteres Bild für '{name}'",
            filetypes=[("Bilder", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Alle", "*.*")],
        )
        if not path: return
        self._process_and_store(path, name)

    def _process_and_store(self, image_path: str, name: str):
        self._db_status.config(text="Verarbeite Bild…", fg=GRAY)
        self.update_idletasks()
        try:
            img = cv2.imread(image_path)
            if img is None: raise ValueError("Bild konnte nicht geladen werden")
            faces = self.detector.detect(img)
            if not faces:
                self._db_status.config(text="Kein Gesicht gefunden.", fg=RED)
                return
            best = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) *
                                             (f["bbox"][3] - f["bbox"][1]))
            self._db.add(name, self.arcface.get_embedding(img, best["kps"]))
            n = self._db.count(name)
            self._db_status.config(
                text=f"'{name}' gespeichert. ({n} Bild{'er' if n != 1 else ''})", fg=GREEN)
            self._refresh_listbox()
        except Exception as e:
            self._db_status.config(text=f"Fehler: {e}", fg=RED)

    def _remove_person(self):
        sel = self._listbox.curselection()
        if not sel: return
        name = self._listbox.get(sel[0]).split("  (")[0]
        if not messagebox.askyesno("Entfernen", f"'{name}' löschen?"): return
        self._db.remove(name)
        self._db_status.config(text=f"'{name}' entfernt.", fg=ORANGE)
        self._refresh_listbox()
        self._add_img_btn.config(state="disabled")
        self._remove_btn.config(state="disabled")

    def _refresh_listbox(self):
        self._listbox.delete(0, "end")
        for name in self._db.names():
            n = self._db.count(name)
            self._listbox.insert("end", f"{name}  ({n} Bild{'er' if n != 1 else ''})")
        self._db_count_lbl.config(text=self._db_count_text())

    def _db_count_text(self) -> str:
        n = len(self._db.names())
        return f"{n} Person{'en' if n != 1 else ''} in Datenbank"

    def _on_listbox_select(self, _event=None):
        has_sel = bool(self._listbox.curselection())
        self._remove_btn.config(state="normal" if has_sel else "disabled")
        self._add_img_btn.config(
            state="normal" if (has_sel and self._models_ready) else "disabled")

    def _on_threshold_change(self, _val=None):
        self._threshold = self._thresh_var.get()
        self._thresh_lbl.config(text=f"{self._threshold:.0%}")

    def _set_footer(self, text: str, color: str = GRAY):
        self._footer.config(text=text, fg=color)

    def _on_close(self):
        self._stop_event.set()
        self.destroy()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = FaceCheckApp()
    app.mainloop()
