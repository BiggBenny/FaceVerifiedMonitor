"""
Erstellt SFace-Embeddings aus Bildern (OpenCV-Modell, ArcFace-Architektur).

Ordnerstruktur:
    images/
        alice/
            foto1.jpg
            foto2.jpg
        bob/
            foto1.jpg

Ausgabe: embeddings.npy  →  { "alice": [array, array], "bob": [array] }
"""

import numpy as np
from pathlib import Path

from deepface import DeepFace


def extract_embedding(img_path: Path) -> np.ndarray | None:
    try:
        result = DeepFace.represent(
            img_path=str(img_path),
            model_name="SFace",
            detector_backend="mtcnn",
            enforce_detection=True,
        )
        if not result:
            print(f"  [!] Kein Gesicht gefunden: {img_path.name}")
            return None
        if len(result) > 1:
            print(f"  [!] Mehrere Gesichter, nehme das größte: {img_path.name}")
        emb = np.array(result[0]["embedding"])
        emb = emb / np.linalg.norm(emb)  # L2-normieren
        return emb  # shape (512,)
    except Exception as e:
        print(f"  [!] Fehler bei {img_path.name}: {e}")
        return None


def main():
    images_dir = Path("data/images")
    if not images_dir.exists():
        print("Ordner 'data/images/' nicht gefunden.")
        print("Lege Bilder ab unter:  data/images/<person>/<foto>.jpg")
        return

    embeddings: dict[str, list[np.ndarray]] = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for person_dir in sorted(images_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person = person_dir.name
        print(f"\n{person}:")
        embeddings[person] = []

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue
            emb = extract_embedding(img_path)
            if emb is not None:
                embeddings[person].append(emb)
                print(f"  OK  {img_path.name}  shape={emb.shape}")

        if not embeddings[person]:
            del embeddings[person]
            print(f"  [!] Keine Embeddings für {person}")

    if not embeddings:
        print("\nKeine Embeddings erstellt.")
        return

    out = {name: np.stack(vecs) for name, vecs in embeddings.items()}
    np.save("embeddings.npy", out, allow_pickle=True)

    print(f"\nGespeichert: embeddings.npy")
    for name, vecs in out.items():
        print(f"  {name}: {vecs.shape[0]} Bild(er), Embedding-Dim={vecs.shape[1]}")


if __name__ == "__main__":
    main()
