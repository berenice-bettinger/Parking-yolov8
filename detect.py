"""Détection de places de parking libres / occupées avec un modèle YOLOv8 entraîné sur PKLot.

Exemples :
    python detect.py --source parking.jpg
    python detect.py --source dossier_images/
    python detect.py --source video.mp4

Pour chaque image (ou chaque image de la vidéo), le script dessine les places
libres en vert et les places occupées en rouge, affiche le compte en haut de
l'image et enregistre le résultat dans le dossier --output.
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

FREE_COLOR = (60, 200, 60)       # BGR : vert
OCCUPIED_COLOR = (40, 40, 230)   # BGR : rouge
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def class_roles(names):
    """Associe chaque id de classe à 'free' ou 'occupied' à partir de son nom.

    Le nom des classes dépend de l'export du dataset (ex. 'space-empty',
    'space-occupied'), on se base donc sur le texte plutôt que sur l'ordre.
    """
    roles = {}
    for class_id, name in names.items():
        lowered = name.lower()
        if "occup" in lowered:
            roles[class_id] = "occupied"
        elif "empty" in lowered or "free" in lowered or "libre" in lowered:
            roles[class_id] = "free"
    return roles


def annotate(frame, result, roles):
    """Dessine les boîtes et le compteur sur l'image. Retourne (image, libres, occupées)."""
    free = occupied = 0
    for box in result.boxes:
        role = roles.get(int(box.cls))
        if role is None:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = FREE_COLOR if role == "free" else OCCUPIED_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if role == "free":
            free += 1
        else:
            occupied += 1

    total = free + occupied
    label = f"Libres : {free}   Occupees : {occupied}   Total : {total}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (0, 0), (w + 20, h + 20), (30, 30, 30), -1)
    cv2.putText(frame, label, (10, h + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame, free, occupied


def process_image(model, path, output_dir, roles, conf):
    result = model.predict(str(path), conf=conf, verbose=False)[0]
    frame, free, occupied = annotate(result.orig_img.copy(), result, roles)
    out_path = output_dir / f"{path.stem}_detect{path.suffix}"
    cv2.imwrite(str(out_path), frame)
    print(f"{path.name} : {free} libres, {occupied} occupées -> {out_path}")


def process_video(model, path, output_dir, roles, conf):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = output_dir / f"{path.stem}_detect.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, conf=conf, verbose=False)[0]
        frame, _, _ = annotate(frame, result, roles)
        writer.write(frame)
        frames += 1

    capture.release()
    writer.release()
    print(f"{path.name} : {frames} images traitées -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compte les places de parking libres et occupées.")
    parser.add_argument("--weights", default="models/best.pt", help="modèle YOLOv8 entraîné (.pt)")
    parser.add_argument("--source", required=True, help="image, dossier d'images ou vidéo")
    parser.add_argument("--output", default="resultats", help="dossier de sortie")
    parser.add_argument("--conf", type=float, default=0.25, help="seuil de confiance (0 à 1)")
    args = parser.parse_args()

    model = YOLO(args.weights)
    roles = class_roles(model.names)
    if set(roles.values()) != {"free", "occupied"}:
        raise SystemExit(f"Classes inattendues dans le modèle : {model.names}")

    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for path in files:
            process_image(model, path, output_dir, roles, args.conf)
    elif source.suffix.lower() in VIDEO_EXTENSIONS:
        process_video(model, source, output_dir, roles, args.conf)
    else:
        process_image(model, source, output_dir, roles, args.conf)


if __name__ == "__main__":
    main()
