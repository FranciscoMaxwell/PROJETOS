import cv2
import torch
import json
import threading
from tkinter import Tk, Label, Entry, Button
import numpy as np

DATA_FILE = "registry.json"

try:
    with open(DATA_FILE, "r") as f:
        registry = json.load(f)
except:
    registry = {}

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.classes = [0, 16, 17, 18]  # person, cat, dog, bird

cap = cv2.VideoCapture(0)

pending = {}  # janelas abertas
tracked_objects = {}  # objeto_id : (centro_x, centro_y)

def save_registry():
    with open(DATA_FILE, "w") as f:
        json.dump(registry, f, indent=2)

def ask_info(obj_id):
    if obj_id in registry:
        return
    def submit():
        registry[obj_id] = {
            "nome": name_var.get(),
            "origem": origin_var.get(),
            "raca": race_var.get(),
            "descricao": desc_var.get()
        }
        save_registry()
        root.destroy()
        pending.pop(obj_id, None)

    root = Tk()
    root.title("Registrar Objeto")
    name_var = Entry(root)
    origin_var = Entry(root)
    race_var = Entry(root)
    desc_var = Entry(root)

    Label(root, text="Nome:").pack()
    name_var.pack()
    Label(root, text="Origem:").pack()
    origin_var.pack()
    Label(root, text="Raça:").pack()
    race_var.pack()
    Label(root, text="Descrição:").pack()
    desc_var.pack()
    Button(root, text="Salvar", command=submit).pack()
    root.mainloop()

def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2)//2, (y1 + y2)//2)

DIST_THRESHOLD = 50  # pixels, para considerar mesmo objeto

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame)
    current_centroids = []
    boxes = []

    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        centroid = get_centroid((x1,y1,x2,y2))
        current_centroids.append(centroid)
        boxes.append((x1,y1,x2,y2, int(cls)))

    new_tracked = {}
    # atribuir objetos detectados aos objetos rastreados
    for i, centroid in enumerate(current_centroids):
        matched_id = None
        for obj_id, old_centroid in tracked_objects.items():
            dist = np.linalg.norm(np.array(centroid)-np.array(old_centroid))
            if dist < DIST_THRESHOLD:
                matched_id = obj_id
                break
        if matched_id is None:
            matched_id = f"{boxes[i][4]}_{len(tracked_objects)+len(new_tracked)}"
        new_tracked[matched_id] = centroid

        # desenhar
        x1, y1, x2, y2, cls_idx = boxes[i]
        label = model.names[cls_idx]
        info = registry.get(matched_id, None)
        text = f"{info['nome']} ({info['origem']})" if info else label

        if not info and matched_id not in pending:
            pending[matched_id] = True
            threading.Thread(target=ask_info, args=(matched_id,), daemon=True).start()

        cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,text,(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    tracked_objects = new_tracked

    cv2.imshow("Detecção", frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
