# import cv2
# import numpy as np
# import json
# import os
# from deepface import DeepFace
# import time

# DB_FILE = "face_db.json"
# CAPTURE_COUNT = 3        # Número de capturas por pessoa
# THRESHOLD = 0.6          # Limite de distância para reconhecimento
# WINDOW_SIZE = (400, 400) # Tamanho da janela

# # ----------------- FUNÇÕES -----------------
# def load_db():
#     if os.path.exists(DB_FILE):
#         with open(DB_FILE, "r") as f:
#             return json.load(f)
#     return {}

# def save_db(db):
#     with open(DB_FILE, "w") as f:
#         json.dump(db, f, indent=2)

# def capture_face(name, origin="", breed="", description=""):
#     cap = cv2.VideoCapture(0)
#     captured = 0
#     embeddings = []
    
#     print(f"Posicione o rosto dentro do oval e pressione 'c' para capturar ({CAPTURE_COUNT} vezes).")

#     while captured < CAPTURE_COUNT:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         x1, y1 = w//2 - 150, h//2 - 180
#         x2, y2 = w//2 + 150, h//2 + 180

#         overlay = frame.copy()
#         cv2.ellipse(overlay, (w//2, h//2), (150, 180), 0, 0, 360, (0,255,0), 2)
#         alpha = 0.3
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         cv2.imshow("Capture Face", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('c'):
#             face_img = frame[y1:y2, x1:x2]
#             try:
#                 embedding = DeepFace.represent(face_img, enforce_detection=True)[0]["embedding"]
#                 embeddings.append(embedding)
#                 captured += 1
#                 print(f"Captura {captured}/{CAPTURE_COUNT} registrada.")
#             except:
#                 print("Face não detectada. Tente novamente.")
#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

#     if embeddings:
#         db = load_db()
#         db[name] = {
#             "origin": origin,
#             "breed": breed,
#             "description": description,
#             "embeddings": embeddings
#         }
#         save_db(db)
#         print(f"{name} cadastrado com sucesso!")

# def recognize_face():
#     db = load_db()
#     if not db:
#         print("Banco de dados vazio!")
#         return

#     cap = cv2.VideoCapture(0)
#     print("Posicione o rosto dentro do oval e pressione 'c' para verificar; 'q' para cancelar.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         x1, y1 = w//2 - 150, h//2 - 180
#         x2, y2 = w//2 + 150, h//2 + 180

#         overlay = frame.copy()
#         cv2.ellipse(overlay, (w//2, h//2), (150, 180), 0, 0, 360, (0,255,0), 2)
#         alpha = 0.3
#         cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#         cv2.imshow("Recognize Face", frame)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('c'):
#             face_img = frame[y1:y2, x1:x2]
#             try:
#                 emb = DeepFace.represent(face_img, enforce_detection=True)[0]["embedding"]
#                 match_name, min_dist = "Desconhecido", float('inf')
#                 for name, data in db.items():
#                     for stored_emb in data["embeddings"]:
#                         dist = np.linalg.norm(np.array(emb) - np.array(stored_emb))
#                         if dist < min_dist:
#                             min_dist = dist
#                             match_name = name if dist < THRESHOLD else "Desconhecido"
#                 if match_name != "Desconhecido":
#                     info = db[match_name]
#                     print(f"Reconhecido: {match_name}")
#                     print(f"Origem: {info['origin']}, Raça: {info['breed']}, Descrição: {info.get('description','')}")
#                 else:
#                     print("Não reconhecido.")
#             except:
#                 print("Face não detectada.")
#         elif key == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# def list_faces():
#     db = load_db()
#     if not db:
#         print("Banco vazio!")
#         return
#     for name, data in db.items():
#         print(f"Nome: {name}, Origem: {data['origin']}, Raça: {data['breed']}, Descrição: {data.get('description','')}")

# # ----------------- SCRIPT PRINCIPAL -----------------
# def main():
#     while True:
#         print("\nEscolha a opção:")
#         print("1 - Cadastrar novo rosto")
#         print("2 - Verificar rosto")
#         print("3 - Listar rostos cadastrados")
#         print("0 - Sair")
#         choice = input("Opção: ")

#         if choice == '1':
#             name = input("Nome: ")
#             origin = input("Origem: ")
#             breed = input("Raça: ")
#             description = input("Descrição (opcional): ")
#             capture_face(name, origin, breed, description)
#         elif choice == '2':
#             recognize_face()
#         elif choice == '3':
#             list_faces()
#         elif choice == '0':
#             break
#         else:
#             print("Opção inválida!")

# if __name__ == "__main__":
#     main()
