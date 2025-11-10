# import cv2
# import os
# import json
# import numpy as np
# from deepface import DeepFace

# # Caminho do banco de dados
# DB_PATH = "rostos_db.json"

# # Funções de utilidade
# def carregar_banco():
#     if not os.path.exists(DB_PATH):
#         return {}
#     with open(DB_PATH, "r", encoding="utf-8") as f:
#         return json.load(f)

# def salvar_banco(data):
#     with open(DB_PATH, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=4)

# # Captura e retorna a imagem do rosto enquadrado no círculo
# def capturar_rosto():
#     cap = cv2.VideoCapture(0)
#     largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     centro = (largura // 2, altura // 2)
#     raio = min(largura, altura) // 3  # círculo central

#     print("Posicione o rosto dentro do círculo e pressione 'c' para capturar, 'q' para cancelar.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Erro ao acessar a câmera.")
#             break

#         # Desenhar círculo guia
#         overlay = frame.copy()
#         cv2.circle(overlay, centro, raio, (0, 255, 0), 2)
#         cv2.putText(overlay, "Posicione o rosto dentro do circulo", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#         cv2.imshow("Captura de Rosto", overlay)
#         key = cv2.waitKey(1) & 0xFF

#         if key == ord('c'):
#             # recorte dentro do círculo (opcional: usar tudo)
#             mask = np.zeros_like(frame, dtype=np.uint8)
#             cv2.circle(mask, centro, raio, (255, 255, 255), -1)
#             rosto = cv2.bitwise_and(frame, mask)
#             cap.release()
#             cv2.destroyAllWindows()
#             return rosto

#         elif key == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             print("Cancelado.")
#             return None

# # Gera representação facial (embedding)
# def gerar_embedding(img):
#     try:
#         embedding = DeepFace.represent(img, model_name="Facenet", enforce_detection=False)
#         if isinstance(embedding, list) and len(embedding) > 0:
#             return embedding[0]['embedding']
#     except Exception as e:
#         print("Erro ao extrair embedding:", e)
#     return None

# # Cadastro de novo rosto
# def cadastrar_rosto():
#     nome = input("Informe o nome da pessoa: ").strip()
#     img = capturar_rosto()
#     if img is None:
#         return
#     emb = gerar_embedding(img)
#     if emb is None:
#         print("Falha ao gerar representação facial.")
#         return

#     db = carregar_banco()
#     db[nome] = emb
#     salvar_banco(db)
#     print(f"Rosto de '{nome}' cadastrado com sucesso!")

# # Verificação de rosto
# def verificar_rosto():
#     img = capturar_rosto()
#     if img is None:
#         return
#     emb = gerar_embedding(img)
#     if emb is None:
#         print("Não foi possível extrair características faciais.")
#         return

#     db = carregar_banco()
#     if not db:
#         print("Banco de rostos vazio.")
#         return

#     # comparação simples por distância euclidiana
#     def dist(a, b):
#         return np.linalg.norm(np.array(a) - np.array(b))

#     resultados = {nome: dist(emb, np.array(v)) for nome, v in db.items()}
#     nome_mais_provavel = min(resultados, key=resultados.get)
#     menor_dist = resultados[nome_mais_provavel]

#     if menor_dist < 10:  # limiar ajustável
#         print(f"Rosto reconhecido como: {nome_mais_provavel} (distância: {menor_dist:.2f})")
#     else:
#         print(f"Nenhum rosto correspondente encontrado. (menor distância: {menor_dist:.2f})")

# # Listar rostos cadastrados
# def listar_rostos():
#     db = carregar_banco()
#     if not db:
#         print("Nenhum rosto cadastrado.")
#         return
#     print("\nRostos cadastrados:")
#     for nome in db.keys():
#         print(" -", nome)
#     print()

# # Menu principal
# def menu():
#     while True:
#         print("\nEscolha a opção:")
#         print("1 - Cadastrar novo rosto")
#         print("2 - Verificar rosto (comparar com banco)")
#         print("3 - Listar rostos cadastrados")
#         print("0 - Sair")
#         op = input("Opção: ")

#         if op == "1":
#             cadastrar_rosto()
#         elif op == "2":
#             verificar_rosto()
#         elif op == "3":
#             listar_rostos()
#         elif op == "0":
#             break
#         else:
#             print("Opção inválida.")

# if __name__ == "__main__":
#     menu()
