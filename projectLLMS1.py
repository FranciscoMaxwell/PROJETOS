# import os
# from diffusers import StableDiffusionPipeline
# import torch
# from PIL import Image

# # -----------------------------
# # CONFIGURAÇÕES
# # -----------------------------
# MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # Pode trocar por waifu-diffusion ou NovelAI
# OUTPUT_DIR = "images"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Certifica que a pasta de saída existe
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # -----------------------------
# # CARREGAR PIPELINE
# # -----------------------------
# pipe = StableDiffusionPipeline.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32,
# )
# pipe = pipe.to(DEVICE)

# # -----------------------------
# # FUNÇÃO PARA GERAR IMAGEM
# # -----------------------------
# def gerar_imagem(prompt: str, filename: str = "imagem_gerada.png", steps: int = 50, scale: float = 7.5):
#     """
#     Gera uma imagem a partir de um prompt.

#     Args:
#         prompt (str): texto do prompt.
#         filename (str): nome do arquivo de saída.
#         steps (int): número de passos (mais passos = mais detalhado).
#         scale (float): guidance scale (quanto maior, mais fiel ao prompt).
#     """
#     image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale).images[0]
#     save_path = os.path.join(OUTPUT_DIR, filename)
#     image.save(save_path)
#     print(f"✅ Imagem salva em: {save_path}")

# # -----------------------------
# # EXEMPLO DE USO
# # -----------------------------
# if __name__ == "__main__":
#     prompt = input("Digite seu prompt: ")
#     gerar_imagem(prompt, filename="imagem_gerada.png", steps=50, scale=7.5)
