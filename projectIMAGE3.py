# import os
# import random
# import torch
# from diffusers import StableDiffusionPipeline

# # =======================
# # Configurações
# # =======================
# MODEL_NAME = "hakurei/waifu-diffusion"  # Ou substitua por outro modelo anime
# OUTPUT_DIR = "outputs_anime"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🧠 Usando dispositivo: {device}")

# # =======================
# # Carrega pipeline
# # =======================
# pipe = StableDiffusionPipeline.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     safety_checker=None,  # Remove filtro NSFW
# ).to(device)

# # Semente aleatória para reprodutibilidade
# seed = random.randint(0, 2**32 - 1)
# generator = torch.manual_seed(seed)
# print(f"🔢 Semente aleatória usada: {seed}")

# # =======================
# # Prompt detalhado
# # =======================
# prompt = (
#     "1girl, anime, ultra detailed, perfect face, big eyes, "
#     "high resolution, dynamic lighting, intricate hair, "
#     "soft skin, masterpiece, professional illustration, vibrant colors"
# )

# negative_prompt = (
#     "bad anatomy, blurry, low quality, watermark, extra limbs, multiple faces, "
#     "pixelated, text, signature, nsfw"
# )

# # =======================
# # Geração da imagem
# # =======================
# result = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=140,    # Mais passos = mais detalhes
#     guidance_scale=8.5,        # Mantém fidelidade ao prompt
#     height=768,
#     width=512,
#     generator=generator
# )

# image = result.images[0]
# save_path = os.path.join(OUTPUT_DIR, f"anime_gen_{seed}.png")
# image.save(save_path)
# print(f"✅ Imagem anime gerada e salva em: {save_path}")
