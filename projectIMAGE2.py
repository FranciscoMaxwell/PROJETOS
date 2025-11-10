# from diffusers import StableDiffusionPipeline
# import torch
# from datetime import datetime
# import os

# # Configuração inicial
# modelo = "SG161222/Realistic_Vision_V1.4"
# dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

# # Carregar o modelo com precisão mista (mais rápido e leve)
# pipe = StableDiffusionPipeline.from_pretrained(
#     modelo,
#     torch_dtype=torch.float16 if dispositivo == "cuda" else torch.float32,
#     safety_checker=None  # desativa filtro de segurança para imagens bloqueadas
# ).to(dispositivo)

# # Diretório de saída
# os.makedirs("outputs", exist_ok=True)

# # Prompt detalhado e otimizado
# prompt = (
#     "realistic portrait of a beautiful woman, natural skin texture, cinematic lighting, "
#     "sharp focus, ultra-detailed, 8k resolution, soft background, depth of field, "
#     "emotionally expressive face, professional photography style, photorealistic, "
#     "skin pores visible, detailed eyes, elegant, subtle makeup"
# )

# # Prompt negativo (remove imperfeições e distorções)
# negative_prompt = (
#     "blurry, cartoon, anime, low quality, extra limbs, distorted face, deformed hands, "
#     "overexposed, underexposed, unrealistic, nsfw, watermark, text, logo"
# )

# # Geração da imagem
# image = pipe(
#     prompt=prompt,
#     negative_prompt=negative_prompt,
#     num_inference_steps=40,  # número de passos (maior = mais detalhado)
#     guidance_scale=8.0,      # controla o quanto o modelo segue o prompt
# ).images[0]

# # Nome automático baseado na hora
# nome_arquivo = f"outputs/mulher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

# # Salvar
# image.save(nome_arquivo)

# print(f"✅ Imagem gerada com sucesso e salva em: {nome_arquivo}")
