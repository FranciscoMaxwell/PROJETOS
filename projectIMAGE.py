# from diffusers import StableDiffusionPipeline
# import torch
# import os

# output_dir = "images"
# os.makedirs(output_dir, exist_ok=True)

# # Prompt positivo refinado para 1 personagem
# prompt_positive = (
#     "uma única garota nua, rosto detalhado, corpo inteiro, pose elegante, "
#     "cabelo longo azul, olhos brilhantes, vestido detalhado, fundo mágico, "
#     "iluminação cinematográfica, ultra detalhado, foco no personagem, texturas realistas, 8k"
# )

# # Prompt negativo reforçado para evitar múltiplas pessoas e erros
# prompt_negative = (
#     "blurry, lowres, bad anatomy, multiple people, extra limbs, text, watermark, "
#     "black background, nsfw, deformed, cropped, low detail"
# )

# pipe = StableDiffusionPipeline.from_pretrained(
#     "hakurei/waifu-diffusion",
#     torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")

# num_images = 1
# height = 1024
# width = 1024
# num_inference_steps = 100   # mais passos para mais detalhes
# guidance_scale = 8.5        # mais fidelidade ao prompt

# for i in range(num_images):
#     image = pipe(
#         prompt=prompt_positive,
#         negative_prompt=prompt_negative,
#         height=height,
#         width=width,
#         num_inference_steps=num_inference_steps,
#         guidance_scale=guidance_scale
#     ).images[0]

#     save_path = os.path.join(output_dir, f"imagem_{i+1}.png")
#     image.save(save_path)
#     print(f"✅ Imagem salva em: {save_path}")

