import itertools
import json
import os
import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from PIL import Image

from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from diffusers.utils import load_image

model_id = "Qwen/Qwen-Image-Edit"
torch_dtype = torch.bfloat16
device = "cuda"

quantization_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
)
transformer = QwenImageTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
transformer = transformer.to("cpu")

quantization_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=quantization_config,
    torch_dtype=torch_dtype,
)
text_encoder = text_encoder.to("cpu")

pipe = QwenImageEditPipeline.from_pretrained(
    model_id, transformer=transformer, text_encoder=text_encoder, torch_dtype=torch_dtype
)

pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(42)

with open("comments.json", "r", encoding="utf-8") as f:
    comments_data = json.load(f)

output_folder = "Output"
os.makedirs(output_folder, exist_ok=True)

for key, data in comments_data.items():
    prompt = data["comment"]
    image_url = data["image_url"]
    comment_id = data["id"]

    image = load_image(image_url).convert("RGB")
    image = image.resize((512, 512), resample=Image.LANCZOS)

    edited_image = pipe(image, prompt, num_inference_steps=32).images[0]

    output_filename = os.path.join(output_folder, f"{comment_id}_output_qwen_01.png")
    edited_image.save(output_filename)
    print(f"Saved edited image for {comment_id} -> {output_filename}")