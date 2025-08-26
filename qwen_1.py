import torch
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration

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

# optionally load LoRA weights to speed up inference
pipe.load_lora_weights("lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V1.1.safetensors")
# pipe.load_lora_weights(
#     "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
# )
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cuda").manual_seed(42)
image = load_image(
    "https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/resources/dog_plushie.png"
).convert("RGB")

prompt = "change the dog plushie for a cat preserving the background, the lighting, colors, shadows, also the cat plushie should have the same style of the dog plushie with the same eyes and lines."

# change steps to 8 or 4 if you used the lighting loras
image = pipe(image, prompt, num_inference_steps=8).images[0]

image.save("qwenimageedit.png")