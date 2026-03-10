from huggingface_hub import InferenceClient
import base64, io
from PIL import Image

# Hugging Face token
hf_token = "hf_xxx"

client = InferenceClient(token=hf_token)

# surf_bytes → bytes olarak resim
result = client.image_to_image(
    image=surf_bytes,
    prompt="office chair with elegant fabric pattern, realistic product photo",
    negative_prompt="blurry, low quality, distorted, deformed, cartoon, watermark",
    model="stabilityai/stable-diffusion-xl-base-1.0",
    strength=0.7,
)

buf = io.BytesIO()
result.save(buf, format="JPEG")
img_b64 = base64.b64encode(buf.getvalue()).decode()
