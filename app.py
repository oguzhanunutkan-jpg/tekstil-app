from huggingface_hub import InferenceClient
import base64, io
from PIL import Image

hf_token = "hf_xxx"
client = InferenceClient(token=hf_token)

# surf_bytes = request.files['image'].read()

try:
    result = client.image_to_image(
        model="timbrooks/instruct-pix2pix",  # ✅ img2img destekleyen model
        image=surf_bytes,
        prompt="office chair with elegant fabric pattern, realistic product photo",
        negative_prompt="blurry, low quality, distorted, deformed, cartoon, watermark",
    )

    if not isinstance(result, Image.Image):
        raise ValueError("Beklenmedik yanıt tipi: PIL.Image bekleniyor")

    buf = io.BytesIO()
    result.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    print(img_b64)

except Exception as e:
    print(f"Inference hatası: {e}")
    raise
