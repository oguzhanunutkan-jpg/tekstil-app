from huggingface_hub import InferenceClient
import base64, io
from PIL import Image

# Hugging Face token
hf_token = "hf_xxx"

# Router üzerinden client
client = InferenceClient(token=hf_token)

# surf_bytes → bytes olarak resim (örn. Flask upload ile alınan image.read())
# Örnek:
# surf_bytes = request.files['image'].read()

# image_to_image çağrısı
result = client.image_to_image(
    model="stabilityai/stable-diffusion-xl-base-1.0",
    prompt="office chair with elegant fabric pattern, realistic product photo",
    negative_prompt="blurry, low quality, distorted, deformed, cartoon, watermark",
    image=surf_bytes,      # bytes olarak resmi gönderiyoruz
    strength=0.7,
    num_inference_steps=25
)

# PIL.Image olarak sonucu alıyoruz
if isinstance(result, Image.Image):
    buf = io.BytesIO()
    result.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
else:
    raise ValueError("Result is not an image. Check the model/parameters.")

# img_b64 → Base64 encoded string, direkt HTML veya JSON ile dönebilir
print(img_b64)
