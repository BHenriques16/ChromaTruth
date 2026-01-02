import os
import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInpaintPipeline
from tqdm import tqdm


# Configure the Inpainting Pipeline (Stable Diffusion)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

def generate_realistic_fake(image_path, output_path):
    init_image = Image.open(image_path).convert("RGB").resize((256, 256))
    
    # Create a mask for the area to be manipulated (e.g., eyes and nose).
    mask_image = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(mask_image)
    # Definir um retângulo na zona central da face
    draw.rectangle([64, 80, 192, 160], fill=255) 

    # Generate manipulation (Inpainting Attack)
    # The prompt defines what the AI ​​should "change" in the mask zone.
    prompt = "high quality detailed human face, realistic skin texture"
    
    fake_image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=20
    ).images[0]

    fake_image.save(output_path)

# Folder paths
real_dir = 'data/03_Test_Pairs/Real'
fake_dir = 'data/03_Test_Pairs/Fake'
os.makedirs(fake_dir, exist_ok=True)

# Process sample for the test dataset
print("Generating realistic deepfakes...")
test_images = os.listdir(real_dir)[:100]

for img_name in tqdm(test_images):
    generate_realistic_fake(
        os.path.join(real_dir, img_name),
        os.path.join(fake_dir, img_name)
    )
