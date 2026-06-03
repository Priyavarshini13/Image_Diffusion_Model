from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Load the Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    )

pipe = pipe.to(device)

# Get prompt from user
prompt = input("Enter your image description: ")

# Generate image
if device == "cuda":
    with torch.autocast("cuda"):
        image = pipe(prompt, num_inference_steps=20).images[0]
else:
    image = pipe(prompt, num_inference_steps=20).images[0]

# Display image
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()

# Save image
filename = "generated_image.png"
image.save(filename)

print(f"Image saved as {filename}")