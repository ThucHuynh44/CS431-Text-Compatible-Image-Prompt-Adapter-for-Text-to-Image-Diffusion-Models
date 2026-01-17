from huggingface_hub import hf_hub_download

# Runwayml SD v1.5, file ckpt pruned
ckpt_path = hf_hub_download(
    repo_id="runwayml/stable-diffusion-v1-5",
    filename="v1-5-pruned.ckpt",
    local_dir="models/StableDiffusion",
)

print("Saved to:", ckpt_path)