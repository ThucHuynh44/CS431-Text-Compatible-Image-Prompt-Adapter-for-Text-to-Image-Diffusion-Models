from dataclasses import dataclass
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    CLIPVisionModelWithProjection, CLIPImageProcessor
)

# -----------------------------
# Global Adapter Module
# -----------------------------
class GlobalAdapter(nn.Module):
    def __init__(self, in_dim=768, out_dim=768, num_tokens=4, hidden_mult=4):
        super().__init__()
        hidden_dim = in_dim * hidden_mult
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim * num_tokens),
        )

    def forward(self, img_embeds):
        b, _ = img_embeds.shape
        x = self.net(img_embeds)
        return x.view(b, self.num_tokens, self.out_dim)


# ================================================================
#                GLOBAL ADAPTER PIPELINE (MAIN CLASS)
# ================================================================
class GlobalAdapterPipeline:
    def __init__(
        self,
        sd15_name,
        adapter_ckpt,
        device="cuda",
        global_tokens=4
    ):
        self.device = torch.device(device)

        # ----------- Load SD components -----------
        self.tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder").to(self.device)

        self.vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet").to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(sd15_name, subfolder="scheduler")

        # ----------- CLIP Vision Encoder -----------
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)

        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # ----------- Global Adapter -----------
        self.global_adapter = GlobalAdapter(
            in_dim=768,
            out_dim=768,
            num_tokens=global_tokens,
        ).to(self.device)

        ckpt = torch.load(adapter_ckpt, map_location="cpu")
        self.global_adapter.load_state_dict(ckpt)
        self.global_adapter.eval()

    # ================================================================
    #                       DDIM SAMPLING
    # ================================================================
    @torch.no_grad()
    def ddim_sample(self, latents, cond, uncond, steps=30, cfg_scale=7.5):
        self.scheduler.set_timesteps(steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        for t in timesteps:
            noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=uncond).sample
            noise_pred_cond   = self.unet(latents, t, encoder_hidden_states=cond).sample

            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents

    # ================================================================
    #                       MAIN GENERATE FUNCTION
    # ================================================================
    @torch.no_grad()
    def generate(
        self,
        prompt,
        pil_image,
        resolution=512,
        num_samples=4,
        num_inference_steps=30,
        cfg_scale=7.5,
        scale=1.0,
        seed=0
    ):
        images = []

        for i in range(num_samples):
            torch.manual_seed(seed + i)

            # 1) image -> CLIP image embedding
            clip_inputs = self.clip_processor(images=pil_image, return_tensors="pt").to(self.device)
            img_embeds = self.clip_vision(**clip_inputs).image_embeds  # (1,768)

            # 2) text embeddings
            text_inputs = self.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)

            text_embeds = self.text_encoder(**text_inputs).last_hidden_state  # (1,T,768)

            # 3) Global tokens
            zeros_img = torch.zeros_like(img_embeds)
            g_zero = self.global_adapter(zeros_img)
            g_img  = self.global_adapter(img_embeds)

            global_tokens = g_zero + scale * (g_img - g_zero)  # (1,N,768)

            # cond
            cond_states = torch.cat([text_embeds, global_tokens], dim=1)

            # uncond
            uncond_inputs = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.device)

            uncond_embeds = self.text_encoder(**uncond_inputs).last_hidden_state
            uncond_states = torch.cat([uncond_embeds, g_zero], dim=1)

            # 4) latents
            h = w = resolution // 8
            latents = torch.randn(1, 4, h, w, device=self.device)

            # 5) DDIM sample
            latents = self.ddim_sample(
                latents,
                cond_states,
                uncond_states,
                steps=num_inference_steps,
                cfg_scale=cfg_scale,
            )

            # 6) Decode
            image_out = self.vae.decode(latents / 0.18215).sample
            image_out = (image_out.clamp(-1, 1) + 1) * 0.5
            image_out = image_out.cpu().permute(0, 2, 3, 1).numpy()[0]
            image_out = Image.fromarray((image_out * 255).astype("uint8"))

            images.append(image_out)

        return images
