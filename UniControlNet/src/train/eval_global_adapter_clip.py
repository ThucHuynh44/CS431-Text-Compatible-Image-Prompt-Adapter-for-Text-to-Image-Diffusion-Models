#!/usr/bin/env python
# coding: utf-8

import os
import argparse
from pathlib import Path
import json

import torch
import torch.nn as nn
from PIL import Image

import webdataset as wds
from webdataset import handlers

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
    CLIPModel,
    CLIPProcessor,
)

# ============================================================
# GlobalAdapter (giống hệt code train)
# ============================================================

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

    def forward(self, image_embeds: torch.Tensor):
        """
        image_embeds: (B, in_dim)
        return: (B, num_tokens, out_dim)
        """
        b, _ = image_embeds.shape
        x = self.net(image_embeds)  # (B, num_tokens * out_dim)
        x = x.view(b, self.num_tokens, self.out_dim)
        return x


# ============================================================
# Build SD1.5 + GlobalAdapter (giống logic train, cho inference)
# ============================================================

def build_sd15_global_adapter(
    sd_path: str,
    global_ckpt_path: str,
    device: torch.device,
    weight_dtype=torch.float16,
    num_global_tokens: int = 4,
):
    # ---- Scheduler (giống code train: DDPMScheduler.from_pretrained) ----
    noise_scheduler = DDPMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    # ---- Load SD base ----
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(
        device, dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(
        device, dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(sd_path, subfolder="unet").to(
        device, dtype=weight_dtype
    )

    # ---- CLIP Vision (image encoder) giống code train GlobalAdapter ----
    clip_vision = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    ).to(device, dtype=weight_dtype)
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    # ---- GlobalAdapter ----
    assert os.path.isfile(global_ckpt_path), f"Không tìm thấy GlobalAdapter ckpt: {global_ckpt_path}"
    global_adapter = GlobalAdapter(
        in_dim=768,
        out_dim=768,
        num_tokens=num_global_tokens,
        hidden_mult=4,
    ).to(device, dtype=weight_dtype)
    sd = torch.load(global_ckpt_path, map_location="cpu")
    global_adapter.load_state_dict(sd, strict=True)
    print(f"[GlobalAdapter] Loaded weights from {global_ckpt_path}")

    # Freeze giống IP-Adapter eval
    text_encoder.eval()
    vae.eval()
    unet.eval()
    clip_vision.eval()
    global_adapter.eval()

    return {
        "noise_scheduler": noise_scheduler,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "clip_vision": clip_vision,
        "clip_image_processor": clip_image_processor,
        "global_adapter": global_adapter,
    }


# ============================================================
# Encode helpers (giống IP-Adapter eval để công bằng)
# ============================================================

@torch.no_grad()
def encode_text_empty(tokenizer, text_encoder, batch_size: int, device, max_length=None):
    """
    Text prompt rỗng ("") cho cả cond và uncond – giống IP-Adapter eval.
    """
    texts = [""] * batch_size
    enc = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length or tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    outputs = text_encoder(input_ids, attention_mask=attn_mask)
    return outputs.last_hidden_state  # [B, L, C]


@torch.no_grad()
def encode_image_clip(clip_vision, clip_processor, image_pil: Image.Image, device, dtype):
    inputs = clip_processor(images=image_pil, return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, enabled=(dtype == torch.float16)):
        out = clip_vision(**inputs)
    return out.image_embeds  # [1, 768]


# ============================================================
# Sampling với GlobalAdapter (1 ảnh -> N ảnh sinh)
# Cấu hình CFG / scheduler giống IP-Adapter để công bằng
# ============================================================

@torch.no_grad()
def generate_global_adapter_batch(
    ref_image: Image.Image,
    models,
    num_samples: int = 4,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    lambda_img: float = 1.0,   # giống lambda_img của IP-Adapter: scale sức ảnh
    height: int = 512,
    width: int = 512,
    device: torch.device = torch.device("cuda"),
    seed: int | None = None,
):
    """
    Generate N ảnh (num_samples) từ 1 ảnh ref bằng GlobalAdapter.
    - Text prompt rỗng (""), giống IP-Adapter eval.
    - CFG giữa "no-global" vs "global" để điều khiển ảnh.
    - lambda_img scale lên image_embeds trước GlobalAdapter (giống IP-Adapter).
    """
    noise_scheduler = models["noise_scheduler"]
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    unet = models["unet"]
    clip_vision = models["clip_vision"]
    clip_image_processor = models["clip_image_processor"]
    global_adapter = models["global_adapter"]

    weight_dtype = next(unet.parameters()).dtype

    if seed is not None:
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # --- text embeddings (cond & uncond đều "") ---
    max_length = tokenizer.model_max_length
    text_embeds = encode_text_empty(
        tokenizer, text_encoder, num_samples, device, max_length=max_length
    )  # [B, L, C]
    uncond_embeds = encode_text_empty(
        tokenizer, text_encoder, num_samples, device, max_length=max_length
    )  # [B, L, C]

    # --- encode image prompt bằng CLIP ViT-L/14 ---
    ref_image = ref_image.convert("RGB")
    ref_image = ref_image.resize((512, 512))
    image_embeds_1 = encode_image_clip(
        clip_vision, clip_image_processor, ref_image, device, weight_dtype
    )  # [1, 768]
    image_embeds = image_embeds_1.expand(num_samples, -1)  # [B, 768]

    # cond: dùng ảnh (scaled bởi lambda_img)
    image_embeds_cond = image_embeds * lambda_img
    global_tokens_cond = global_adapter(image_embeds_cond.to(weight_dtype))  # [B, N, 768]

    # uncond: "drop" global theo đúng training (image_embeds = 0)
    zero_image_embeds = torch.zeros_like(image_embeds)
    global_tokens_uncond = global_adapter(zero_image_embeds.to(weight_dtype))  # [B, N, 768]

    # concat text + global tokens
    cond_states = torch.cat([text_embeds, global_tokens_cond.to(text_embeds.dtype)], dim=1)     # [B, L+N, C]
    uncond_states = torch.cat([uncond_embeds, global_tokens_uncond.to(uncond_embeds.dtype)], dim=1)  # [B, L+N, C]

    # pack thành batch 2B cho CFG (giống IP-Adapter eval)
    encoder_hidden_states = torch.cat([uncond_states, cond_states], dim=0)  # [2B, L+N, C]

    # --- init latents ---
    latent_shape = (num_samples, unet.in_channels, height // 8, width // 8)
    latents = torch.randn(latent_shape, device=device, dtype=weight_dtype)
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    latents = latents * noise_scheduler.init_noise_sigma

    # --- DDPMScheduler sampling loop ---
    for t in noise_scheduler.timesteps:
        latent_model_input = torch.cat([latents] * 2, dim=0)  # [2B, C, H, W]

        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample  # [2B, C, H, W]

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # --- decode latents -> images ---
    latents = latents / vae.config.scaling_factor
    with torch.autocast(device_type=device.type, enabled=(weight_dtype == torch.float16)):
        images = vae.decode(latents).sample  # [B, 3, H, W]

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    pil_images = [Image.fromarray((img * 255).round().astype("uint8")) for img in images]
    return pil_images


# ============================================================
# CLIP-I & CLIP-T (CLIP ViT-L/14) – y hệt IP-Adapter script
# ============================================================

def build_clip_metric_model(
    model_name: str = "openai/clip-vit-large-patch14",
    device: torch.device = torch.device("cuda"),
):
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


@torch.no_grad()
def compute_clip_features_image(model, processor, images_pil, device):
    if not isinstance(images_pil, (list, tuple)):
        images_pil = [images_pil]
    inputs = processor(images=images_pil, return_tensors="pt", padding=True).to(device)
    feats = model.get_image_features(pixel_values=inputs["pixel_values"])
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [B, D]


@torch.no_grad()
def compute_clip_features_text(model, processor, texts, device):
    if not isinstance(texts, (list, tuple)):
        texts = [texts]
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    feats = model.get_text_features(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # [T, D]


def clip_i_scores_for_batch(gen_feats, ref_feat):
    ref = ref_feat.unsqueeze(0)  # [1, D]
    sims = (gen_feats * ref).sum(dim=-1)
    return sims  # [B]


def clip_t_scores_for_batch(gen_feats, text_feats):
    """
    CLIPScore ~ cosine(gen_feat, text_feat) (ở cc3m mỗi ảnh 1 caption)
    gen_feats: [B, D], text_feats: [1, D] hoặc [T, D]
    """
    sims = gen_feats @ text_feats.T  # [B, T]
    max_sims, _ = sims.max(dim=1)
    return max_sims  # [B]


# ============================================================
# CC3M WebDataset loader (y hệt IP-Adapter eval)
# ============================================================

def cc3m_webdataset_iter(shards_pattern: str):
    """
    Đọc cc3m-wds:
      - image: jpg/png/jpeg
      - caption: txt
      - key: __key__

    Dùng DataPipeline + split_by_node/split_by_worker để nếu sau này
    bạn chạy multi-process (accelerate/DDP) thì WebDataset không báo lỗi.
    """
    dataset = wds.DataPipeline(
        wds.SimpleShardList(shards_pattern),

        # Quan trọng nếu chạy multi-process
        wds.split_by_node,
        wds.split_by_worker,

        wds.tarfile_to_samples(handler=handlers.warn_and_continue),
        wds.decode("pil"),
        wds.to_tuple("jpg;png;jpeg", "txt", "__key__"),
    )

    for img, txt, key in dataset:
        if isinstance(txt, (bytes, bytearray)):
            txt = txt.decode("utf-8", errors="ignore")
        txt = txt.strip()
        yield img, txt, key


# ============================================================
# Main eval
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GlobalAdapter on CC3M WebDataset with CLIP-I & CLIP-T."
    )
    parser.add_argument("--sd_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--global_ckpt",
        type=str,
        required=True,
        help="Checkpoint GlobalAdapter (.pt) – state_dict như code train.",
    )
    parser.add_argument(
        "--cc3m_shards",
        type=str,
        required=True,
        help='Pattern tới các file .tar, vd: "/path/to/cc3m/cc3m-train-{0000..0575}.tar"',
    )
    parser.add_argument("--num_samples_per_image", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--lambda_img", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--max_images", type=int, default=1000, help="Giới hạn số ảnh CC3M để test (cc3m rất lớn)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_global_tokens", type=int, default=4)

    # file json để lưu kết quả CLIP
    parser.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Đường dẫn file .json để lưu CLIP-I/CLIP-T (vd: results_clip_scores_global.json)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("== Build SD1.5 + GlobalAdapter (giống script train) ==")
    models = build_sd15_global_adapter(
        sd_path=args.sd_path,
        global_ckpt_path=args.global_ckpt,
        device=device,
        weight_dtype=torch.float16,
        num_global_tokens=args.num_global_tokens,
    )

    print("== Build CLIP ViT-L/14 for metrics ==")
    clip_metric_model, clip_metric_processor = build_clip_metric_model(
        model_name="openai/clip-vit-large-patch14", device=device
    )

    clip_i_scores = []
    clip_t_scores = []

    img_idx = 0
    global_seed = args.seed

    print("== Start CC3M evaluation (GlobalAdapter) ==")
    for img, caption, key in cc3m_webdataset_iter(args.cc3m_shards):
        img_idx += 1
        if args.max_images is not None and img_idx > args.max_images:
            break

        print(f"[Image {img_idx}] key={key}")
        # Generate N ảnh từ 1 ảnh ref
        imgs_gen = generate_global_adapter_batch(
            ref_image=img,
            models=models,
            num_samples=args.num_samples_per_image,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            lambda_img=args.lambda_img,
            height=args.height,
            width=args.width,
            device=device,
            seed=global_seed + img_idx,
        )

        # ---- CLIP metrics ----
        # 1) ref image (GT) feature
        ref_feat = compute_clip_features_image(
            clip_metric_model, clip_metric_processor, img, device
        )[0]  # [D]

        # 2) gen images features
        gen_feats = compute_clip_features_image(
            clip_metric_model, clip_metric_processor, imgs_gen, device
        )  # [B,D]

        # 3) text feature (một caption)
        text_feats = compute_clip_features_text(
            clip_metric_model, clip_metric_processor, caption, device
        )  # [1,D]

        # CLIP-I
        clip_i_batch = clip_i_scores_for_batch(gen_feats, ref_feat)
        clip_i_scores.extend(clip_i_batch.cpu().tolist())

        # CLIP-T
        clip_t_batch = clip_t_scores_for_batch(gen_feats, text_feats)
        clip_t_scores.extend(clip_t_batch.cpu().tolist())

        if img_idx % 50 == 0:
            clip_i_mean = sum(clip_i_scores) / len(clip_i_scores)
            clip_t_mean = sum(clip_t_scores) / len(clip_t_scores)
            print(
                f"[{img_idx} images] "
                f"CLIP-I (mean so far)={clip_i_mean:.4f}, "
                f"CLIP-T (mean so far)={clip_t_mean:.4f}"
            )

    # ---- final stats ----
    clip_i_mean = sum(clip_i_scores) / max(1, len(clip_i_scores))
    clip_t_mean = sum(clip_t_scores) / max(1, len(clip_t_scores))

    print("====================================")
    print(f"[GlobalAdapter] Final CLIP-I (mean over all generated): {clip_i_mean:.4f}")
    print(f"[GlobalAdapter] Final CLIP-T (mean over all generated): {clip_t_mean:.4f}")
    print(f"Total generated images: {len(clip_i_scores)}")
    print("====================================")

    # ---- save metrics to file (optional) ----
    if args.metrics_out is not None:
        metrics = {
            "clip_i_mean": clip_i_mean,
            "clip_t_mean": clip_t_mean,
            "num_generated_images": len(clip_i_scores),
            # nếu muốn file nhẹ thì có thể xoá 2 dòng dưới
            "clip_i_all": clip_i_scores,
            "clip_t_all": clip_t_scores,
        }

        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
