#!/usr/bin/env python
# coding: utf-8

import os
import math
import random
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import IterableDataset, DataLoader

import webdataset as wds
from PIL import Image

from torchvision import transforms

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


# ============================================================
# GlobalAdapter: từ image embedding (768) -> global tokens (N_tokens, 768)
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
        x = self.net(image_embeds)              # (B, num_tokens * out_dim)
        x = x.view(b, self.num_tokens, self.out_dim)
        return x


# ============================================================
# Streaming dataset dùng WebDataset (CC3M style: key "jpg", "txt")
# ============================================================

class StreamingCC3MDataset(IterableDataset):
    def __init__(
        self,
        shards: str,
        resolution: int = 256,
        drop_txt_prob: float = 0.0,
        shuffle_buffer: int = 1000,
    ):
        super().__init__()
        self.shards = shards
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self):
        import webdataset as wds
        from webdataset import handlers

        # Pipeline có split_by_node + split_by_worker để DDP/accelerate khỏi complain
        dataset = wds.DataPipeline(
            wds.SimpleShardList(self.shards),
            wds.split_by_node,   # <<< quan trọng
            wds.split_by_worker, # <<< mỗi worker xử lý shard khác nhau
            # đọc tar -> sample
            wds.tarfile_to_samples(handler=handlers.warn_and_continue),
            # gộp key -> { "jpg": ..., "txt": ... }
            wds.decode("pil"),
            wds.shuffle(self.shuffle_buffer),
        )

        for sample in dataset:
            if "jpg" not in sample:
                continue
            img = sample["jpg"]
            if not isinstance(img, Image.Image):
                continue

            img = img.convert("RGB")

            caption = sample.get("txt", "")
            if isinstance(caption, bytes):
                caption = caption.decode("utf-8", errors="ignore")

            # nếu muốn có pre-drop text ở đây thì dùng self.drop_txt_prob,
            # còn không thì cứ để "" xử lý trong training loop
            if random.random() < self.drop_txt_prob:
                caption = ""

            yield {
                "image": img,
                "text": caption,
            }


def collate_fn(examples):
    images = [ex["image"] for ex in examples]
    texts = [ex["text"] for ex in examples]
    return {"images": images, "texts": texts}


# ============================================================
# Training
# ============================================================

@dataclass
class TrainArgs:
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    train_shards: str = "https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-{0000..0575}.tar"
    output_dir: str = "./global_adapter_ckpt"

    resolution: int = 512
    train_batch_size: int = 4
    num_workers: int = 4
    learning_rate: float = 1e-4
    max_train_steps: int = 363750
    gradient_accumulation_steps: int = 1
    weight_decay: float = 1e-2

    num_global_tokens: int = 4

    drop_txt_prob: float = 0.5       # xác suất drop caption -> ""
    drop_global_prob: float = 0.5    # xác suất drop global cond (zero image_embeds)

    seed: int = 42
    mixed_precision: str = "fp16"  # "no" / "fp16" / "bf16"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained-model-name-or-path", type=str,
                        default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--train-shards", type=str,
                        default="https://huggingface.co/datasets/pixparse/cc3m-wds/resolve/main/cc3m-train-{0000..0575}.tar")
    parser.add_argument("--output-dir", type=str, default="./global_adapter_ckpt")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-train-steps", type=int, default=100000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)

    parser.add_argument("--num-global-tokens", type=int, default=4)

    parser.add_argument("--drop-txt-prob", type=float, default=0.5)
    parser.add_argument("--drop-global-prob", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"])

    args = parser.parse_args()
    return TrainArgs(**vars(args))


def main():
    args = parse_args()

    # =======================
    # Accelerator (multi-GPU)
    # =======================
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=logging_dir,
        ),
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    # Chọn dtype cho model frozen
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # =======================
    # Load SD components
    # =======================
    accelerator.print("Loading Stable Diffusion components...")

    # Text
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
    )

    # VAE
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )

    # UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )

    # CLIP Vision for global image embedding
    accelerator.print("Loading CLIP Vision...")
    clip_image_processor = CLIPImageProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )
    clip_vision = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    # =======================
    # Freeze SD + CLIP Vision
    # =======================
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    clip_vision.requires_grad_(False)

    # Đưa frozen models sang device + dtype tối ưu
    vae.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    unet.to(device, dtype=weight_dtype)
    clip_vision.to(device, dtype=weight_dtype)

    # =======================
    # Create GlobalAdapter (trainable)
    # =======================
    global_adapter = GlobalAdapter(
        in_dim=768,
        out_dim=768,
        num_tokens=args.num_global_tokens,
        hidden_mult=4,
    )

    num_trainable = sum(p.numel() for p in global_adapter.parameters()
                        if p.requires_grad)
    accelerator.print(f"GlobalAdapter trainable params: {num_trainable / 1e6:.2f} M")

    # =======================
    # Optimizer
    # =======================
    optimizer = torch.optim.AdamW(
        global_adapter.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay,
    )

    # =======================
    # Dataset & DataLoader
    # =======================
    dataset = StreamingCC3MDataset(
        shards=args.train_shards,
        resolution=args.resolution,
        drop_txt_prob=0.0,      # dropout text làm trong training loop
        shuffle_buffer=2048,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Image transform cho VAE ([-1, 1])
    image_transform = transforms.Compose([
        transforms.Resize(
            (args.resolution, args.resolution),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),           # [0,1]
        transforms.Normalize([0.5], [0.5]),  # -> [-1,1]
    ])

    # =======================
    # Chuẩn bị với Accelerator (DDP)
    # =======================
    global_adapter, optimizer = accelerator.prepare(
        global_adapter, optimizer
    )

    unet.eval()
    vae.eval()
    text_encoder.eval()
    clip_vision.eval()
    global_adapter.train()

    global_step = 0
    if accelerator.is_main_process:
        progress_bar = tqdm(total=args.max_train_steps, desc="Training")
    else:
        progress_bar = None

    # =======================
    # Training loop
    # =======================
    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            images = batch["images"]  # list[PIL]
            texts = batch["texts"]    # list[str]

            # Nếu batch cuối nhỏ hơn, bỏ qua cho đơn giản
            if len(images) != args.train_batch_size:
                continue

            B = len(images)

            # =====================================================
            # DROPOUT: text & global condition (image embedding)
            # =====================================================
            drop_global_flags = []
            for i in range(B):
                # Drop caption?
                if random.random() < args.drop_txt_prob:
                    texts[i] = ""  # giống Uni-ControlNet: anno = ""

                # Drop global cond?
                drop_g = (random.random() < args.drop_global_prob)
                drop_global_flags.append(drop_g)

            with accelerator.accumulate(global_adapter):
                # ========== Chuẩn bị data ==========
                # VAE input
                pixel_values = torch.stack(
                    [image_transform(im) for im in images]
                ).to(device, dtype=weight_dtype)  # (B, 3, H, W)

                # Text input (sau khi đã drop caption)
                text_inputs = tokenizer(
                    texts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)

                # CLIP Vision input
                clip_inputs = clip_image_processor(
                    images=images,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    # Encode text
                    text_outputs = text_encoder(**text_inputs)
                    text_embeds = text_outputs.last_hidden_state  # (B, T_txt, 768)

                    # Encode image to latent (VAE)
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor  # ~0.18215

                    # CLIP Vision embedding
                    clip_outputs = clip_vision(**clip_inputs)
                    image_embeds = clip_outputs.image_embeds  # (B, 768)

                    # Dropout global: zero hóa embedding cho sample bị drop
                    if any(drop_global_flags):
                        mask = torch.tensor(drop_global_flags, device=device, dtype=torch.bool)
                        image_embeds[mask] = 0.0

                # Global tokens từ ảnh (sau dropout)
                # (GlobalAdapter được accelerate wrap, nên phải dùng accelerator.device dtype float32)
                image_embeds_for_adapter = image_embeds.to(global_adapter.parameters().__next__().dtype)
                global_tokens = global_adapter(image_embeds_for_adapter)  # (B, N_tok, 768)

                # Concatenate global tokens vào text embeddings
                encoder_hidden_states = torch.cat(
                    [text_embeds, global_tokens.to(text_embeds.dtype)], dim=1
                )  # (B, T_txt + N_tok, 768)

                # Sample noise + timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )

                noisy_latents = noise_scheduler.add_noise(
                    latents, noise, timesteps
                )

                # ========== Forward UNet + loss ==========
                with accelerator.autocast():
                    model_output = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample

                    loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item())

                # Save định kỳ
                if global_step % 10000 == 0:
                    save_path = os.path.join(
                        args.output_dir, f"global_adapter_step_{global_step}.pt"
                    )
                    unwrapped = accelerator.unwrap_model(global_adapter)
                    torch.save(unwrapped.state_dict(), save_path)
                    print(f"\nSaved adapter to {save_path}")

            if global_step >= args.max_train_steps:
                break

    # Save final
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(args.output_dir, "global_adapter_final.pt")
        unwrapped = accelerator.unwrap_model(global_adapter)
        torch.save(unwrapped.state_dict(), final_path)
        print(f"Training done. Final adapter saved to {final_path}")

    try:
        accelerator.end_training()
    except Exception:
        pass


if __name__ == "__main__":
    main()
