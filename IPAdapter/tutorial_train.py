#!/usr/bin/env python
# coding=utf-8

import os
import io
import re
import json
import time
import random
import argparse
import itertools
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

# ---- IP-Adapter bits ----
from ip_adapter.ip_adapter import ImageProjModel, ImageProjMLP
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def enable_tf32():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def save_compact(accelerator, ip_adapter, outdir, tag):
    if accelerator.is_main_process:
        os.makedirs(outdir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(ip_adapter)
        image_proj_sd = {k: v.detach().cpu() for k, v in unwrapped.image_proj_model.state_dict().items()}
        ip_adapter_sd = {k: v.detach().cpu() for k, v in unwrapped.adapter_modules.state_dict().items()}
        torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_adapter_sd},
                   os.path.join(outdir, f"ip_adapter_{tag}.pt"))
        torch.save(image_proj_sd, os.path.join(outdir, f"image_proj_{tag}.bin"))
        torch.save(ip_adapter_sd, os.path.join(outdir, f"ip_adapter_{tag}.bin"))


# ------------------------------------------------------------
# Local JSON dataset
# ------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, tokenizer, size=512,
                 t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, image_root_path=""):
        super().__init__()
        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.image_root_path = image_root_path

        with open(json_file, "r") as f:
            self.data = json.load(f)  # list of {"image_file": "...", "text": "..."}

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        image_file = item["image_file"]

        raw_image = Image.open(os.path.join(self.image_root_path, image_file)).convert("RGB")
        image = self.transform(raw_image)
        clip_image = self.clip_image_processor(images=raw_image, return_tensors="pt").pixel_values

        drop_image_embed = 0
        r = random.random()
        if r < self.i_drop_rate:
            drop_image_embed = 1
        elif r < (self.i_drop_rate + self.t_drop_rate):
            text = ""
        elif r < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            text = ""
            drop_image_embed = 1

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    images = torch.stack([example["image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    clip_images = torch.cat([example["clip_image"] for example in data], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in data]
    return {
        "images": images,
        "text_input_ids": text_input_ids,
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds
    }


# ------------------------------------------------------------
# WebDataset streaming (tránh OOM)
# ------------------------------------------------------------
def build_streaming_iter(
    urls_pattern,
    tokenizer,
    size=512,
    batch_size=4,
    num_workers=2,
    i_drop_rate=0.05,
    t_drop_rate=0.05,
    ti_drop_rate=0.05,
    clip_image_processor=None,
    shuffle_buffer=1000,
    prefetch_factor=2,
):
    try:
        import webdataset as wds
        from webdataset import handlers
    except ImportError as e:
        raise RuntimeError("Bạn cần cài webdataset: pip install webdataset") from e

    if clip_image_processor is None:
        clip_image_processor = CLIPImageProcessor()

    img_tx = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def _process_sample(sample):
        # sample["img"] là bytes -> mở bằng PIL
        raw = Image.open(io.BytesIO(sample["img"])).convert("RGB")
        image = img_tx(raw)
        clip_image = clip_image_processor(images=raw, return_tensors="pt").pixel_values[0]

        text = sample["caption"]
        drop_image_embed = 0
        r = random.random()
        if r < i_drop_rate:
            drop_image_embed = 1
        elif r < (i_drop_rate + t_drop_rate):
            text = ""
        elif r < (i_drop_rate + t_drop_rate + ti_drop_rate):
            text = ""
            drop_image_embed = 1

        text_input_ids = tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids[0]

        return {
            "image": image,
            "text_input_ids": text_input_ids,
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed
        }

    # Dùng DataPipeline để chèn split_by_node/worker rõ ràng
    dataset = wds.DataPipeline(
        wds.SimpleShardList(urls_pattern),
        wds.split_by_node,
        wds.split_by_worker,

        # Khi có lỗi đọc tar (như curl 18), handler sẽ cảnh báo và NHẢY QUA shard đó
        wds.tarfile_to_samples(handler=handlers.warn_and_continue),

        wds.shuffle(shuffle_buffer),

        wds.map(lambda s: {
            "img": s.get("jpg", s.get("jpeg", s.get("png"))),
            "caption": (
                s.get("txt", b"").decode("utf-8", errors="ignore")
                if isinstance(s.get("txt", b""), (bytes, bytearray))
                else (s.get("txt") or "")
            ),
        }),
        wds.select(lambda s: s["img"] is not None),
        wds.map(_process_sample),

        # batching ở đây để handler có thể bỏ qua sample lỗi trước khi pack batch
        wds.batched(batch_size, partial=False),
    )

    loader = wds.WebLoader(
        dataset,
        num_workers=num_workers,
        batch_size=None,                  # không double-batch
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )

    def _merge_batch(batch):
        # case A: list of dicts
        if isinstance(batch, list):
            images = torch.stack([b["image"] for b in batch], dim=0)
            text_input_ids = torch.stack([b["text_input_ids"] for b in batch], dim=0)
            clip_images = torch.stack([b["clip_image"] for b in batch], dim=0)
            drop_image_embeds = [b["drop_image_embed"] for b in batch]
            return {
                "images": images,
                "text_input_ids": text_input_ids,
                "clip_images": clip_images,
                "drop_image_embeds": drop_image_embeds
            }
        # case B: dict of lists/tensors
        if isinstance(batch, dict):
            def _to_tensor_stack(val):
                if isinstance(val, list):
                    return torch.stack(val, dim=0)
                return val
            images = _to_tensor_stack(batch["image"])
            text_input_ids = _to_tensor_stack(batch["text_input_ids"])
            clip_images = _to_tensor_stack(batch["clip_image"])
            drop_image_embeds = batch["drop_image_embed"]
            if torch.is_tensor(drop_image_embeds):
                drop_image_embeds = drop_image_embeds.tolist()
            return {
                "images": images,
                "text_input_ids": text_input_ids,
                "clip_images": clip_images,
                "drop_image_embeds": drop_image_embeds
            }
        raise TypeError(f"Unexpected batch type: {type(batch)}")

    def _iter():
        for b in loader:
            yield _merge_batch(b)

    return _iter


# ------------------------------------------------------------
# IP-Adapter module wrapper
# ------------------------------------------------------------
class IPAdapter(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, image_embeds):
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"
        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


# ------------------------------------------------------------
# Argparse
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train IP-Adapter with local JSON or WebDataset streaming.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_ip_adapter_path", type=str, default=None)
    parser.add_argument("--image_encoder_path", type=str, required=True)

    # Local JSON mode
    parser.add_argument("--data_json_file", type=str, default=None, help="JSON file (local) for training")
    parser.add_argument("--data_root_path", type=str, default="", help="root folder of images (local)")

    # WebDataset mode
    parser.add_argument("--webdataset_urls", type=str, default=None,
                        help='Pattern URLs for shards, e.g. "data/cc3m-{0000..0999}.tar" or "https://.../cc3m-{0000..0999}.tar"')
    parser.add_argument("--steps_per_epoch", type=int, default=1000,
                        help="Only used for streaming mode to limit steps per epoch.")

    # Train params
    parser.add_argument("--output_dir", type=str, default="sd-ip_adapter")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--seed", type=int, default=42)

    # WebDataset loader knobs
    parser.add_argument("--wds_shuffle", type=int, default=1000)
    parser.add_argument("--wds_prefetch", type=int, default=2)

    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    seed_everything(args.seed)
    enable_tf32()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load teacher / base models ---
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)

    # Freeze to save memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # --- IP-Adapter init ---
    # image_proj_model = ImageProjModel(
    #     cross_attention_dim=unet.config.cross_attention_dim,
    #     clip_embeddings_dim=image_encoder.config.projection_dim,
    #     clip_extra_context_tokens=4,
    # )
    image_proj_model = ImageProjMLP(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=image_encoder.config.projection_dim,
        clip_extra_context_tokens=4,
    )

    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)

    # --- Dtype / devices ---
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Frozen modules có thể ở fp16/bf16 (tiết kiệm VRAM)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    # >>> Trainable modules: LUÔN ở float32 <<<
    ip_adapter.image_proj_model.to(accelerator.device)
    ip_adapter.image_proj_model.float()
    ip_adapter.adapter_modules.to(accelerator.device)
    ip_adapter.adapter_modules.float()

    # --- Optimizer ---
    params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # --- Data ---
    use_streaming = args.webdataset_urls is not None and len(args.webdataset_urls) > 0
    if use_streaming:
        stream_iter_builder = build_streaming_iter(
            urls_pattern=args.webdataset_urls,
            tokenizer=tokenizer,
            size=args.resolution,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            clip_image_processor=CLIPImageProcessor(),
            shuffle_buffer=args.wds_shuffle,
            prefetch_factor=args.wds_prefetch,
        )
        steps_per_epoch = max(1, args.steps_per_epoch)

        # Bọc MODEL + OPTIMIZER bằng accelerate để DDP hoạt động đúng
        ip_adapter, optimizer = accelerator.prepare(ip_adapter, optimizer)

    else:
        assert args.data_json_file is not None and len(args.data_json_file) > 0, \
            "Missing --data_json_file for local dataset"
        train_dataset = MyDataset(
            args.data_json_file,
            tokenizer=tokenizer,
            size=args.resolution,
            image_root_path=args.data_root_path
        )
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
            pin_memory=True,
            persistent_workers=(args.dataloader_num_workers > 0),
            prefetch_factor=(2 if args.dataloader_num_workers > 0 else None),
        )
        ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    # --- Train loop ---
    global_step = 0
    for epoch in range(args.num_train_epochs):
        begin = time.perf_counter()

        if use_streaming:
            stream_iter = stream_iter_builder()  # refresh iterator mỗi epoch
            step_iter = range(steps_per_epoch)
        else:
            step_iter = enumerate(train_dataloader)

        for step_item in step_iter:
            load_data_time = time.perf_counter() - begin
            if use_streaming:
                step = step_item
                batch = next(stream_iter)
            else:
                step, batch = step_item

            with accelerator.accumulate(ip_adapter):
                # Encode image -> latents
                with torch.no_grad():
                    images = batch["images"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                    latents = vae.encode(images).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # Noise sampling
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # CLIP image embeds
                with torch.no_grad():
                    clip_images = batch["clip_images"].to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                    image_embeds = image_encoder(clip_images).image_embeds
                image_embeds_ = []
                for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
                    if drop_image_embed == 1:
                        image_embeds_.append(torch.zeros_like(image_embed))
                    else:
                        image_embeds_.append(image_embed)
                image_embeds = torch.stack(image_embeds_)

                # Text encoder
                with torch.no_grad():
                    text_input_ids = batch["text_input_ids"].to(accelerator.device, non_blocking=True)
                    encoder_hidden_states = text_encoder(text_input_ids)[0]

                # Forward IP-Adapter
                noise_pred = ip_adapter(noisy_latents, timesteps, encoder_hidden_states, image_embeds)

                # Loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.repeat(bsz)).mean().item()

                # Backward + step
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process:
                    print(f"Epoch {epoch}, step {step}, data_time: {load_data_time:.3f}, "
                          f"time: {time.perf_counter() - begin:.3f}, step_loss: {avg_loss:.6f}")

            global_step += 1
            if accelerator.is_main_process and (global_step % args.save_steps == 0):
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                save_compact(accelerator, ip_adapter, args.output_dir, f"step{global_step}")

            begin = time.perf_counter()

    # Save final
    accelerator.wait_for_everyone()
    save_compact(accelerator, ip_adapter, args.output_dir, "final")
    try:
        accelerator.end_training()
    except Exception:
        pass


if __name__ == "__main__":
    main()
