import io
import random
import numpy as np
import webdataset as wds

from torch.utils.data import IterableDataset

from .util import keep_and_drop  # giống như UniDataset cũ

# CLIP để làm global condition từ ảnh
import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class StreamingUniDataset(IterableDataset):
    """
    Streaming UniDataset dùng WebDataset.
    Mỗi sample trong shard cần có các key:
      - "jpg" : ảnh gốc
      - "txt" : caption (str hoặc bytes)

    Global condition sẽ được tính trực tiếp từ ảnh bằng CLIP Vision
    (không đọc từ "global_{name}" trong tar nữa).
    """

    def __init__(
        self,
        shards,
        local_type_list,
        global_type_list,   # giữ lại cho hợp config, nhưng không dùng trực tiếp
        resolution,
        drop_txt_prob,
        keep_all_cond_prob,
        drop_all_cond_prob,
        drop_each_cond_prob,
        shuffle_buffer=1000,
    ):
        super().__init__()
        self.shards = shards
        self.local_type_list = local_type_list
        self.global_type_list = global_type_list
        self.resolution = resolution
        self.drop_txt_prob = drop_txt_prob
        self.keep_all_cond_prob = keep_all_cond_prob
        self.drop_all_cond_prob = drop_all_cond_prob
        self.drop_each_cond_prob = drop_each_cond_prob
        self.shuffle_buffer = shuffle_buffer

        # Định nghĩa pipeline WebDataset (nhưng không iterate ở đây)
        self._dataset = (
            wds.WebDataset(self.shards, resampled=True)  # stream vô hạn
            .shuffle(self.shuffle_buffer)                # shuffle trong buffer
            .decode("pil")                               # image -> PIL.Image
        )

        # CLIP Vision để tính global condition từ ảnh
        # NOTE: để an toàn bạn có thể chọn model khác, nhưng
        # giả định projection output có thể map về 768-dim
        self.clip_processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.clip_vision.eval()
        for p in self.clip_vision.parameters():
            p.requires_grad = False

    def _image_to_global_cond(self, img_pil):
        """Tính global condition 768-dim từ ảnh dùng CLIP Vision."""
        # img_pil: PIL.Image (đã decode từ WebDataset)
        with torch.no_grad():
            inputs = self.clip_processor(images=img_pil, return_tensors="pt")
            outputs = self.clip_vision(**inputs)
            # lấy image_embeds (batch, D)
            emb = outputs.image_embeds[0]  # (D,)
            emb = emb.cpu().numpy().astype(np.float32)

        # Đảm bảo chiều = 768 (khớp in_dim trong global_control_config)
        D = emb.shape[0]
        TARGET_D = 768
        if D > TARGET_D:
            emb = emb[:TARGET_D]
        elif D < TARGET_D:
            emb = np.pad(emb, (0, TARGET_D - D), mode="constant")

        return emb  # shape (768,)

    def __iter__(self):
        import cv2  # nếu bạn không dùng thì có thể bỏ import này

        for sample in self._dataset:
            # ------------------------
            # 1. Ảnh chính
            # ------------------------
            img = sample["jpg"]  # PIL.Image do decode("pil") trả về
            img = img.resize((self.resolution, self.resolution))
            image = np.array(img).astype(np.float32)
            image = (image / 127.5) - 1.0  # [-1, 1] giống code cũ

            # ------------------------
            # 2. Caption / anno
            # ------------------------
            anno = sample.get("txt", "")
            if isinstance(anno, bytes):
                anno = anno.decode("utf-8")

            # ------------------------
            # 3. Local conditions: mỗi cond là một ảnh
            # key: "local_{type}"
            # ------------------------
            local_conditions = []
            for name in self.local_type_list:
                key = f"local_{name}"
                if key not in sample:
                    continue

                cond_img = sample[key]  # PIL.Image
                cond_img = cond_img.resize((self.resolution, self.resolution))
                cond = np.array(cond_img).astype(np.float32) / 255.0
                local_conditions.append(cond)

            # ------------------------
            # 4. Global condition: dùng ảnh chính -> CLIP Vision embedding
            # ------------------------
            global_conditions = []

            # ở đây bỏ hẳn việc đọc "global_{name}" từ sample;
            # thay bằng embedding từ ảnh
            emb = self._image_to_global_cond(img)  # (768,)
            global_conditions.append(emb)

            # ------------------------
            # 5. Drop text / cond giống UniDataset cũ
            # ------------------------
            if random.random() < self.drop_txt_prob:
                anno = ""

            local_conditions = keep_and_drop(
                local_conditions,
                self.keep_all_cond_prob,
                self.drop_all_cond_prob,
                self.drop_each_cond_prob,
            )
            global_conditions = keep_and_drop(
                global_conditions,
                self.keep_all_cond_prob,
                self.drop_all_cond_prob,
                self.drop_each_cond_prob,
            )

            # Local: concat theo channel, shape (H, W, C_total)
            if len(local_conditions) != 0:
                local_conditions = np.concatenate(local_conditions, axis=2)
            else:
                local_conditions = np.zeros(
                    (self.resolution, self.resolution, 0), dtype=np.float32
                )

            # Global: giờ mỗi phần tử là vector (768,)
            if len(global_conditions) != 0:
                # vì mỗi sample chỉ có 1 emb, nhưng để tổng quát cứ concat
                global_conditions = np.concatenate(global_conditions, axis=0)
            else:
                global_conditions = np.zeros((768,), dtype=np.float32)

            yield dict(
                jpg=image,
                txt=anno,
                local_conditions=local_conditions,
                global_conditions=global_conditions,
            )
