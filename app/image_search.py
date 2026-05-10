import io
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageSearchEngine:
    model = None
    preprocess = None
    tokenizer = None
    signature = None

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        self.load_model(model_name, pretrained)
        self.vectors = None
        self.index_ids = []

    @classmethod
    def load_model(cls, model_name: str, pretrained: str) -> None:
        if cls.model is not None and cls.signature == (model_name, pretrained):
            return
        cls.model, _, cls.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=DEVICE)
        cls.model.eval()
        cls.tokenizer = open_clip.get_tokenizer(model_name)
        cls.signature = (model_name, pretrained)

    @property
    def is_ready(self) -> bool:
        return self.vectors is not None and len(self.index_ids) > 0

    def encode_image_bytes(self, data: bytes | io.BytesIO) -> np.ndarray:
        if isinstance(data, bytes):
            data = io.BytesIO(data)
        tensor = self.preprocess(Image.open(data).convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()

    def encode_text(self, text: str) -> np.ndarray:
        tokens = self.tokenizer([text]).to(DEVICE)
        with torch.no_grad():
            feat = self.model.encode_text(tokens)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat[0].cpu().numpy()

    def build_index(self, image_paths: list[str | Path], row_indices: list[int], batch_size: int = 32) -> int:
        vecs, ids = [], []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_ids = row_indices[i:i + batch_size]
            images = [self.preprocess(Image.open(p).convert("RGB")) for p in batch_paths]
            with torch.no_grad():
                feats = self.model.encode_image(torch.stack(images).to(DEVICE))
                feats /= feats.norm(dim=-1, keepdim=True)
            vecs.append(feats.cpu().numpy())
            ids.extend(batch_ids)
        self.vectors = np.vstack(vecs)
        self.index_ids = ids
        return len(ids)

    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors, index_ids=np.array(self.index_ids, dtype=np.int64))

    def load_index(self, path: str | Path) -> bool:
        data = np.load(path, allow_pickle=False)
        self.vectors = data["vectors"]
        self.index_ids = [int(x) for x in data["index_ids"].tolist()]
        return self.is_ready

    def clear_index(self) -> None:
        self.vectors = None
        self.index_ids = []

    def search_vector(self, vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        scores = self.vectors @ vector
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.index_ids[i], float(scores[i])) for i in idx]

    def search_by_image(self, data: bytes | io.BytesIO, top_k: int = 5) -> list[tuple[int, float]]:
        return self.search_vector(self.encode_image_bytes(data), top_k)

    def search_by_text(self, text: str, top_k: int = 5) -> list[tuple[int, float]]:
        return self.search_vector(self.encode_text(text), top_k)

    def search_hybrid(self, data: bytes | io.BytesIO, text: str, top_k: int = 5, alpha: float = 0.5) -> list[tuple[int, float]]:
        v = alpha * self.encode_image_bytes(data) + (1 - alpha) * self.encode_text(text)
        return self.search_vector(v / np.linalg.norm(v), top_k)
