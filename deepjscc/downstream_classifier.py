"""Optional downstream EuroSAT land-cover classifier evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


EUROSAT_CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def _normalize_label(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


@dataclass
class EuroSATClassifier:
    """Pretrained EuroSAT land-cover classifier from Hugging Face / timm."""

    model_id: str = "cm93/resnet18-eurosat"
    device: str = "cpu"

    def __post_init__(self):
        try:
            import torch
            import timm
            from PIL import Image
            from timm.data import create_transform, resolve_model_data_config
        except ModuleNotFoundError as exc:
            raise ImportError(
                "Downstream classification requires optional dependencies. Install with: "
                "pip install -r requirements.txt && "
                "pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision && "
                "pip install -r requirements-downstream.txt"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Downstream classification dependencies are installed, but timm/torchvision failed "
                "to load correctly. This usually means torch and torchvision are incompatible. "
                "Reinstall matching CPU wheels with: "
                "pip uninstall -y torch torchvision torchaudio && "
                "pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision "
                f"Original error: {exc}"
            ) from exc

        self._torch = torch
        self._Image = Image
        self._timm = timm
        self._model = timm.create_model(f"hf_hub:{self.model_id}", pretrained=True)
        self._model.to(self.device)
        self._model.eval()
        try:
            data_config = resolve_model_data_config(self._model.pretrained_cfg, model=self._model)
        except TypeError:
            data_config = resolve_model_data_config(self._model)
        self._transform = create_transform(
            **data_config,
            is_training=False,
        )
        self.label_names = self._extract_label_names()

    def predict(self, images) -> np.ndarray:
        batch = self._prepare_batch(images)
        with self._torch.inference_mode():
            logits = self._model(batch)
        return logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)

    def _prepare_batch(self, images):
        images = self._to_uint8_numpy(images)
        tensors = []
        for image in images:
            pil_image = self._Image.fromarray(image)
            tensors.append(self._transform(pil_image))
        batch = self._torch.stack(tensors, dim=0)
        return batch.to(self.device)

    def _extract_label_names(self) -> list[str]:
        cfg = getattr(self._model, "pretrained_cfg", {}) or {}
        for key in ("label_names", "labels", "classes"):
            value = cfg.get(key)
            if isinstance(value, (list, tuple)) and value:
                return [str(x) for x in value]
        if getattr(self._model, "num_classes", None) == len(EUROSAT_CLASS_NAMES):
            return list(EUROSAT_CLASS_NAMES)
        raise ValueError(
            f"Could not determine label names for classifier '{self.model_id}'. "
            "Use a EuroSAT checkpoint with label metadata."
        )

    def build_label_mapping(self, dataset_class_names: Iterable[str]) -> np.ndarray:
        classifier_lookup = {_normalize_label(name): idx for idx, name in enumerate(self.label_names)}
        mapping = []
        for class_name in dataset_class_names:
            key = _normalize_label(class_name)
            if key not in classifier_lookup:
                raise ValueError(
                    f"Dataset class '{class_name}' not found in classifier labels {self.label_names}."
                )
            mapping.append(classifier_lookup[key])
        return np.asarray(mapping, dtype=np.int64)

    @staticmethod
    def _to_uint8_numpy(images) -> np.ndarray:
        try:
            import tensorflow as tf

            if tf.is_tensor(images):
                images = images.numpy()
        except Exception:
            pass
        images = np.asarray(images)
        if images.dtype != np.uint8:
            images = np.clip(images, 0.0, 1.0)
            images = (images * 255.0).round().astype(np.uint8)
        return images
