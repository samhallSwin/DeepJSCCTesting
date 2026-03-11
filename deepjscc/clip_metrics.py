"""Optional CLIP-based image similarity metrics for reconstruction assessment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CLIPImageSimilarity:
    """Compute cosine similarity between CLIP image embeddings."""

    model_id: str = "openai/clip-vit-base-patch32"
    device: str | None = None

    def __post_init__(self):
        try:
            import torch
            from transformers import AutoProcessor, CLIPModel
        except ModuleNotFoundError as exc:
            raise ImportError(
                "CLIP metrics require optional dependencies. Install with: "
                "pip install -r requirements.txt -r requirements-clip.txt"
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "CLIP dependencies are installed, but PyTorch failed to load correctly. "
                "This usually means the installed torch build is incompatible with the local "
                "CUDA runtime/driver. Original error: "
                f"{exc}"
            ) from exc

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = CLIPModel.from_pretrained(self.model_id)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self.device)
        self._model.eval()

    def score_batch(self, reference_images, candidate_images) -> np.ndarray:
        """Return per-image cosine similarities in [-1, 1]."""
        ref = self._to_uint8_numpy(reference_images)
        cand = self._to_uint8_numpy(candidate_images)
        with self._torch.inference_mode():
            ref_inputs = self._processor(images=list(ref), return_tensors="pt")
            cand_inputs = self._processor(images=list(cand), return_tensors="pt")
            ref_inputs = {k: v.to(self.device) for k, v in ref_inputs.items()}
            cand_inputs = {k: v.to(self.device) for k, v in cand_inputs.items()}
            ref_features = self._extract_image_features(ref_inputs)
            cand_features = self._extract_image_features(cand_inputs)
            ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            cand_features = cand_features / cand_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            scores = (ref_features * cand_features).sum(dim=-1)
        return scores.detach().cpu().numpy().astype(np.float32)

    def _extract_image_features(self, model_inputs):
        if hasattr(self._model, "get_image_features"):
            outputs = self._model.get_image_features(**model_inputs)
            if hasattr(outputs, "image_embeds"):
                return outputs.image_embeds
            if self._torch.is_tensor(outputs):
                return outputs

        if hasattr(self._model, "vision_model"):
            vision_outputs = self._model.vision_model(**model_inputs)
            if hasattr(vision_outputs, "pooler_output"):
                features = vision_outputs.pooler_output
            elif hasattr(vision_outputs, "last_hidden_state"):
                features = vision_outputs.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"Unsupported CLIP vision output type: {type(vision_outputs)!r}")

            if hasattr(self._model, "visual_projection"):
                features = self._model.visual_projection(features)
            return features

        raise TypeError("Model does not expose usable CLIP image feature extraction APIs.")

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
