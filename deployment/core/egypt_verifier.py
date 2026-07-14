"""
=============================================================================
Egypt Verifier — Embedding-Based Identity Verification
=============================================================================
Egypt is the priority target. When the classifier reports Egypt in the top-3,
this module runs an additional verification step using the 512-D embedding
from the classifier against a precomputed database of Egypt embeddings.

The database combines:
    1. Synthetic clean renders (generated offline, no augmentation)
    2. Real camera frames from eg_cam_data/

All embeddings are precomputed offline and stored as a .npz file.
At runtime, only a single matrix-vector multiply is needed.

Usage (online):
    >>> verifier = EgyptVerifier(cfg)
    >>> result = verifier.verify(embedding_512d)
    >>> if result.is_egypt: ...

Usage (offline — database building):
    >>> verifier = EgyptVerifier(cfg)
    >>> verifier.build_database(classifier)
=============================================================================
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .similarity import SimilarityMethod, create_similarity_method

logger = logging.getLogger(__name__)

# Image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# =========================================================================
# Verification Result
# =========================================================================

@dataclass
class EgyptVerification:
    """Result of Egypt embedding verification."""
    is_egypt: bool = False
    similarity_score: float = 0.0
    method_used: str = ""
    threshold: float = 0.0


# =========================================================================
# Egypt Verifier
# =========================================================================

class EgyptVerifier:
    """
    Verifies Egypt identity using embedding similarity against a precomputed
    database. The database is built offline and loaded at startup.

    Parameters are loaded from config under 'egypt_verification'.
    """

    def __init__(self, cfg: dict):
        ev_cfg = cfg.get("egypt_verification", {})

        self.enabled = ev_cfg.get("enabled", True)
        self.threshold = ev_cfg.get("similarity_threshold", 0.75)
        self.top_k = ev_cfg.get("top_k", 20)
        self.egypt_class_name = ev_cfg.get("egypt_class_name", "Egypt")

        # Similarity method
        method_name = ev_cfg.get("similarity_method", "weighted_top_k")
        self.similarity: SimilarityMethod = create_similarity_method(
            method_name, top_k=self.top_k
        )

        # Database (loaded later)
        self._db_embeddings: Optional[np.ndarray] = None
        self._db_loaded = False

        # Try to load database from config path
        paths = cfg.get("paths", {})
        db_path = paths.get("egypt_db", "")
        if db_path and os.path.exists(db_path):
            self.load_database(db_path)

    # ------------------------------------------------------------------
    # Online: Verification
    # ------------------------------------------------------------------

    def verify(self, embedding: np.ndarray) -> EgyptVerification:
        """
        Verify if an embedding matches Egypt.

        Parameters
        ----------
        embedding : np.ndarray of shape (512,)
            L2-normalised embedding from the classifier.

        Returns
        -------
        EgyptVerification with is_egypt flag and similarity score.
        """
        if not self.enabled or not self._db_loaded:
            return EgyptVerification(
                is_egypt=False,
                similarity_score=0.0,
                method_used="disabled",
                threshold=self.threshold,
            )

        # Compute similarity
        score = self.similarity.compute(embedding, self._db_embeddings)

        return EgyptVerification(
            is_egypt=score >= self.threshold,
            similarity_score=score,
            method_used=type(self.similarity).__name__,
            threshold=self.threshold,
        )

    @property
    def is_loaded(self) -> bool:
        return self._db_loaded

    # ------------------------------------------------------------------
    # Database Loading
    # ------------------------------------------------------------------

    def load_database(self, path: str) -> None:
        """Load precomputed Egypt embeddings from a .npz file."""
        try:
            data = np.load(path, allow_pickle=True)
            self._db_embeddings = data["embeddings"].astype(np.float32)

            # Ensure L2-normalised
            norms = np.linalg.norm(self._db_embeddings, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            self._db_embeddings = self._db_embeddings / norms

            self._db_loaded = True
            logger.info(
                f"Egypt DB loaded: {self._db_embeddings.shape[0]} embeddings "
                f"from {path}"
            )

            # Log sources breakdown if available
            if "sources" in data:
                sources = data["sources"]
                unique, counts = np.unique(sources, return_counts=True)
                for src, cnt in zip(unique, counts):
                    logger.info(f"  {src}: {cnt} embeddings")

        except Exception as e:
            logger.error(f"Failed to load Egypt DB from {path}: {e}")
            self._db_loaded = False

    # ------------------------------------------------------------------
    # Offline: Database Building
    # ------------------------------------------------------------------

    def build_database(
        self,
        classifier,
        cfg: dict,
        output_path: str = "./embeddings/egypt_db.npz",
    ) -> None:
        """
        Build the Egypt embedding database offline.

        1. Generate clean synthetic Egypt flag images (no augmentation)
        2. Load real camera frames from eg_cam_data
        3. Extract 512-D embeddings using the classifier
        4. Save as .npz

        Parameters
        ----------
        classifier : FlagClassifier
            The ONNX classifier instance (for embedding extraction).
        cfg : dict
            Full config dict (for paths and generator settings).
        output_path : str
            Where to save the .npz file.
        """
        db_cfg = cfg.get("egypt_database", {})

        all_embeddings = []
        all_sources = []

        # --- 1. Synthetic images ---
        synthetic_count = db_cfg.get("synthetic_count", 200)
        gen_config_path = db_cfg.get("generator_config", "")
        flags_dir = db_cfg.get("flags_dir", "")
        backgrounds_dir = db_cfg.get("backgrounds_dir", "")

        if gen_config_path and flags_dir and os.path.exists(gen_config_path):
            logger.info(f"Generating {synthetic_count} synthetic Egypt images...")
            synthetic_images = self._generate_synthetic_egypt(
                gen_config_path, flags_dir, backgrounds_dir, synthetic_count
            )
            logger.info(f"Generated {len(synthetic_images)} synthetic images")

            for img in synthetic_images:
                emb = classifier.extract_embedding(img)
                all_embeddings.append(emb)
                all_sources.append("synthetic")
        else:
            logger.warning(
                f"Generator config not found at '{gen_config_path}'. "
                "Skipping synthetic Egypt images."
            )

        # --- 2. Real camera frames ---
        real_cam_dir = db_cfg.get("real_cam_dir", "")
        if real_cam_dir and os.path.isdir(real_cam_dir):
            logger.info(f"Loading real camera frames from {real_cam_dir}...")
            real_images = self._load_real_images(real_cam_dir)
            logger.info(f"Loaded {len(real_images)} real camera frames")

            for img in real_images:
                emb = classifier.extract_embedding(img)
                all_embeddings.append(emb)
                all_sources.append("real_cam")
        else:
            logger.warning(
                f"Real camera directory not found at '{real_cam_dir}'. "
                "Skipping real Egypt images."
            )

        if not all_embeddings:
            logger.error("No images found for Egypt DB — aborting.")
            return

        # --- 3. Save ---
        embeddings_matrix = np.stack(all_embeddings).astype(np.float32)
        sources_array = np.array(all_sources)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez(
            output_path,
            embeddings=embeddings_matrix,
            sources=sources_array,
        )
        logger.info(
            f"Egypt DB saved: {embeddings_matrix.shape[0]} embeddings → {output_path}"
        )

    # ------------------------------------------------------------------
    # Internal: Synthetic Generation (no augmentation)
    # ------------------------------------------------------------------

    def _generate_synthetic_egypt(
        self,
        gen_config_path: str,
        flags_dir: str,
        backgrounds_dir: str,
        count: int,
    ) -> list:
        """
        Generate clean synthetic Egypt flag crops using the rendering pipeline.
        No augmentation — pure perspective-projected flag on background.

        Uses the VirtualCamera and rendering logic from generate_target_img.py
        but with all augmentation disabled.
        """
        import random
        import yaml

        # Load generator config
        with open(gen_config_path, "r") as f:
            gen_cfg = yaml.safe_load(f)

        # Find Egypt flag textures
        egypt_dir = os.path.join(flags_dir, "Egypt")
        if not os.path.isdir(egypt_dir):
            logger.error(f"Egypt flag directory not found: {egypt_dir}")
            return []

        egypt_textures = [
            os.path.join(egypt_dir, f)
            for f in os.listdir(egypt_dir)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        ]
        if not egypt_textures:
            logger.error("No Egypt flag textures found")
            return []

        # Load backgrounds
        bg_images = []
        if backgrounds_dir and os.path.isdir(backgrounds_dir):
            for f in os.listdir(backgrounds_dir):
                if os.path.splitext(f)[1].lower() in _IMG_EXTS:
                    bg_path = os.path.join(backgrounds_dir, f)
                    bg = cv2.imread(bg_path)
                    if bg is not None:
                        bg_images.append(bg)

        if not bg_images:
            # Create a simple sandy background
            bg_images = [np.full((1296, 2304, 3), (140, 180, 200), dtype=np.uint8)]
            logger.warning("No backgrounds found, using plain sandy color")

        # Camera parameters from generator config
        output_size = tuple(gen_cfg.get("output", {}).get("image_size", [2304, 1296]))
        crop_cfg = gen_cfg.get("crop", {})
        crop_output_size = crop_cfg.get("output_size", 224)
        flag_w_m = gen_cfg.get("flag", {}).get("width_m", 2.0)
        flag_h_m = gen_cfg.get("flag", {}).get("height_m", 1.0)

        # Camera model constants
        SENSOR_W_MM = 6.954
        FOCAL_MM = 12.0
        NATIVE_W = 2304

        # Flight envelope
        flight_cfg = gen_cfg.get("flight", {})
        alt_min = flight_cfg.get("altitude_min_m", 60)
        alt_max = flight_cfg.get("altitude_max_m", 80)

        images = []
        for i in range(count):
            # Random altitude
            altitude = random.uniform(alt_min, alt_max)

            # GSD
            gsd = (SENSOR_W_MM * altitude) / (FOCAL_MM * NATIVE_W)

            # Flag size in pixels
            flag_px_w = flag_w_m / gsd
            flag_px_h = flag_h_m / gsd

            # Random flag texture
            tex_path = random.choice(egypt_textures)
            texture = cv2.imread(tex_path, cv2.IMREAD_UNCHANGED)
            if texture is None:
                continue

            # Resize texture to flag pixel size
            flag_img = cv2.resize(
                texture, (int(flag_px_w), int(flag_px_h)),
                interpolation=cv2.INTER_AREA,
            )

            # If RGBA, composite on background; if RGB, use directly
            if flag_img.shape[2] == 4:
                flag_bgr = flag_img[:, :, :3]
                flag_alpha = flag_img[:, :, 3:] / 255.0
            else:
                flag_bgr = flag_img
                flag_alpha = np.ones(
                    (flag_img.shape[0], flag_img.shape[1], 1), dtype=np.float32
                )

            # Random background crop
            bg = random.choice(bg_images)
            bg_h, bg_w = bg.shape[:2]
            scene_w, scene_h = output_size

            # Resize background to scene size
            bg_resized = cv2.resize(bg, (scene_w, scene_h))

            # Random position for flag on scene
            fh, fw = flag_bgr.shape[:2]
            max_x = max(1, scene_w - fw)
            max_y = max(1, scene_h - fh)
            ox = random.randint(0, max_x)
            oy = random.randint(0, max_y)

            # Composite flag onto background
            scene = bg_resized.copy()
            roi = scene[oy:oy + fh, ox:ox + fw]
            blended = (flag_bgr.astype(np.float32) * flag_alpha +
                       roi.astype(np.float32) * (1.0 - flag_alpha))
            scene[oy:oy + fh, ox:ox + fw] = blended.astype(np.uint8)

            # Extract crop with small margin (simulating detector bbox)
            margin_ratio = random.uniform(
                crop_cfg.get("margin_ratio_min", 0.002),
                crop_cfg.get("margin_ratio_max", 0.005),
            )
            margin_x = int(fw * margin_ratio)
            margin_y = int(fh * margin_ratio)

            cx1 = max(0, ox - margin_x)
            cy1 = max(0, oy - margin_y)
            cx2 = min(scene_w, ox + fw + margin_x)
            cy2 = min(scene_h, oy + fh + margin_y)

            crop = scene[cy1:cy2, cx1:cx2]

            # Resize to classifier input size
            crop = cv2.resize(crop, (crop_output_size, crop_output_size))
            images.append(crop)

        return images

    # ------------------------------------------------------------------
    # Internal: Real Image Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_real_images(directory: str) -> list:
        """Load all images from a directory."""
        images = []
        for f in sorted(os.listdir(directory)):
            if os.path.splitext(f)[1].lower() in _IMG_EXTS:
                path = os.path.join(directory, f)
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
        return images
