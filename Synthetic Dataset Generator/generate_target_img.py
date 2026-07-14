"""
=============================================================================
Synthetic Dataset Generator — 188-Country Flag Recognition (ConvNeXt V2 + ArcFace)
=============================================================================
Generates a train / val / test split dataset of realistic synthetic aerial
images of national flags, laid flat on terrain and viewed nadir by a UAV
camera.

Camera model: EZVIZ H6c Pro 2K (CS-H6c-R105-1L3WF) with a 12 mm M12 lens.
    Sensor:     1/2.7" CMOS → 6.954 mm × 3.562 mm (back-calculated from
                the datasheet-stated FOV with the 4 mm reference lens,
                then applied to the 12 mm replacement lens)
    Resolution: 2304 × 1296 px  (2 K)
    Focal:      12 mm  (M12 lens swap — user-specified)
    H-FOV:      32.32 °    V-FOV: 16.88 °    D-FOV: 36.07 °

    Datasheet source: H6c Pro 2K_datasheet — model CS-H6c-R105-1L3WF
        Native lens: 4 mm @ F1.6, M12, 1/2.7" Progressive Scan CMOS
        Native FOV:  98° diag / 82° horiz / 48° vert
        → sensor_w = 2 × 4 × tan(41°) = 6.954 mm
        → sensor_h = 2 × 4 × tan(24°) = 3.562 mm

Flight envelope (NEVER below 60 m):
    Altitude range : 60 m – 80 m
    Primary range  : 60 m – 65 m  (70 % weight)
    Camera pitch   : 0 ° (nadir — straight down)
    Camera roll    : Gaussian(0, 1.5°) clipped to ±3°

Asset layout expected:
    flags/
        Egypt/
            *.png
        France/
            *.png
        ...   (188 country folders)
    backgrounds/
        *.jpg
        *.png

Output:
    dataset/
        train/<country_name>/img_XXXXXX.jpg
        val/<country_name>/img_XXXXXX.jpg
        test/<country_name>/img_XXXXXX.jpg
    labels.json

Usage:
   python generate_target_img.py --config config.yaml --samples-per-country 10  
    python generate_target_img.py --help

Author: Synthetic Pipeline — UAV Flag Recognition
=============================================================================
"""

import os
import json
import math
import random
import argparse
import logging
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG HELPERS
# ============================================================================

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config: {path}")
    return cfg


def get_nested(cfg: dict, *keys, default=None):
    val = cfg
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, default)
        else:
            return default
    return val


# ============================================================================
# COUNTRY / CLASS DISCOVERY
# ============================================================================

class CountryRegistry:
    """
    Auto-discovers all country sub-folders under flags_dir.
    Folder names become class labels. Only folders containing
    at least one image file are registered.

    Expected layout:
        flags_dir/
            Egypt/
                *.png
            France/
                *.png
            ...
    """

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    def __init__(self, flags_dir: str):
        self.flags_dir = flags_dir
        self.countries: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.images: Dict[str, List[str]] = {}
        self._discover()

    def _discover(self):
        if not os.path.isdir(self.flags_dir):
            raise FileNotFoundError(
                f"Flags directory not found: '{self.flags_dir}'. "
                "Expected flags/<CountryName>/*.png structure."
            )
        found = []
        for entry in sorted(os.scandir(self.flags_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            imgs = [
                os.path.join(entry.path, f)
                for f in os.listdir(entry.path)
                if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS
            ]
            if imgs:
                found.append(entry.name)
                self.images[entry.name] = imgs

        self.countries = found
        self.class_to_idx = {name: idx for idx, name in enumerate(self.countries)}
        logger.info(
            f"Discovered {len(self.countries)} countries in '{self.flags_dir}'"
        )
        if not self.countries:
            raise RuntimeError("No valid country folders with images found.")

    def num_classes(self) -> int:
        return len(self.countries)

    def random_country(self) -> str:
        return random.choice(self.countries)

    def random_image(self, country: str) -> str:
        return random.choice(self.images[country])

    def save_labels(self, output_path: str):
        data = {
            "classes": self.countries,
            "class_to_idx": self.class_to_idx,
            "num_classes": self.num_classes(),
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved labels.json -> {output_path}")


# ============================================================================
# VIRTUAL CAMERA MODEL
# H6c Pro 2K  (CS-H6c-R105-1L3WF)  +  12 mm M12 lens
# ============================================================================

class VirtualCamera:
    """
    Physically-based camera model derived from the EZVIZ H6c Pro 2K datasheet.

    Sensor dimensions back-calculated from the datasheet-stated FOVs and
    the native 4 mm focal length:
        sensor_w = 2 * 4 * tan(82/2 deg) = 6.954 mm
        sensor_h = 2 * 4 * tan(48/2 deg) = 3.562 mm

    With the 12 mm M12 replacement lens:
        H-FOV = 2 * arctan(6.954 / (2*12)) = 32.32 deg
        V-FOV = 2 * arctan(3.562 / (2*12)) = 16.88 deg

    GSD & flag pixel size at training altitudes (2304 x 1296 native):
        60 m -> GSD = 1.51 cm/px  | 2x1 m flag -> 133 x 66 px
        65 m -> GSD = 1.63 cm/px  | 2x1 m flag -> 122 x 61 px
        70 m -> GSD = 1.76 cm/px  | 2x1 m flag -> 114 x 57 px
        80 m -> GSD = 2.01 cm/px  | 2x1 m flag ->  99 x 50 px
    """

    # --- Datasheet-derived sensor (back-calculated) ---
    SENSOR_W_MM: float = 6.954
    SENSOR_H_MM: float = 3.562
    NATIVE_W: int = 2304
    NATIVE_H: int = 1296
    FOCAL_MM: float = 12.0        # user-specified M12 replacement lens

    # Derived FOVs (informational)
    H_FOV_DEG: float = 32.32
    V_FOV_DEG: float = 16.88
    D_FOV_DEG: float = 36.07

    def __init__(self, output_size: Tuple[int, int]):
        """Args: output_size: (W, H) of the rendered scene in pixels."""
        self.output_size = output_size  # (W, H)

    # ------------------------------------------------------------------
    # GSD / footprint helpers
    # ------------------------------------------------------------------

    def compute_gsd(self, altitude_m: float) -> float:
        """Ground Sampling Distance in m/px (width axis)."""
        return (self.SENSOR_W_MM * altitude_m) / (self.FOCAL_MM * self.NATIVE_W)

    def compute_ground_footprint(self, altitude_m: float) -> Tuple[float, float]:
        """Ground area (m) visible in one full output frame."""
        gsd_w = (self.SENSOR_W_MM * altitude_m) / (self.FOCAL_MM * self.NATIVE_W)
        gsd_h = (self.SENSOR_H_MM * altitude_m) / (self.FOCAL_MM * self.NATIVE_H)
        return gsd_w * self.output_size[0], gsd_h * self.output_size[1]

    def flag_pixel_size(self, flag_w_m: float, flag_h_m: float,
                        altitude_m: float) -> Tuple[float, float]:
        """Expected flag size in pixels at this altitude."""
        gsd = self.compute_gsd(altitude_m)
        return flag_w_m / gsd, flag_h_m / gsd

    # ------------------------------------------------------------------
    # Perspective projection
    # ------------------------------------------------------------------

    def project_ground_plane(
        self,
        ground_pts: np.ndarray,
        altitude_m: float,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> np.ndarray:
        """
        Project 3-D ground-plane points (Z=0) to 2-D pixel coordinates.

        Camera mounted nadir (0 deg pitch = straight down). Intrinsic
        matrix is scaled from native to output resolution.

        Args:
            ground_pts : (N, 3) float64, world coords with Z=0
            altitude_m : Camera height above ground
            pitch_deg  : Pitch from nadir (0 = straight down)
            roll_deg   : Roll angle
            yaw_deg    : Heading angle

        Returns:
            (N, 2) pixel coordinates in output frame
        """
        sx = self.output_size[0] / self.NATIVE_W
        sy = self.output_size[1] / self.NATIVE_H
        fx = (self.FOCAL_MM / self.SENSOR_W_MM) * self.NATIVE_W * sx
        fy = (self.FOCAL_MM / self.SENSOR_H_MM) * self.NATIVE_H * sy
        cx = self.output_size[0] / 2.0
        cy = self.output_size[1] / 2.0

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], dtype=np.float64)

        pr = math.radians(pitch_deg)
        rr = math.radians(roll_deg)
        yr = math.radians(yaw_deg)

        Rx = np.array([[1,           0,            0],
                       [0, math.cos(pr), -math.sin(pr)],
                       [0, math.sin(pr),  math.cos(pr)]])
        Ry = np.array([[ math.cos(rr), 0, math.sin(rr)],
                       [0,             1,             0],
                       [-math.sin(rr), 0, math.cos(rr)]])
        Rz = np.array([[math.cos(yr), -math.sin(yr), 0],
                       [math.sin(yr),  math.cos(yr), 0],
                       [0,             0,             1]])

        # Nadir orientation: camera -Z points toward ground
        R_nadir = np.array([[1,  0,  0],
                             [0, -1,  0],
                             [0,  0, -1]], dtype=np.float64)

        R = Rz @ Ry @ Rx @ R_nadir
        t = np.array([[0], [0], [altitude_m]], dtype=np.float64)
        Rt = np.hstack([R, -R @ t])

        pts_h = np.hstack([ground_pts, np.ones((len(ground_pts), 1))])
        proj = (K @ Rt @ pts_h.T).T
        eps = 1e-8
        return np.column_stack([proj[:, 0] / (proj[:, 2] + eps),
                                 proj[:, 1] / (proj[:, 2] + eps)])


# ============================================================================
# BACKGROUND LOADER
# ============================================================================

class BackgroundLoader:
    """Loads real terrain backgrounds with random crop + colour jitter."""

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self, bg_dir: str, output_size: Tuple[int, int]):
        self.output_size = output_size  # (W, H)
        self.paths: List[str] = []
        if os.path.isdir(bg_dir):
            self.paths = [
                os.path.join(bg_dir, f)
                for f in os.listdir(bg_dir)
                if os.path.splitext(f)[1].lower() in self.IMAGE_EXTS
            ]
        if not self.paths:
            raise FileNotFoundError(
                f"No background images found in '{bg_dir}'. "
                "Place .jpg/.png terrain images there."
            )
        logger.info(f"Found {len(self.paths)} background images in '{bg_dir}'")
        
        # Pre-load/cache background images in RAM to eliminate disk reads/decodes during loop
        logger.info("Pre-loading backgrounds into memory...")
        self.bg_images: List[np.ndarray] = []
        for path in self.paths:
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is not None:
                self.bg_images.append(img)
        logger.info(f"Loaded {len(self.bg_images)} background images into RAM.")

    def load_crop(self, cx1: int, cy1: int, cx2: int, cy2: int) -> np.ndarray:
        """Loads a random background crop matching the target region, avoiding full-res rendering."""
        img = random.choice(self.bg_images)
        out_w, out_h = self.output_size
        h, w = img.shape[:2]
        
        if h >= out_h and w >= out_w:
            bx = random.randint(0, w - out_w)
            by = random.randint(0, h - out_h)
            crop_bg = img[by + cy1:by + cy2, bx + cx1:bx + cx2].copy()
        else:
            resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            crop_bg = resized[cy1:cy2, cx1:cx2].copy()
            
        return self._colour_jitter(crop_bg)

    @staticmethod
    def _colour_jitter(img: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-5, 5)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.88, 1.12), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.88, 1.12), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ============================================================================
# FLAG TEXTURE LOADER
# ============================================================================

class FlagTextureLoader:
    """
    Loads a flag image from disk and resizes it to the required pixel size.
    Supports RGBA (with per-pixel alpha) and BGR / grayscale (full alpha=1).
    Caches original decoded flag textures to avoid disk I/O bottlenecks.
    """
    _raw_cache: Dict[str, np.ndarray] = {}

    @staticmethod
    def load(path: str, width_px: int, height_px: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            bgr  : (H, W, 3) uint8
            alpha: (H, W)    float32 in [0, 1]
        """
        if path in FlagTextureLoader._raw_cache:
            img = FlagTextureLoader._raw_cache[path]
        else:
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Failed to load flag image: {path}")
            FlagTextureLoader._raw_cache[path] = img

        h, w = img.shape[:2]
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3].astype(np.float32) / 255.0
        elif img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = np.ones((h, w), dtype=np.float32)
        else:
            bgr = img[:, :, :3]
            alpha = np.ones((h, w), dtype=np.float32)

        bgr = cv2.resize(bgr, (width_px, height_px), interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, (width_px, height_px), interpolation=cv2.INTER_AREA)
        
        # Apply texture-level domain randomization for diversity
        bgr, alpha = FlagTextureLoader.apply_texture_augmentations(bgr, alpha)
        return bgr, alpha

    @staticmethod
    def apply_texture_augmentations(bgr: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Color jitter (HSV shift)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-6, 6)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.65, 1.15), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.70, 1.15), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 2. Print quality blur (simulates slightly blurred vector graphics)
        if random.random() < 0.70:
            sigma = random.uniform(0.4, 1.1)
            bgr = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma)

        # 3. Fade/wear: blend slightly with a neutral sand color to simulate fading under desert sun
        if random.random() < 0.50:
            fade_factor = random.uniform(0.05, 0.20)
            # Sandy/dusty color in BGR (e.g. 170, 190, 200)
            sand_color = np.array([170, 190, 200], dtype=np.float32)
            bgr = (bgr.astype(np.float32) * (1 - fade_factor) + sand_color * fade_factor).astype(np.uint8)

        return bgr, alpha


# ============================================================================
# WIND DEFORMER
# ============================================================================

class WindDeformer:
    """
    Applies realistic wind / wrinkle deformation to the flag texture.
    The flag lies on the ground so full sinusoidal displacement +
    optional corner-lifting is used.
    """

    def __init__(self, cfg: dict):
        w = cfg.get('wind', {})
        self.probability = w.get('probability', 0.65)
        self.max_disp_pct = w.get('displacement_max_pct', 0.12)
        self.wave_min = w.get('wave_count_min', 2)
        self.wave_max = w.get('wave_count_max', 5)
        self.corner_lift_prob = w.get('corner_lift_probability', 0.25)

    def apply(
        self,
        texture: np.ndarray,
        src_alpha: Optional[np.ndarray] = None,
        effect_level: str = 'strong',
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W = texture.shape[:2]
        if src_alpha is None:
            src_alpha = np.ones((H, W), dtype=np.float32)

        scale_map = {
            'subtle': (0.10, 0.15),
            'medium': (0.40, 0.50),
            'strong': (1.00, 1.00),
        }
        disp_scale, prob_scale = scale_map.get(effect_level, (1.0, 1.0))

        if random.random() > self.probability * prob_scale:
            return texture, src_alpha

        max_disp = int(max(W, H) * self.max_disp_pct * disp_scale)
        if max_disp < 1:
            return texture, src_alpha

        num_waves = random.randint(self.wave_min, self.wave_max)
        dx = np.zeros((H, W), dtype=np.float32)
        dy = np.zeros((H, W), dtype=np.float32)
        
        # Pre-generate meshgrid for coordinates once to avoid np.mgrid overhead inside wave loop
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        xc, yc = np.meshgrid(xs, ys)

        for _ in range(num_waves):
            fx = random.uniform(1, 4) * math.pi / W
            fy = random.uniform(1, 4) * math.pi / H
            px = random.uniform(0, 2 * math.pi)
            py = random.uniform(0, 2 * math.pi)
            amp = random.uniform(0.3, 1.0) * max_disp / num_waves
            dx += (amp * np.sin(fx * xc + fy * yc + px)).astype(np.float32)
            dy += (amp * np.sin(fy * yc + fx * xc + py) * 0.5).astype(np.float32)

        # Corner lifting
        if random.random() < self.corner_lift_prob:
            corner = random.choice(['tl', 'tr', 'bl', 'br'])
            lift = np.zeros((H, W), dtype=np.float32)
            r = min(W, H) // 3
            pts = {'tl': (0, 0), 'tr': (W, 0), 'bl': (0, H), 'br': (W, H)}
            cv2.circle(lift, pts[corner], r, 1.0, -1)
            lift = cv2.GaussianBlur(lift, (0, 0), sigmaX=r * 0.3)
            corner_alpha = 1.0 - lift * 0.3
        else:
            corner_alpha = np.ones((H, W), dtype=np.float32)

        map_x = (np.arange(W, dtype=np.float32)[np.newaxis, :] + dx)
        map_y = (np.arange(H, dtype=np.float32)[:, np.newaxis] + dy)
        deformed = cv2.remap(texture, map_x, map_y,
                             cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        rem_alpha = cv2.remap(src_alpha, map_x, map_y,
                              cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, borderValue=0.0)

        grad_mag = np.sqrt(
            cv2.Sobel(dy, cv2.CV_32F, 1, 0, ksize=3) ** 2 +
            cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3) ** 2
        )
        if grad_mag.max() > 1e-6:
            grad_mag /= grad_mag.max()
        wrinkle_shadow = 1.0 - grad_mag * 0.2
        alpha = np.clip(wrinkle_shadow * corner_alpha * rem_alpha, 0, 1)
        return deformed, alpha


# ============================================================================
# SCENE COMPOSITOR
# ============================================================================

class SceneCompositor:
    """Alpha-blends the perspective-warped flag onto the background crop."""

    def __init__(self, output_size: Tuple[int, int], shadow_cfg: dict):
        self.output_size = output_size
        self.shadow_cfg = shadow_cfg

    def composite(self, background: np.ndarray, flag_data: dict) -> np.ndarray:
        # Keep legacy method for compatibility if needed
        scene = self._paste_flag(background.copy(), flag_data)
        if random.random() < self.shadow_cfg.get('probability', 0.12):
            scene = self._apply_drone_shadow(scene, flag_data['bbox'])
        return scene

    def composite_crop(self, bg_crop: np.ndarray, flag_data: dict, crop_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Composites and shadows directly on the target crop, achieving a ~100x speedup."""
        cx1, cy1, cx2, cy2 = crop_coords
        scene_crop = self._paste_flag_crop(bg_crop, flag_data, cx1, cy1)
        if random.random() < self.shadow_cfg.get('probability', 0.12):
            scene_crop = self._apply_drone_shadow_crop(scene_crop, flag_data['bbox'], cx1, cy1)
        return scene_crop

    def _paste_flag_crop(self, crop: np.ndarray, flag_data: dict, cx1: int, cy1: int) -> np.ndarray:
        texture = flag_data['texture']
        alpha = flag_data['alpha']
        
        # Shift projected corners by crop offset
        corners = flag_data['corners_px'].astype(np.float32) - np.array([cx1, cy1], dtype=np.float32)
        
        th, tw = texture.shape[:2]
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, corners)
        H_crop, W_crop = crop.shape[:2]
        
        warped = cv2.warpPerspective(
            texture, M, (W_crop, H_crop),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        warped_a = cv2.warpPerspective(
            alpha, M, (W_crop, H_crop),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )

        # 1. Local Ambient Light matching (Color Bleeding/Ambient Cast)
        # Sample average color of the background to apply as a color wash/light source
        bg_mean = np.mean(crop, axis=(0, 1))  # (3,) BGR mean
        blend_ambient = random.uniform(0.08, 0.22)
        ambient_layer = np.tile(bg_mean, (H_crop, W_crop, 1)).astype(np.float32)
        warped_matched = warped.astype(np.float32) * (1.0 - blend_ambient) + ambient_layer * blend_ambient
        warped = np.clip(warped_matched, 0, 255).astype(np.uint8)

        # 2. Ground Texture Bump Mapping (makes flag look draped over sand/gravel)
        # Extract high-frequency variations from background crop and project onto flag
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        crop_blur = cv2.blur(crop_gray, (7, 7))
        variation = crop_gray / (crop_blur + 1e-5)
        # Scale/limit range to prevent extreme contrast changes
        variation = np.clip(variation, 0.80, 1.20)
        variation3 = variation[:, :, np.newaxis]
        warped = np.clip(warped.astype(np.float32) * variation3, 0, 255).astype(np.uint8)

        # 3. Dust / Dirt Overlay (procedurally generated cloud noise)
        if random.random() < 0.60:
            noise = np.random.uniform(0.70, 1.0, (16, 16)).astype(np.float32)
            dust_mask = cv2.resize(noise, (W_crop, H_crop), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
            warped = np.clip(warped.astype(np.float32) * dust_mask, 0, 255).astype(np.uint8)

        # 4. Soft variable edge feathering
        blur_k = random.choice([3, 5, 7])
        warped_a = cv2.GaussianBlur(warped_a, (blur_k, blur_k), 0)
        
        a3 = warped_a[:, :, np.newaxis]
        blended = (crop.astype(np.float32) * (1 - a3) +
                   warped.astype(np.float32) * a3)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_drone_shadow_crop(
        self, crop: np.ndarray, bbox: Tuple[int, int, int, int], cx1: int, cy1: int
    ) -> np.ndarray:
        x, y, w, h = bbox
        H_crop, W_crop = crop.shape[:2]

        drone_span = int(max(w, h) * random.uniform(0.7, 1.2))
        canvas_size = max(4, int(drone_span * 1.5))
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cx, cy = canvas_size // 2, canvas_size // 2

        body_r    = max(2, int(drone_span * 0.10))
        arm_len   = max(1, int(drone_span * 0.40))
        arm_thick = max(1, int(drone_span * 0.04))
        prop_r    = max(3, int(drone_span * 0.15))

        cv2.circle(mask, (cx, cy), body_r, 255, -1)

        d = int(arm_len * 0.707)
        for sign_x, sign_y in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            tx, ty = cx + sign_x * d, cy + sign_y * d
            cv2.line(mask, (cx, cy), (tx, ty), 255, arm_thick)
            cv2.circle(mask, (tx, ty), prop_r, 255, -1)
            cv2.circle(mask, (tx, ty), int(prop_r * 0.55), 170, -1)

        angle = random.uniform(0, 360)
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        mask  = cv2.warpAffine(mask, M_rot, (canvas_size, canvas_size))

        blur_k = int(drone_span * random.uniform(0.15, 0.35))
        blur_k = max(3, blur_k + (blur_k % 2 == 0))
        mask_f = cv2.GaussianBlur(mask, (blur_k, blur_k), 0).astype(np.float32) / 255.0

        opacity = random.uniform(
            self.shadow_cfg.get('opacity_min', 0.35),
            self.shadow_cfg.get('opacity_max', 0.60),
        )
        mask_f *= opacity

        flag_cx = x + w // 2
        flag_cy = y + h // 2
        offset_angle = random.uniform(0.0, 2.0 * math.pi)
        shift_dist    = random.uniform(0.4, 0.9) * max(w, h)
        shadow_cx = int(flag_cx + shift_dist * math.cos(offset_angle))
        shadow_cy = int(flag_cy + shift_dist * math.sin(offset_angle))

        # Shift shadow center to crop coordinates
        shadow_cx_crop = shadow_cx - cx1
        shadow_cy_crop = shadow_cy - cy1

        full_mask = np.zeros((H_crop, W_crop), dtype=np.float32)
        x_min, y_min = shadow_cx_crop - canvas_size // 2, shadow_cy_crop - canvas_size // 2
        x_max, y_max = x_min + canvas_size, y_min + canvas_size

        src_x1 = max(0, -x_min);   src_y1 = max(0, -y_min)
        src_x2 = canvas_size - max(0, x_max - W_crop)
        src_y2 = canvas_size - max(0, y_max - H_crop)
        dst_x1 = max(0, x_min);    dst_y1 = max(0, y_min)
        dst_x2 = min(W_crop, x_max);    dst_y2 = min(H_crop, y_max)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = \
                mask_f[src_y1:src_y2, src_x1:src_x2]

        shadowed = crop.astype(np.float32) * (1.0 - full_mask[:, :, np.newaxis])
        return np.clip(shadowed, 0, 255).astype(np.uint8)

    def _paste_flag(self, scene: np.ndarray, flag_data: dict) -> np.ndarray:
        texture = flag_data['texture']
        alpha = flag_data['alpha']
        corners = flag_data['corners_px'].astype(np.float32)
        th, tw = texture.shape[:2]
        src = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, corners)
        H, W = scene.shape[:2]
        warped = cv2.warpPerspective(
            texture, M, (W, H),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        warped_a = cv2.warpPerspective(
            alpha, M, (W, H),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
        )
        warped_a = cv2.GaussianBlur(warped_a, (3, 3), 0)
        a3 = warped_a[:, :, np.newaxis]
        blended = (scene.astype(np.float32) * (1 - a3) +
                   warped.astype(np.float32) * a3)
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _apply_drone_shadow(
        self, scene: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, w, h = bbox
        H, W = scene.shape[:2]

        drone_span = int(max(w, h) * random.uniform(0.7, 1.2))
        canvas_size = max(4, int(drone_span * 1.5))
        mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        cx, cy = canvas_size // 2, canvas_size // 2

        body_r    = max(2, int(drone_span * 0.10))
        arm_len   = max(1, int(drone_span * 0.40))
        arm_thick = max(1, int(drone_span * 0.04))
        prop_r    = max(3, int(drone_span * 0.15))

        cv2.circle(mask, (cx, cy), body_r, 255, -1)

        d = int(arm_len * 0.707)
        for sign_x, sign_y in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            tx, ty = cx + sign_x * d, cy + sign_y * d
            cv2.line(mask, (cx, cy), (tx, ty), 255, arm_thick)
            cv2.circle(mask, (tx, ty), prop_r, 255, -1)
            cv2.circle(mask, (tx, ty), int(prop_r * 0.55), 170, -1)

        angle = random.uniform(0, 360)
        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        mask  = cv2.warpAffine(mask, M_rot, (canvas_size, canvas_size))

        blur_k = int(drone_span * random.uniform(0.15, 0.35))
        blur_k = max(3, blur_k + (blur_k % 2 == 0))
        mask_f = cv2.GaussianBlur(mask, (blur_k, blur_k), 0).astype(np.float32) / 255.0

        opacity = random.uniform(
            self.shadow_cfg.get('opacity_min', 0.35),
            self.shadow_cfg.get('opacity_max', 0.60),
        )
        mask_f *= opacity

        flag_cx = x + w // 2
        flag_cy = y + h // 2
        offset_angle = random.uniform(0.0, 2.0 * math.pi)
        shift_dist    = random.uniform(0.4, 0.9) * max(w, h)
        shadow_cx = int(flag_cx + shift_dist * math.cos(offset_angle))
        shadow_cy = int(flag_cy + shift_dist * math.sin(offset_angle))

        full_mask = np.zeros((H, W), dtype=np.float32)
        x_min, y_min = shadow_cx - canvas_size // 2, shadow_cy - canvas_size // 2
        x_max, y_max = x_min + canvas_size, y_min + canvas_size

        src_x1 = max(0, -x_min);   src_y1 = max(0, -y_min)
        src_x2 = canvas_size - max(0, x_max - W)
        src_y2 = canvas_size - max(0, y_max - H)
        dst_x1 = max(0, x_min);    dst_y1 = max(0, y_min)
        dst_x2 = min(W, x_max);    dst_y2 = min(H, y_max)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            full_mask[dst_y1:dst_y2, dst_x1:dst_x2] = \
                mask_f[src_y1:src_y2, src_x1:src_x2]

        shadowed = scene.astype(np.float32) * (1.0 - full_mask[:, :, np.newaxis])
        return np.clip(shadowed, 0, 255).astype(np.uint8)


# ============================================================================
# UAV AUGMENTATION PIPELINE
# ============================================================================

class UAVAugmentationPipeline:
    """
    Applies physically-motivated UAV camera degradations after compositing.
    Preserves original pipeline order and all effect magnitudes.

    Effect tiers:
        subtle : 30 % magnitude, 55 % probability multiplier
        medium : 65 % magnitude, 80 % probability multiplier
        strong : 100 % magnitude, 100 % probability multiplier
    """

    def __init__(self, cfg: dict):
        self.aug = cfg.get('augmentations', {})

    def apply(
        self, image: np.ndarray, effect_level: str = 'strong'
    ) -> Tuple[np.ndarray, dict]:
        scale_map = {
            'subtle': (0.30, 0.55),
            'medium': (0.65, 0.80),
            'strong': (1.00, 1.00),
        }
        self._mag, self._prob = scale_map.get(effect_level, (1.0, 1.0))
        applied: Dict[str, Any] = {'effect_level': effect_level}
        
        # 1. Defocus Blur (Defocus / Lens PSF)
        image, a = self._defocus_blur(image)
        applied.update(a)
        
        # 2. Chromatic Aberration
        image, a = self._chromatic_aberration(image)
        applied.update(a)
        
        # 3. Motion Blur
        image, a = self._motion_blur(image)
        applied.update(a)
        
        # 4. Vibration Blur
        image, a = self._vibration_blur(image)
        applied.update(a)
        
        # 5. Exposure Shift
        image, a = self._exposure(image)
        applied.update(a)
        
        # 6. White Balance Shift
        image, a = self._white_balance(image)
        applied.update(a)
        
        # 7. Gamma Variation
        image, a = self._gamma(image)
        applied.update(a)
        
        # 8. Vignetting
        image, a = self._vignetting(image)
        applied.update(a)
        
        # 9. Atmospheric Haze
        image, a = self._atmospheric_haze(image)
        applied.update(a)
        
        # 10. Sensor Noise (Gaussian + Shot noise)
        image, a = self._sensor_noise(image)
        applied.update(a)
        
        # 11. Video Compression Simulation (JPEG encode/decode loop)
        image, a = self._video_compression(image)
        applied.update(a)
        
        return image, applied

    def _defocus_blur(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('defocus_blur', {})
        if random.random() > cfg.get('probability', 0.55) * self._prob:
            return img, {}
        sigma = random.uniform(cfg.get('sigma_min', 0.8), cfg.get('sigma_max', 3.0)) * self._mag
        if sigma < 0.1:
            return img, {}
        ksize = int(round(sigma * 3.0))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(3, ksize)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma), {'defocus_blur': sigma}

    def _chromatic_aberration(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('chromatic_aberration', {})
        if random.random() > cfg.get('probability', 0.25) * self._prob:
            return img, {}
        h, w = img.shape[:2]
        shift_pct = random.uniform(cfg.get('shift_min', 0.002), cfg.get('shift_max', 0.008)) * self._mag
        max_shift = max(1, int(round(min(w, h) * shift_pct)))
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        if dx == 0 and dy == 0:
            return img, {}
        M_r = np.float32([[1, 0, dx], [0, 1, dy]])
        M_b = np.float32([[1, 0, -dx], [0, 1, -dy]])
        b = cv2.warpAffine(img[:, :, 0], M_b, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        g = img[:, :, 1]
        r = cv2.warpAffine(img[:, :, 2], M_r, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        return cv2.merge([b, g, r]), {'chromatic_aberration': (dx, dy)}

    def _motion_blur(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('motion_blur', {})
        if random.random() > cfg.get('probability', 0.55) * self._prob:
            return img, {}
        if random.random() < cfg.get('severe_probability', 0.08) * self._prob:
            ks = random.randint(
                cfg.get('severe_kernel_min', 11), cfg.get('severe_kernel_max', 23)
            )
        else:
            ks = random.randint(cfg.get('kernel_min', 3), cfg.get('kernel_max', 13))
        ks = max(3, round(ks * self._mag))
        if ks % 2 == 0:
            ks += 1
        angle = random.uniform(0, 360)
        k = np.zeros((ks, ks), dtype=np.float32)
        k[ks // 2, :] = 1.0 / ks
        M = cv2.getRotationMatrix2D((ks / 2, ks / 2), angle, 1.0)
        k = cv2.warpAffine(k, M, (ks, ks))
        k /= k.sum() + 1e-8
        return cv2.filter2D(img, -1, k), {'motion_blur': {'kernel': ks, 'angle': angle}}

    def _vibration_blur(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('vibration_blur', {})
        if random.random() > cfg.get('probability', 0.45) * self._prob:
            return img, {}
        sigma = (random.uniform(cfg.get('sigma_min', 0.5),
                                 cfg.get('sigma_max', 3.5)) * self._mag)
        if sigma < 0.1:
            return img, {}
        return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma), {'vibration_blur': sigma}

    def _exposure(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('exposure', {})
        if random.random() > cfg.get('probability', 0.60) * self._prob:
            return img, {}
        ev_range = cfg.get('ev_range', 1.0)
        ev = random.uniform(-ev_range, ev_range) * self._mag
        if random.random() < cfg.get('underexpose_bias', 0.30):
            ev -= random.uniform(0.3, 1.0) * self._mag
        factor = 2.0 ** ev
        img_exp = img.astype(np.float32) * factor
        return np.clip(img_exp, 0, 255).astype(np.uint8), {'exposure_ev': ev}

    def _white_balance(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('white_balance', {})
        if random.random() > cfg.get('probability', 0.50) * self._prob:
            return img, {}
        r_gain = random.uniform(cfg.get('r_min', 0.85), cfg.get('r_max', 1.15))
        g_gain = random.uniform(cfg.get('g_min', 0.95), cfg.get('g_max', 1.05))
        b_gain = random.uniform(cfg.get('b_min', 0.85), cfg.get('b_max', 1.15))
        if random.random() < cfg.get('warm_bias_prob', 0.40):
            r_gain *= random.uniform(1.02, 1.12)
            b_gain *= random.uniform(0.85, 0.95)
        r_gain = 1.0 + (r_gain - 1.0) * self._mag
        g_gain = 1.0 + (g_gain - 1.0) * self._mag
        b_gain = 1.0 + (b_gain - 1.0) * self._mag
        img_wb = img.astype(np.float32)
        img_wb[:, :, 0] *= b_gain
        img_wb[:, :, 1] *= g_gain
        img_wb[:, :, 2] *= r_gain
        return np.clip(img_wb, 0, 255).astype(np.uint8), {'wb_gains': (b_gain, g_gain, r_gain)}

    def _gamma(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('gamma', {})
        if random.random() > cfg.get('probability', 0.45) * self._prob:
            return img, {}
        gamma = random.uniform(cfg.get('gamma_min', 0.7), cfg.get('gamma_max', 1.4))
        gamma = 1.0 + (gamma - 1.0) * self._mag
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table), {'gamma': gamma}

    def _vignetting(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('vignetting', {})
        if random.random() > cfg.get('probability', 0.35) * self._prob:
            return img, {}
        h, w = img.shape[:2]
        strength = random.uniform(cfg.get('strength_min', 0.10), cfg.get('strength_max', 0.30)) * self._mag
        cx, cy = w / 2, h / 2
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2) + 1e-5
        mask = np.clip(1.0 - strength * (r / r_max), 0.0, 1.0)[:, :, np.newaxis]
        return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8), {'vignette_strength': strength}

    def _atmospheric_haze(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('atmospheric_haze', {})
        if random.random() > cfg.get('probability', 0.40) * self._prob:
            return img, {}
        intensity = (random.uniform(cfg.get('intensity_min', 0.05),
                                     cfg.get('intensity_max', 0.40)) * self._mag)
        haze_color = list(cfg.get('haze_color', [200, 190, 170]))
        c_var = cfg.get('color_variation', 25)
        haze_color[0] = int(np.clip(haze_color[0] + random.randint(-c_var, c_var), 0, 255))
        haze_color[1] = int(np.clip(haze_color[1] + random.randint(-c_var, c_var), 0, 255))
        haze_color[2] = int(np.clip(haze_color[2] + random.randint(-c_var, c_var), 0, 255))
        layer = np.full_like(img, haze_color, dtype=np.uint8)
        var = np.random.normal(0, 0.03, img.shape[:2]).astype(np.float32)
        a = np.clip(intensity + var, 0, 0.7)[:, :, np.newaxis]
        result = img.astype(np.float32) * (1.0 - a) + layer.astype(np.float32) * a
        if intensity > 0.15:
            result = cv2.GaussianBlur(result.astype(np.uint8), (0, 0), sigmaX=intensity * 2.5)
        return np.clip(result, 0, 255).astype(np.uint8), {'atmospheric_haze': intensity, 'color': haze_color}

    def _sensor_noise(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('sensor_noise', {})
        if random.random() > cfg.get('probability', 0.70) * self._prob:
            return img, {}
        sigma_g = random.uniform(cfg.get('sigma_min', 3), cfg.get('sigma_max', 18)) * self._mag
        img_f = img.astype(np.float32)
        if random.random() < cfg.get('shot_noise_prob', 0.40):
            shot_scale = random.uniform(0.1, 0.4) * self._mag
            shot_noise = np.random.normal(0, 1.0, img.shape).astype(np.float32)
            shot_noise *= np.sqrt(np.clip(img_f, 0.0, None)) * shot_scale
            img_f += shot_noise
        read_noise = np.random.normal(0, sigma_g, img.shape).astype(np.float32)
        img_f += read_noise
        return np.clip(img_f, 0, 255).astype(np.uint8), {'sensor_noise': sigma_g}

    def _video_compression(self, img: np.ndarray) -> Tuple[np.ndarray, dict]:
        cfg = self.aug.get('video_compression', {})
        if random.random() > cfg.get('probability', 0.55) * self._prob:
            return img, {}
        q = random.randint(cfg.get('quality_min', 30), cfg.get('quality_max', 70))
        q = int(np.clip(100 - (100 - q) * self._mag, 10, 95))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img, {'video_compression_q': q}


# ============================================================================
# DETECTOR-CROP EXTRACTOR
# ============================================================================

class DetectorCropExtractor:
    """
    Crops the composite scene around the true flag location, simulating
    an imperfect upstream detector output:

        * Small random off-centre offsets  (realistic detector inaccuracy)
        * Randomised margin around the flag (loose detector boxes)
        * Hard guarantee: flag is NEVER truncated in the crop
        * Minimum pixel size enforced
    """

    def __init__(self, cfg: dict):
        c = cfg.get('crop', {})
        self.margin_min = c.get('margin_ratio_min', 0.20)
        self.margin_max = c.get('margin_ratio_max', 0.60)
        self.offset_max = c.get('offset_max_ratio', 0.15)
        self.min_px = c.get('min_px_size', 32)

    def flag_rect(
        self, corners_px: np.ndarray, image_size: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Axis-aligned bounding box of the projected flag corners."""
        W, H = image_size
        x1 = max(0, int(np.floor(corners_px[:, 0].min())))
        y1 = max(0, int(np.floor(corners_px[:, 1].min())))
        x2 = min(W, int(np.ceil(corners_px[:, 0].max())))
        y2 = min(H, int(np.ceil(corners_px[:, 1].max())))
        if (x2 - x1) < self.min_px or (y2 - y1) < self.min_px:
            return None
        return x1, y1, x2, y2

    def get_crop_coords(
        self,
        flag_rect: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
    ) -> Optional[Tuple[Tuple[int, int, int, int], dict]]:
        W, H = image_size
        x1, y1, x2, y2 = flag_rect
        fw, fh = x2 - x1, y2 - y1

        mr = random.uniform(self.margin_min, self.margin_max)
        mx = int(fw * mr)
        my = int(fh * mr)

        ox = int(random.uniform(-self.offset_max, self.offset_max) * fw)
        oy = int(random.uniform(-self.offset_max, self.offset_max) * fh)

        cx1 = x1 - mx + ox
        cy1 = y1 - my + oy
        cx2 = x2 + mx + ox
        cy2 = y2 + my + oy

        cx1 = min(cx1, x1)
        cy1 = min(cy1, y1)
        cx2 = max(cx2, x2)
        cy2 = max(cy2, y2)

        cx1 = max(0, cx1)
        cy1 = max(0, cy1)
        cx2 = min(W, cx2)
        cy2 = min(H, cy2)

        if (cx2 - cx1) < self.min_px or (cy2 - cy1) < self.min_px:
            return None

        crop_info = {
            'crop_x1': cx1, 'crop_y1': cy1, 'crop_x2': cx2, 'crop_y2': cy2,
            'margin_ratio': mr, 'offset': (ox, oy),
        }
        return (cx1, cy1, cx2, cy2), crop_info

    def extract(
        self,
        image: np.ndarray,
        flag_rect: Tuple[int, int, int, int],
    ) -> Optional[Tuple[np.ndarray, dict]]:
        H, W = image.shape[:2]
        res = self.get_crop_coords(flag_rect, (W, H))
        if res is None:
            return None
        coords, crop_info = res
        cx1, cy1, cx2, cy2 = coords
        crop = image[cy1:cy2, cx1:cx2].copy()
        return crop, crop_info


# ============================================================================
# ALTITUDE SAMPLER
# ============================================================================

def sample_altitude(flight_cfg: dict) -> float:
    """
    Sample UAV altitude.
    Primary range 60-65 m gets 70 % weight for realistic training distribution.
    Hard floor: 60 m (never below).
    """
    alt_floor = 60.0
    alt_min = max(alt_floor, float(flight_cfg.get('altitude_min_m', 60.0)))
    alt_max = float(flight_cfg.get('altitude_max_m', 80.0))
    pri_min = max(alt_floor, float(flight_cfg.get('altitude_primary_min_m', 60.0)))
    pri_max = float(flight_cfg.get('altitude_primary_max_m', 65.0))
    weight = float(flight_cfg.get('altitude_primary_weight', 0.70))

    if random.random() < weight:
        alt = random.uniform(pri_min, pri_max)
    else:
        alt = random.uniform(alt_min, alt_max)

    return max(alt_floor, alt)   # safety clamp


# ============================================================================
# DATASET OUTPUT MANAGER
# ============================================================================

class DatasetManager:
    """
    Manages the output directory tree:

        dataset/
            train/<country>/
            val/<country>/
            test/<country>/
        labels.json
    """

    SPLITS = ('train', 'val', 'test')

    def __init__(self, base_dir: str, countries: List[str],
                 image_format: str = 'jpg'):
        self.base_dir = base_dir
        self.countries = countries
        self.fmt = image_format
        self.counters: Dict[str, Dict[str, int]] = {
            s: {c: 0 for c in countries} for s in self.SPLITS
        }

    def setup(self):
        for split in self.SPLITS:
            for country in self.countries:
                os.makedirs(
                    os.path.join(self.base_dir, split, country), exist_ok=True
                )
        logger.info(f"Output directories ready in '{self.base_dir}'")

    def save_image(
        self,
        image: np.ndarray,
        split: str,
        country: str,
        jpeg_quality: int = 95,
    ) -> str:
        idx = self.counters[split][country]
        self.counters[split][country] += 1
        fname = f"img_{idx:06d}.{self.fmt}"
        path = os.path.join(self.base_dir, split, country, fname)
        # Use imencode + open('wb') so Unicode folder names (e.g. São Tomé)
        # are handled correctly on Windows (cv2.imwrite silently fails there).
        ext = "." + self.fmt
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if self.fmt in ('jpg', 'jpeg') else []
        ok, buf = cv2.imencode(ext, image, params)
        if not ok:
            raise IOError(f"cv2.imencode failed for {path}")
        with open(path, 'wb') as f:
            f.write(buf.tobytes())
        return path

    def summary(self) -> Dict[str, int]:
        return {s: sum(self.counters[s].values()) for s in self.SPLITS}


# ============================================================================
# MAIN GENERATOR
# ============================================================================

class ClassificationDatasetGenerator:
    """
    Orchestrates the full synthetic dataset generation pipeline.

    Per-image pipeline:
      1. Choose split (train/val/test) by configured ratios
      2. Choose random country from discovered folders
      3. Pick random flag image from that country's folder
      4. Sample altitude (60-80 m, primary 60-65 m, hard floor 60 m)
      5. Load random background (crop + colour jitter)
      6. Compute flag pixel size at altitude via physical GSD model
      7. Load and resize flag texture (4x oversample)
      8. Apply wind deformation
      9. Compute perspective projection (nadir, ±roll)
     10. Composite flag onto background
     11. Apply UAV augmentation pipeline
         (motion blur, vibration blur, atmospheric haze, sensor noise)
     12. Extract detector-crop (random margin, off-centre, no truncation)
     13. Apply optional JPEG compression degradation
     14. Save to dataset/<split>/<country>/img_XXXXXX.jpg
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        out_size: Tuple[int, int] = tuple(cfg['output']['image_size'])  # (W, H)
        self.output_size = out_size

        flags_dir = cfg['flag_texture']['flags_dir']
        bg_dir = cfg['background']['real_backgrounds_dir']

        self.camera = VirtualCamera(output_size=out_size)
        self.registry = CountryRegistry(flags_dir)
        self.bg_loader = BackgroundLoader(bg_dir, out_size)
        self.wind = WindDeformer(cfg)
        self.compositor = SceneCompositor(
            out_size,
            cfg.get('augmentations', {}).get('drone_shadow', {}),
        )
        self.augmentor = UAVAugmentationPipeline(cfg)
        self.crop_extractor = DetectorCropExtractor(cfg)

        self.flag_cfg = cfg['flag']
        self.flight_cfg = cfg['flight']
        self.persp_cfg = cfg['perspective']
        self.crop_output_size = cfg.get('crop', {}).get('output_size', 224)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, samples_per_country: int):
        """
        Generate a *perfectly balanced* dataset: every country gets exactly
        `samples_per_country` images, split as train/val/test per-country.
        Failed renders are silently retried so the final counts are exact.
        """
        base_dir = self.cfg['output']['base_dir']
        mgr = DatasetManager(
            base_dir,
            self.registry.countries,
            self.cfg['output'].get('image_format', 'jpg'),
        )
        mgr.setup()
        self.registry.save_labels(os.path.join(base_dir, 'labels.json'))

        split_cfg   = self.cfg.get('split', {})
        train_r     = float(split_cfg.get('train', 0.70))
        val_r       = float(split_cfg.get('val',   0.15))
        train_n     = int(samples_per_country * train_r)
        val_n       = int(samples_per_country * val_r)
        test_n      = samples_per_country - train_n - val_n   # absorbs rounding

        total_target = samples_per_country * len(self.registry.countries)
        saved  = 0
        skipped: Dict[str, int] = {}

        logger.info(
            f"Generating balanced dataset: {samples_per_country} samples × "
            f"{len(self.registry.countries)} countries = {total_target} total "
            f"-> {base_dir}"
        )
        logger.info(
            f"Per-country split: train={train_n}  val={val_n}  test={test_n}"
        )

        pbar = tqdm(total=total_target, desc='Generating')

        for country in self.registry.countries:
            for split, target in [
                ('train', train_n), ('val', val_n), ('test', test_n)
            ]:
                count = 0
                while count < target:
                    flag_path = self.registry.random_image(country)
                    try:
                        result, reason = self._generate_single(flag_path)
                        if result is None:
                            skipped[reason] = skipped.get(reason, 0) + 1
                            continue

                        crop, meta = result

                        # JPEG compression degradation
                        jpeg_quality = self.cfg['output'].get('jpeg_quality', 95)
                        aug_comp  = self.cfg.get('augmentations', {}).get('compression', {})
                        comp_prob = aug_comp.get('probability', 0.50)
                        eff = meta.get('effect_level', 'strong')
                        if eff == 'subtle':
                            comp_prob *= 0.20
                        elif eff == 'medium':
                            comp_prob *= 0.60
                        if random.random() < comp_prob:
                            if eff == 'subtle':
                                jpeg_quality = random.randint(92, 95)
                            elif eff == 'medium':
                                jpeg_quality = random.randint(85, 92)
                            else:
                                jpeg_quality = random.randint(
                                    aug_comp.get('quality_min', 75),
                                    aug_comp.get('quality_max', 90),
                                )

                        mgr.save_image(crop, split, country, jpeg_quality)
                        count += 1
                        saved += 1
                        pbar.update(1)

                    except Exception as exc:
                        logger.error(
                            f"Error [{country}][{split}]: {exc}"
                        )
                        skipped['exception'] = skipped.get('exception', 0) + 1

        pbar.close()
        self._print_summary(total_target, saved, skipped, mgr)

    # ------------------------------------------------------------------
    # Single image generation
    # ------------------------------------------------------------------

    def _generate_single(
        self, flag_path: str
    ) -> Tuple[Optional[Tuple[np.ndarray, dict]], Optional[str]]:
        W, H = self.output_size

        effect_level = random.choices(['subtle', 'medium', 'strong'], weights=[0.15, 0.35, 0.50])[0]

        # Altitude — hard floor 60 m
        altitude = sample_altitude(self.flight_cfg)

        # Nadir camera attitude (pitch = 0)
        pitch = 0.0   # strict nadir — camera mounted vertically
        roll_std = self.persp_cfg.get('roll_std_deg', 1.5)
        roll = float(np.clip(
            np.random.normal(0.0, roll_std),
            self.persp_cfg.get('camera_roll_min_deg', -3.0),
            self.persp_cfg.get('camera_roll_max_deg',  3.0),
        ))
        yaw = random.uniform(0.0, 360.0)

        gsd = self.camera.compute_gsd(altitude)

        # Physical flag size with small aspect-ratio jitter
        jitter = float(self.flag_cfg.get('aspect_ratio_jitter', 0.05))
        flag_w_m = float(self.flag_cfg['width_m']) * random.uniform(
            1.0 - jitter, 1.0 + jitter
        )
        flag_h_m = float(self.flag_cfg['height_m']) * random.uniform(
            1.0 - jitter, 1.0 + jitter
        )

        # Flag pixel size (use 2x oversample for quality to allow realistic sub-pixel aliasing)
        fw_px, fh_px = self.camera.flag_pixel_size(flag_w_m, flag_h_m, altitude)
        fw_px = max(int(fw_px), 8)
        fh_px = max(int(fh_px), 8)

        # Load flag texture (2x oversampled for subpixel accuracy while maintaining realistic softness)
        texture, src_alpha = FlagTextureLoader.load(flag_path, fw_px * 2, fh_px * 2)

        # Wind deformation
        texture, alpha = self.wind.apply(texture, src_alpha, effect_level)

        # Ground position (within footprint with edge margin)
        fp_w, fp_h = self.camera.compute_ground_footprint(altitude)
        margin = 0.12
        gx = random.uniform(-fp_w * (0.5 - margin), fp_w * (0.5 - margin))
        gy = random.uniform(-fp_h * (0.5 - margin), fp_h * (0.5 - margin))

        # Flag orientation on the ground plane
        flag_yaw = random.uniform(0.0, 360.0)
        fy_rad = math.radians(flag_yaw)
        hw, hh = flag_w_m / 2.0, flag_h_m / 2.0
        local = np.array(
            [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64
        )
        cos_a, sin_a = math.cos(fy_rad), math.sin(fy_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = (rot @ local.T).T
        ground3d = np.column_stack(
            [rotated[:, 0] + gx, rotated[:, 1] + gy, np.zeros(4)]
        )

        # Project to image pixels
        corners_px = self.camera.project_ground_plane(
            ground3d, altitude, pitch, roll, yaw
        )

        # Reject if flag is fully outside frame
        if (
            corners_px[:, 0].max() < 0 or corners_px[:, 0].min() > W or
            corners_px[:, 1].max() < 0 or corners_px[:, 1].min() > H
        ):
            return None, 'flag_out_of_frame'

        # Reject if flag is partially clipped (any corner outside image bounds).
        # Clipped flags are cut-off / unclear and must not be saved as training data.
        # The balanced retry loop will simply attempt a fresh placement instead.
        if (
            corners_px[:, 0].min() < 0 or corners_px[:, 0].max() > W or
            corners_px[:, 1].min() < 0 or corners_px[:, 1].max() > H
        ):
            return None, 'flag_clipped'

        # Compute axis-aligned bounding rect for cropping
        flag_rect = self.crop_extractor.flag_rect(corners_px, (W, H))
        if flag_rect is None:
            return None, 'flag_rect_too_small'

        # Retrieve crop coordinates and metadata details before rendering/compositing
        crop_coords_res = self.crop_extractor.get_crop_coords(flag_rect, (W, H))
        if crop_coords_res is None:
            return None, 'crop_too_small'
        crop_coords, crop_info = crop_coords_res
        cx1, cy1, cx2, cy2 = crop_coords

        # Load background ONLY for the crop area, applying color jitter on the crop
        background_crop = self.bg_loader.load_crop(cx1, cy1, cx2, cy2)

        x1, y1, x2, y2 = flag_rect
        bbox = (x1, y1, x2 - x1, y2 - y1)

        # Composite directly onto the cropped scene
        flag_data = {
            'texture': texture,
            'alpha': alpha,
            'corners_px': corners_px,
            'bbox': bbox,
        }
        crop = self.compositor.composite_crop(background_crop, flag_data, crop_coords)

        # Apply UAV augmentations (blur, haze, noise) directly on the crop
        crop, applied = self.augmentor.apply(crop, effect_level)

        # True resolution degradation: resize the crop from its native physical dimensions
        # to the target classifier resolution (e.g. 224x224), causing irreversible GSD detail loss.
        crop = cv2.resize(crop, (self.crop_output_size, self.crop_output_size), interpolation=cv2.INTER_LINEAR)

        meta = {
            'effect_level': effect_level,
            'altitude_m': altitude,
            'gsd_cm_px': round(gsd * 100, 4),
            'camera': {'pitch': pitch, 'roll': roll, 'yaw': yaw},
            'flag': {'w_m': flag_w_m, 'h_m': flag_h_m, 'yaw': flag_yaw},
            'augmentations': applied,
            **crop_info,
        }
        return (crop, meta), None

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    @staticmethod
    def _print_summary(
        total: int, saved: int, skipped: Dict[str, int], mgr: DatasetManager
    ):
        totals = mgr.summary()
        logger.info("=" * 62)
        logger.info("DATASET GENERATION COMPLETE")
        logger.info("=" * 62)
        logger.info(f"  Requested : {total}")
        logger.info(f"  Saved     : {saved}")
        logger.info(f"  Skipped   : {sum(skipped.values())}")
        for reason, cnt in sorted(skipped.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {reason:<30s}: {cnt}")
        logger.info(f"  train     : {totals['train']}")
        logger.info(f"  val       : {totals['val']}")
        logger.info(f"  test      : {totals['test']}")
        logger.info("=" * 62)
        if saved == 0:
            logger.warning("NO IMAGES WERE SAVED.")
            logger.warning("Common causes:")
            logger.warning("  flag_out_of_frame  -> check flag.width_m vs altitude & FOV")
            logger.warning("  flag_rect_too_small -> lower crop.min_px_size or increase flag size")
            logger.warning("  crop_too_small     -> lower crop.min_px_size")
            logger.warning("  exception          -> see ERROR lines above")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Synthetic UAV Flag Dataset Generator — 188 Countries (ConvNeXt V2 + ArcFace)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Camera: EZVIZ H6c Pro 2K + 12 mm M12 lens
  Sensor  : 6.954 x 3.562 mm  (back-calc from datasheet FOV)
  H-FOV   : 32.32 deg   V-FOV: 16.88 deg
  Altitude: 60–80 m (primary 60–65 m), NEVER below 60 m

Examples:
  python generate_target_img.py --config config.yaml
  python generate_target_img.py --config config.yaml --samples-per-country 500
  python generate_target_img.py --config config.yaml --samples-per-country 1000 --output ./my_dataset
        """,
    )
    parser.add_argument('--config', '-c', type=str, default='config.yaml',
                        help='YAML configuration file (default: config.yaml)')
    parser.add_argument('--samples-per-country', '-n', type=int, default=None,
                        help='Samples per country (overrides config samples_per_country)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output base directory (overrides config)')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.samples_per_country is not None:
        cfg['output']['samples_per_country'] = args.samples_per_country
    if args.output is not None:
        cfg['output']['base_dir'] = args.output

    seed = args.seed if args.seed is not None else random.randint(0, 2 ** 31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed: {seed}")

    # Print camera model summary
    cam = VirtualCamera(tuple(cfg['output']['image_size']))
    logger.info("=" * 62)
    logger.info("H6c Pro 2K + 12 mm lens — NADIR CONFIGURATION")
    logger.info(f"  Sensor    : {cam.SENSOR_W_MM:.3f} x {cam.SENSOR_H_MM:.3f} mm  "
                f"(1/2.7\" CMOS)")
    logger.info(f"  Focal     : {cam.FOCAL_MM} mm (M12)")
    logger.info(f"  H-FOV     : {cam.H_FOV_DEG:.2f} deg")
    logger.info(f"  V-FOV     : {cam.V_FOV_DEG:.2f} deg")
    logger.info(f"  Resolution: {cam.NATIVE_W} x {cam.NATIVE_H} px")
    for alt in [60, 65, 70, 80]:
        gsd = cam.compute_gsd(alt)
        fw, fh = cam.flag_pixel_size(
            cfg['flag']['width_m'], cfg['flag']['height_m'], alt
        )
        logger.info(
            f"  Alt {alt:>2d} m  : GSD={gsd*100:.2f} cm/px | "
            f"Flag={fw:.0f}x{fh:.0f} px"
        )
    logger.info("=" * 62)

    samples = cfg['output'].get('samples_per_country', 1000)
    gen = ClassificationDatasetGenerator(cfg)
    gen.generate(samples)


if __name__ == '__main__':
    main()
