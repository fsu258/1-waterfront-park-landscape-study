#!/usr/bin/env python3

from __future__ import annotations

import argparse
import io
import logging
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import pipeline
from mmseg.apis import inference_model, init_model

warnings.filterwarnings("ignore")


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
ADE20K_CLASSES = [
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
]
CATEGORY_MAPPING = {
    "trees": ["tree", "palm"],
    "shrubs": ["plant", "flower"],
    "grass": ["grass", "field"],
    "bare_land": ["earth", "sand", "dirt track", "land"],
    "terrain": ["mountain", "hill", "rock"],
    "people": ["person"],
    "vehicles": ["car", "bus", "truck", "van", "minibike", "bicycle", "airplane", "boat", "ship"],
    "buildings": ["building", "house", "skyscraper", "booth", "door", "windowpane", "ceiling"],
    "structures": [
        "sculpture",
        "bridge",
        "pier",
        "tower",
        "stage",
        "canopy",
        "wall",
        "fence",
        "railing",
        "column",
        "stairs",
        "bench",
        "chair",
        "table",
        "streetlight",
        "lamp",
        "flag",
    ],
    "trash_bins": ["ashcan"],
    "signboards": ["signboard"],
    "roads_paving": ["road", "sidewalk", "path", "floor", "runway"],
    "sky": ["sky"],
    "water": ["water", "sea", "river", "lake", "swimming pool", "fountain", "waterfall"],
}
CUSTOM_CATEGORIES = list(CATEGORY_MAPPING.keys()) + ["other"]
CATEGORY_COLORS = {
    "trees": (34, 139, 34),
    "shrubs": (107, 142, 35),
    "grass": (124, 252, 0),
    "bare_land": (160, 82, 45),
    "terrain": (139, 137, 137),
    "people": (255, 105, 180),
    "vehicles": (255, 165, 0),
    "buildings": (128, 128, 128),
    "structures": (148, 0, 211),
    "trash_bins": (205, 92, 92),
    "signboards": (70, 130, 180),
    "roads_paving": (105, 105, 105),
    "sky": (135, 206, 235),
    "water": (0, 119, 190),
    "other": (220, 220, 220),
}
COLOR_FEATURE_KEYS = ("brightness", "contrast", "saturation", "color_count", "color_entropy")


def natural_sort_key(name: str) -> List[object]:
    parts = re.split(r"(\d+)", Path(name).stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return key


def list_image_files(inputs: Sequence[Path]) -> List[Path]:
    files = []
    for item in inputs:
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(item)
        elif item.is_dir():
            for extension in IMAGE_EXTENSIONS:
                files.extend(sorted(item.rglob("*" + extension)))
    unique_files = sorted({path.resolve() for path in files}, key=lambda path: natural_sort_key(path.name))
    return unique_files


def pack_rgb_values(pixels: np.ndarray) -> np.ndarray:
    pixel_array = np.asarray(pixels, dtype=np.uint32)
    if pixel_array.ndim != 2 or pixel_array.shape[1] != 3:
        pixel_array = pixel_array.reshape(-1, 3)
    return (pixel_array[:, 0] << 16) | (pixel_array[:, 1] << 8) | pixel_array[:, 2]


def compute_color_features_for_mask(rgb_image: np.ndarray, hsv_image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    boolean_mask = np.asarray(mask, dtype=bool)
    if not np.any(boolean_mask):
        return {
            "brightness": float("nan"),
            "contrast": float("nan"),
            "saturation": float("nan"),
            "color_count": 0.0,
            "color_entropy": float("nan"),
        }
    pixels_rgb = rgb_image[boolean_mask].astype(np.float64)
    brightness_values = pixels_rgb.mean(axis=1)
    brightness = float(brightness_values.mean())
    contrast = float(np.sqrt(np.mean((brightness_values - brightness) ** 2)))
    saturation_values = hsv_image[..., 1][boolean_mask].astype(np.float64)
    saturation = float(saturation_values.mean())
    packed = pack_rgb_values(rgb_image[boolean_mask])
    unique_values, counts = np.unique(packed, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    color_entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "color_count": float(unique_values.size),
        "color_entropy": color_entropy,
    }


def prefix_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    results = {}
    for key, value in metrics.items():
        metric_key = "%s_%s" % (prefix, key)
        try:
            results[metric_key] = float(value)
        except (TypeError, ValueError):
            results[metric_key] = float("nan")
    return results


def compute_layer_color_metrics(
    rgb_image: np.ndarray,
    hsv_image: np.ndarray,
    layers: Sequence[Tuple[str, np.ndarray]],
) -> Dict[str, float]:
    results = {}
    for prefix, mask in layers:
        results.update(prefix_metrics(prefix, compute_color_features_for_mask(rgb_image, hsv_image, mask)))
    return results


class LandscapeSpatialAnalyzer:
    def __init__(
        self,
        output_dir: Path,
        depth_model_path: str,
        seg_config: str,
        seg_checkpoint: str,
        target_min_depth: float = 0.5,
        foreground_max_depth: float = 10.0,
        midground_max_depth: float = 30.0,
        sky_depth: float = 1000.0,
        device: str = "auto",
    ) -> None:
        self.output_dir = output_dir
        self.target_min_depth = float(target_min_depth)
        self.foreground_max_depth = float(foreground_max_depth)
        self.midground_max_depth = float(midground_max_depth)
        self.sky_depth = float(sky_depth)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA was requested but is unavailable. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = device
        self.depth_dir = self.output_dir / "corrected_depth_maps"
        self.segmentation_dir = self.output_dir / "segmentation_masks"
        self.visualization_dir = self.output_dir / "visualizations"
        for directory in (self.output_dir, self.depth_dir, self.segmentation_dir, self.visualization_dir):
            directory.mkdir(parents=True, exist_ok=True)
        logging.info("Using device: %s", self.device.upper())
        self.depth_estimator = pipeline(
            "depth-estimation",
            model=depth_model_path,
            device=0 if self.device == "cuda" else -1,
        )
        self.segmentation_model = init_model(seg_config, seg_checkpoint, device=self.device)
        self.custom_name_to_index = {name: index for index, name in enumerate(CUSTOM_CATEGORIES)}
        self.ade20k_lookup_table = self.create_ade20k_lookup_table()
        self.results = []

    def create_ade20k_lookup_table(self) -> np.ndarray:
        lookup = np.full(150, self.custom_name_to_index["other"], dtype=np.uint8)
        for custom_category, ade_classes in CATEGORY_MAPPING.items():
            category_index = self.custom_name_to_index[custom_category]
            for ade_class in ade_classes:
                if ade_class in ADE20K_CLASSES:
                    lookup[ADE20K_CLASSES.index(ade_class)] = category_index
        return lookup

    def save_image_with_unicode_path(self, path: Path, image: np.ndarray) -> bool:
        success, buffer = cv2.imencode(path.suffix or ".jpg", image)
        if success:
            with open(path, "wb") as file_handle:
                file_handle.write(buffer)
            return True
        return False

    def save_matplotlib_figure(self, figure: plt.Figure, path: Path) -> None:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="jpg", bbox_inches="tight", pad_inches=0.15)
        plt.close(figure)
        buffer.seek(0)
        image = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), 1)
        self.save_image_with_unicode_path(path, image)

    def save_rgb_panel(self, rgb_image: np.ndarray, title: str, path: Path, figsize: Tuple[int, int] = (6, 5)) -> None:
        figure, axis = plt.subplots(figsize=figsize, dpi=150)
        axis.imshow(rgb_image)
        axis.axis("off")
        if title:
            axis.set_title(title, fontsize=16)
        self.save_matplotlib_figure(figure, path)

    def process_single_image(self, image_path: Path) -> Optional[Dict[str, float]]:
        logging.info("Processing %s", image_path.name)
        try:
            image_bgr = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise IOError("Failed to decode image.")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            edges_binary = cv2.Canny(image_gray, 100, 200) > 0
            pil_image = Image.fromarray(image_rgb)
        except Exception as exc:
            logging.error("Could not read %s: %s", image_path.name, exc)
            return None
        depth_output = self.depth_estimator(pil_image)
        raw_depth = np.asarray(depth_output["predicted_depth"], dtype=float)
        segmentation_result = inference_model(self.segmentation_model, image_bgr)
        custom_segmentation = self.ade20k_lookup_table[segmentation_result.pred_sem_seg.data[0].cpu().numpy()]
        category_masks = {
            name: (custom_segmentation == index)
            for name, index in self.custom_name_to_index.items()
        }
        empty_mask = np.zeros_like(custom_segmentation, dtype=bool)
        sky_mask = category_masks.get("sky", empty_mask)
        current_min_depth = np.min(raw_depth[~sky_mask]) if np.any(~sky_mask) else np.min(raw_depth)
        offset = current_min_depth - self.target_min_depth
        rescaled_depth = np.maximum(0.0, raw_depth - offset)
        corrected_depth = rescaled_depth.copy()
        corrected_depth[sky_mask] = self.sky_depth
        if edges_binary.shape != corrected_depth.shape:
            edges_resized = cv2.resize(
                edges_binary.astype(np.uint8),
                (corrected_depth.shape[1], corrected_depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        else:
            edges_resized = edges_binary
        total_pixels = float(custom_segmentation.size)
        semantic_proportions = {
            name: float(np.sum(custom_segmentation == index) / total_pixels * 100.0)
            for index, name in enumerate(CUSTOM_CATEGORIES)
        }
        depth_for_statistics = corrected_depth[~sky_mask]
        depth_stats = {
            "avg_depth_m": float(np.mean(depth_for_statistics)) if depth_for_statistics.size > 0 else 0.0,
            "median_depth_m": float(np.median(depth_for_statistics)) if depth_for_statistics.size > 0 else 0.0,
            "std_depth_m": float(np.std(depth_for_statistics)) if depth_for_statistics.size > 0 else 0.0,
            "min_depth_m": float(np.min(depth_for_statistics)) if depth_for_statistics.size > 0 else 0.0,
            "max_depth_m": float(np.max(depth_for_statistics)) if depth_for_statistics.size > 0 else 0.0,
        }
        foreground_mask = corrected_depth < self.foreground_max_depth
        midground_mask = (corrected_depth >= self.foreground_max_depth) & (corrected_depth < self.midground_max_depth)
        background_mask = corrected_depth >= self.midground_max_depth
        trees_mask = category_masks.get("trees", empty_mask)
        shrubs_mask = category_masks.get("shrubs", empty_mask)
        grass_mask = category_masks.get("grass", empty_mask)
        plants_mask = trees_mask | shrubs_mask | grass_mask
        water_mask = category_masks.get("water", empty_mask)
        roads_mask = category_masks.get("roads_paving", empty_mask)
        people_mask = category_masks.get("people", empty_mask)
        vehicles_mask = category_masks.get("vehicles", empty_mask)
        structures_mask = category_masks.get("structures", empty_mask)
        trash_mask = category_masks.get("trash_bins", empty_mask)
        signboard_mask = category_masks.get("signboards", empty_mask)
        terrain_mask = category_masks.get("terrain", empty_mask)
        bare_land_mask = category_masks.get("bare_land", empty_mask)
        buildings_mask = category_masks.get("buildings", empty_mask)

        def compute_layer_counts(layer_mask: np.ndarray) -> Dict[str, int]:
            counts = {
                "trees": int(np.sum(layer_mask & trees_mask)),
                "shrubs": int(np.sum(layer_mask & shrubs_mask)),
                "grass": int(np.sum(layer_mask & grass_mask)),
                "water": int(np.sum(layer_mask & water_mask)),
                "roads": int(np.sum(layer_mask & roads_mask)),
                "people": int(np.sum(layer_mask & people_mask)),
                "vehicles": int(np.sum(layer_mask & vehicles_mask)),
                "structures": int(np.sum(layer_mask & structures_mask)),
                "trash_bins": int(np.sum(layer_mask & trash_mask)),
                "signboards": int(np.sum(layer_mask & signboard_mask)),
                "terrain": int(np.sum(layer_mask & terrain_mask)),
                "bare_land": int(np.sum(layer_mask & bare_land_mask)),
                "buildings": int(np.sum(layer_mask & buildings_mask)),
                "sky": int(np.sum(layer_mask & sky_mask)),
                "total": int(np.sum(layer_mask)),
            }
            counts["plants"] = counts["trees"] + counts["shrubs"] + counts["grass"]
            return counts

        foreground_counts = compute_layer_counts(foreground_mask)
        midground_counts = compute_layer_counts(midground_mask)
        background_counts = compute_layer_counts(background_mask)

        def share_from_count(count: int) -> float:
            return float(count / total_pixels * 100.0) if total_pixels else 0.0

        def edge_complexity_pct(layer_mask: np.ndarray) -> float:
            layer_pixels = int(np.sum(layer_mask))
            if layer_pixels == 0:
                return 0.0
            return float(np.sum(edges_resized & layer_mask) / layer_pixels * 100.0)

        layer_metrics = {
            "foreground_tree_shrub_share_pct": share_from_count(foreground_counts["trees"] + foreground_counts["shrubs"]),
            "foreground_grass_share_pct": share_from_count(foreground_counts["grass"]),
            "foreground_water_share_pct": share_from_count(foreground_counts["water"]),
            "foreground_roads_paving_share_pct": share_from_count(foreground_counts["roads"]),
            "foreground_people_share_pct": share_from_count(foreground_counts["people"]),
            "foreground_vehicles_share_pct": share_from_count(foreground_counts["vehicles"]),
            "foreground_structures_share_pct": share_from_count(foreground_counts["structures"]),
            "foreground_terrain_share_pct": share_from_count(foreground_counts["terrain"]),
            "foreground_bare_land_share_pct": share_from_count(foreground_counts["bare_land"]),
            "foreground_buildings_share_pct": share_from_count(foreground_counts["buildings"]),
            "midground_tree_shrub_share_pct": share_from_count(midground_counts["trees"] + midground_counts["shrubs"]),
            "midground_grass_share_pct": share_from_count(midground_counts["grass"]),
            "midground_water_share_pct": share_from_count(midground_counts["water"]),
            "midground_roads_paving_share_pct": share_from_count(midground_counts["roads"]),
            "midground_people_share_pct": share_from_count(midground_counts["people"]),
            "midground_vehicles_share_pct": share_from_count(midground_counts["vehicles"]),
            "midground_structures_share_pct": share_from_count(midground_counts["structures"]),
            "midground_terrain_share_pct": share_from_count(midground_counts["terrain"]),
            "midground_bare_land_share_pct": share_from_count(midground_counts["bare_land"]),
            "midground_buildings_share_pct": share_from_count(midground_counts["buildings"]),
            "background_plants_share_pct": share_from_count(background_counts["plants"]),
            "background_sky_share_pct": share_from_count(background_counts["sky"]),
            "background_water_share_pct": share_from_count(background_counts["water"]),
            "background_people_share_pct": share_from_count(background_counts["people"]),
            "background_vehicles_share_pct": share_from_count(background_counts["vehicles"]),
            "background_structures_share_pct": share_from_count(background_counts["structures"]),
            "background_terrain_share_pct": share_from_count(background_counts["terrain"]),
            "background_bare_land_share_pct": share_from_count(background_counts["bare_land"]),
            "background_buildings_share_pct": share_from_count(background_counts["buildings"]),
        }
        feature_metrics = {
            "foreground_naturalness_pct": share_from_count(foreground_counts["plants"] + foreground_counts["water"]),
            "foreground_disturbance_pct": share_from_count(
                foreground_counts["people"]
                + foreground_counts["vehicles"]
                + foreground_counts["bare_land"]
                + foreground_counts["trash_bins"]
                + foreground_counts["signboards"]
            ),
            "foreground_complexity_pct": edge_complexity_pct(foreground_mask),
            "foreground_intentionality_pct": share_from_count(foreground_counts["structures"] + foreground_counts["terrain"]),
            "foreground_enclosure_pct": share_from_count(
                foreground_counts["trees"]
                + foreground_counts["shrubs"]
                + foreground_counts["structures"]
                + foreground_counts["buildings"]
            ),
            "midground_naturalness_pct": share_from_count(midground_counts["plants"] + midground_counts["water"]),
            "midground_disturbance_pct": share_from_count(
                midground_counts["people"]
                + midground_counts["vehicles"]
                + midground_counts["bare_land"]
                + midground_counts["trash_bins"]
                + midground_counts["signboards"]
            ),
            "midground_complexity_pct": edge_complexity_pct(midground_mask),
            "midground_intentionality_pct": share_from_count(midground_counts["structures"] + midground_counts["terrain"]),
            "midground_enclosure_pct": share_from_count(
                midground_counts["trees"]
                + midground_counts["shrubs"]
                + midground_counts["structures"]
                + midground_counts["buildings"]
            ),
            "background_openness_pct": share_from_count(background_counts["sky"] + background_counts["water"]),
            "background_intentionality_pct": share_from_count(background_counts["structures"] + background_counts["terrain"]),
            "background_complexity_pct": edge_complexity_pct(background_mask),
            "background_disturbance_pct": share_from_count(
                background_counts["people"] + background_counts["vehicles"] + background_counts["buildings"]
            ),
        }
        color_metrics = compute_layer_color_metrics(
            image_rgb,
            image_hsv,
            [
                ("foreground", foreground_mask),
                ("midground", midground_mask),
                ("background", background_mask),
            ],
        )
        base_name = image_path.stem
        np.save(self.depth_dir / ("%s_depth_corrected.npy" % base_name), corrected_depth)
        Image.fromarray(custom_segmentation.astype(np.uint8)).save(self.segmentation_dir / ("%s_segmentation_mask.png" % base_name))
        self.create_visualization_assets(
            image_rgb,
            corrected_depth,
            custom_segmentation,
            semantic_proportions,
            feature_metrics,
            image_path.name,
        )
        result = {
            "image": image_path.name,
            "status": "success",
            "target_min_depth_m": self.target_min_depth,
            "foreground_max_depth_m": self.foreground_max_depth,
            "midground_max_depth_m": self.midground_max_depth,
        }
        result.update({"prop_%s" % key: value for key, value in semantic_proportions.items()})
        result.update(depth_stats)
        result.update(layer_metrics)
        result.update(feature_metrics)
        result.update(color_metrics)
        return result

    def create_visualization_assets(
        self,
        image_rgb: np.ndarray,
        depth_map: np.ndarray,
        segmentation_map: np.ndarray,
        semantic_proportions: Dict[str, float],
        feature_metrics: Dict[str, float],
        image_name: str,
    ) -> None:
        base_name = Path(image_name).stem

        def asset_path(suffix: str) -> Path:
            return self.visualization_dir / ("%s_%s.jpg" % (base_name, suffix))

        color_mask = np.zeros_like(image_rgb)
        for index, category_name in enumerate(CUSTOM_CATEGORIES):
            color_mask[segmentation_map == index] = CATEGORY_COLORS[category_name]
        overlay = np.clip(0.6 * image_rgb + 0.4 * color_mask, 0, 255).astype(np.uint8)
        depth_normalized = np.clip(depth_map / max(self.midground_max_depth, 1.0), 0, 1) * 255.0
        depth_visual = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_PLASMA)
        depth_visual_rgb = cv2.cvtColor(depth_visual, cv2.COLOR_BGR2RGB)
        foreground_mask = depth_map < self.foreground_max_depth
        midground_mask = (depth_map >= self.foreground_max_depth) & (depth_map < self.midground_max_depth)
        background_mask = depth_map >= self.midground_max_depth

        def highlight_layer(mask: np.ndarray) -> np.ndarray:
            highlighted = np.zeros_like(depth_visual)
            highlighted[mask] = depth_visual[mask]
            return cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB)

        self.save_rgb_panel(image_rgb, "Original Image", asset_path("01_original"))
        self.save_rgb_panel(overlay, "Original Image + Segmentation Overlay", asset_path("02_overlay"))
        self.save_rgb_panel(depth_visual_rgb, "Corrected Depth Map", asset_path("03_depth"))
        self.save_rgb_panel(highlight_layer(foreground_mask), "Foreground Layer", asset_path("04_foreground"))
        self.save_rgb_panel(highlight_layer(midground_mask), "Midground Layer", asset_path("05_midground"))
        self.save_rgb_panel(highlight_layer(background_mask), "Background Layer", asset_path("06_background"))

        figure_bar, axis_bar = plt.subplots(figsize=(8, 5), dpi=150)
        valid_props = {key: value for key, value in semantic_proportions.items() if value > 0}
        if valid_props:
            prop_keys = list(valid_props.keys())
            prop_values = list(valid_props.values())
            prop_colors = [tuple(np.array(CATEGORY_COLORS.get(category, (220, 220, 220))) / 255.0) for category in prop_keys]
            axis_bar.bar(prop_keys, prop_values, color=prop_colors, edgecolor="black")
            axis_bar.set_ylim(0, max(prop_values) * 1.15 + 5)
        else:
            axis_bar.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14)
        axis_bar.set_title("Semantic Category Share", fontsize=16)
        axis_bar.set_ylabel("Share (%)")
        axis_bar.tick_params(axis="x", rotation=45)
        self.save_matplotlib_figure(figure_bar, asset_path("07_semantic_bar"))

        figure_legend, axis_legend = plt.subplots(figsize=(6, 5), dpi=150)
        axis_legend.set_title("Color Legend", fontsize=16)
        axis_legend.axis("off")
        legend_entries = sorted(valid_props.items(), key=lambda item: item[1], reverse=True)[:12]
        if legend_entries:
            y_positions = np.linspace(0.9, 0.1, len(legend_entries))
            for (name, value), y_position in zip(legend_entries, y_positions):
                color = np.array(CATEGORY_COLORS.get(name, (200, 200, 200))) / 255.0
                rectangle = patches.Rectangle(
                    (0.05, y_position - 0.035),
                    0.1,
                    0.07,
                    transform=axis_legend.transAxes,
                    facecolor=color,
                    edgecolor="black",
                )
                axis_legend.add_patch(rectangle)
                axis_legend.text(
                    0.18,
                    y_position,
                    "%s (%.1f%%)" % (name, value),
                    transform=axis_legend.transAxes,
                    fontsize=13,
                    va="center",
                )
        else:
            axis_legend.text(0.1, 0.5, "No semantic categories detected", fontsize=13, va="center")
        self.save_matplotlib_figure(figure_legend, asset_path("08_color_legend"))

        metrics_lines = [
            "Feature Summary",
            "",
            "Foreground naturalness: %.1f%%" % feature_metrics.get("foreground_naturalness_pct", 0.0),
            "Foreground disturbance: %.1f%%" % feature_metrics.get("foreground_disturbance_pct", 0.0),
            "Foreground complexity: %.1f%%" % feature_metrics.get("foreground_complexity_pct", 0.0),
            "Foreground intentionality: %.1f%%" % feature_metrics.get("foreground_intentionality_pct", 0.0),
            "Foreground enclosure: %.1f%%" % feature_metrics.get("foreground_enclosure_pct", 0.0),
            "",
            "Midground naturalness: %.1f%%" % feature_metrics.get("midground_naturalness_pct", 0.0),
            "Midground disturbance: %.1f%%" % feature_metrics.get("midground_disturbance_pct", 0.0),
            "Midground complexity: %.1f%%" % feature_metrics.get("midground_complexity_pct", 0.0),
            "Midground intentionality: %.1f%%" % feature_metrics.get("midground_intentionality_pct", 0.0),
            "Midground enclosure: %.1f%%" % feature_metrics.get("midground_enclosure_pct", 0.0),
            "",
            "Background openness: %.1f%%" % feature_metrics.get("background_openness_pct", 0.0),
            "Background intentionality: %.1f%%" % feature_metrics.get("background_intentionality_pct", 0.0),
            "Background complexity: %.1f%%" % feature_metrics.get("background_complexity_pct", 0.0),
            "Background disturbance: %.1f%%" % feature_metrics.get("background_disturbance_pct", 0.0),
        ]
        figure_metrics, axis_metrics = plt.subplots(figsize=(6, 8), dpi=150)
        axis_metrics.axis("off")
        axis_metrics.text(0.0, 1.0, "\n".join(metrics_lines), transform=axis_metrics.transAxes, va="top", ha="left", fontsize=13)
        self.save_matplotlib_figure(figure_metrics, asset_path("09_feature_summary"))

    def process_files(self, image_paths: Sequence[Path], max_images: Optional[int] = None) -> None:
        ordered_paths = sorted(image_paths, key=lambda path: natural_sort_key(path.name))
        if max_images is not None:
            ordered_paths = ordered_paths[:max_images]
        logging.info("Found %d images to process.", len(ordered_paths))
        for image_path in tqdm(ordered_paths, desc="Processing images"):
            try:
                result = self.process_single_image(image_path)
                if result:
                    self.results.append(result)
            except Exception as exc:
                logging.exception("Unhandled error while processing %s: %s", image_path.name, exc)
            if self.device == "cuda":
                torch.cuda.empty_cache()
        self.save_summary_csv()

    def save_summary_csv(self) -> None:
        if not self.results:
            logging.warning("No results were generated.")
            return
        dataframe = pd.DataFrame(self.results)
        csv_path = self.output_dir / "landscape_spatial_analysis_summary.csv"
        dataframe.to_csv(csv_path, index=False, encoding="utf-8")
        logging.info("Saved summary table to %s", csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run spatial landscape analysis with semantic segmentation and monocular depth estimation.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input image files or directories.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/landscape_spatial_analysis"), help="Output directory.")
    parser.add_argument("--depth-model-path", required=True, help="Local path or model identifier for the depth estimation model.")
    parser.add_argument("--seg-config", required=True, help="Path to the MMSegmentation config file.")
    parser.add_argument("--seg-checkpoint", required=True, help="Path to the MMSegmentation checkpoint file.")
    parser.add_argument("--target-min-depth", type=float, default=0.5, help="Target minimum non-sky depth after rescaling.")
    parser.add_argument("--foreground-max-depth", type=float, default=10.0, help="Upper depth boundary for the foreground layer.")
    parser.add_argument("--midground-max-depth", type=float, default=30.0, help="Upper depth boundary for the midground layer.")
    parser.add_argument("--sky-depth", type=float, default=1000.0, help="Depth value assigned to sky pixels.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Execution device.")
    parser.add_argument("--max-images", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO", help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    files = list_image_files([path.resolve() for path in args.inputs])
    if not files:
        logging.error("No supported image files were found.")
        raise SystemExit(1)
    seg_config_path = Path(args.seg_config)
    seg_checkpoint_path = Path(args.seg_checkpoint)
    if not seg_config_path.exists():
        logging.error("Segmentation config not found: %s", seg_config_path)
        raise SystemExit(1)
    if not seg_checkpoint_path.exists():
        logging.error("Segmentation checkpoint not found: %s", seg_checkpoint_path)
        raise SystemExit(1)
    start_time = datetime.now()
    logging.info("Starting spatial landscape analysis at %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    analyzer = LandscapeSpatialAnalyzer(
        output_dir=args.output_dir,
        depth_model_path=args.depth_model_path,
        seg_config=str(seg_config_path),
        seg_checkpoint=str(seg_checkpoint_path),
        target_min_depth=args.target_min_depth,
        foreground_max_depth=args.foreground_max_depth,
        midground_max_depth=args.midground_max_depth,
        sky_depth=args.sky_depth,
        device=args.device,
    )
    analyzer.process_files(files, max_images=args.max_images)
    end_time = datetime.now()
    logging.info("Completed at %s", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info("Total duration: %s", end_time - start_time)


if __name__ == "__main__":
    main()
