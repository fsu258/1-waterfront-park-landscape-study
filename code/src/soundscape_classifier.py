#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow_hub as hub

warnings.filterwarnings("ignore")


AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
CATEGORY_DISPLAY_NAMES = {
    "speech": "Speech",
    "nature": "Nature",
    "traffic_mechanical": "Traffic and Mechanical",
    "music": "Music",
    "silence": "Silence",
}
CATEGORY_COLORS = {
    "speech": "#FF6B6B",
    "nature": "#4ECDC4",
    "traffic_mechanical": "#45B7D1",
    "music": "#F7D154",
    "silence": "#96CEB4",
}
CATEGORY_KEYWORDS = {
    "speech": [
        "Speech",
        "Conversation",
        "Narration",
        "Babbling",
        "Crowd",
        "Laughter",
        "Crying",
        "Shout",
        "Screaming",
        "Whispering",
        "Child speech",
        "Man speaking",
        "Woman speaking",
        "Chatter",
        "Children playing",
        "Children shouting",
    ],
    "nature": [
        "Animal",
        "Bird",
        "Bird vocalization",
        "Chirp",
        "Tweet",
        "Water",
        "Stream",
        "River",
        "Waves",
        "Wind",
        "Rain",
        "Thunder",
        "Rustling leaves",
        "Insect",
        "Frog",
        "Cricket",
        "Bee",
        "Wasp",
        "Fly",
        "Mosquito",
        "Duck",
        "Goose",
        "Crow",
        "Caw",
        "Pigeon",
        "Dove",
        "Coo",
        "Owl",
        "Hoot",
    ],
    "traffic_mechanical": [
        "Vehicle",
        "Car",
        "Bus",
        "Truck",
        "Motorcycle",
        "Train",
        "Aircraft",
        "Helicopter",
        "Engine",
        "Motor",
        "Accelerating",
        "Car passing by",
        "Traffic noise",
        "Honk",
        "Car alarm",
        "Siren",
        "Emergency vehicle",
        "Bicycle",
        "Skateboard",
        "Motor vehicle",
        "Road",
        "Brake",
        "Tire squeal",
    ],
    "music": [
        "Music",
        "Musical instrument",
        "Plucked string instrument",
        "String instrument",
        "Keyboard instrument",
        "Wind instrument",
        "Percussion",
        "Song",
        "Singing",
        "Guitar",
        "Piano",
        "Violin",
        "Drum",
        "Flute",
        "Saxophone",
        "Trumpet",
        "Orchestra",
    ],
    "silence": ["Silence"],
}
SUMMARY_LABELS = {
    "speech": "Speech-dominant environment",
    "nature": "Nature-dominant environment",
    "traffic_mechanical": "Traffic- and mechanical-noise-dominant environment",
    "music": "Music-dominant environment",
    "silence": "Quiet environment",
}


def list_audio_files(inputs: Sequence[Path]) -> List[Path]:
    files = []
    for item in inputs:
        if item.is_file() and item.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(item)
        elif item.is_dir():
            for extension in AUDIO_EXTENSIONS:
                files.extend(sorted(item.rglob("*" + extension)))
    return sorted({path.resolve() for path in files})


class SoundscapeClassifier:
    def __init__(self) -> None:
        logging.info("Loading YAMNet model.")
        self.model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.class_names = self.load_yamnet_classes()
        self.class_names_list = self.class_names.tolist()
        self.frame_hop_seconds = 0.48
        self.category_indices = {}
        for category, keywords in CATEGORY_KEYWORDS.items():
            if category == "silence":
                continue
            indices = set()
            for keyword in keywords:
                for index, class_name in enumerate(self.class_names):
                    if keyword.lower() in class_name.lower():
                        indices.add(index)
            self.category_indices[category] = sorted(indices)
        logging.info("YAMNet is ready.")

    def load_yamnet_classes(self) -> np.ndarray:
        class_map_url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        try:
            return pd.read_csv(class_map_url)["display_name"].values
        except Exception as exc:
            logging.warning("Falling back to placeholder YAMNet class labels: %s", exc)
            return np.array(["Class_%d" % i for i in range(521)])

    def load_audio(self, audio_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
        try:
            waveform, sample_rate = sf.read(audio_path, dtype="float32")
        except Exception:
            waveform, sample_rate = librosa.load(str(audio_path), sr=None)
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        if sample_rate != target_sr:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
        return waveform, sample_rate

    def classify_audio_by_second(self, audio_path: Path) -> Tuple[List[Dict[str, float]], np.ndarray, int]:
        waveform, sample_rate = self.load_audio(audio_path)
        if len(waveform) == 0:
            logging.warning("Skipping empty audio file: %s", audio_path)
            return [], waveform, sample_rate
        scores, _, _ = self.model(waveform)
        scores = scores.numpy()
        frame_timestamps = np.arange(scores.shape[0]) * self.frame_hop_seconds
        duration_seconds = len(waveform) / sample_rate
        number_of_seconds = int(np.ceil(duration_seconds))
        per_second_results = []
        for second in range(number_of_seconds):
            frame_indices = np.where((frame_timestamps >= second) & (frame_timestamps < second + 1))[0]
            if frame_indices.size > 0:
                second_scores = np.mean(scores[frame_indices], axis=0)
            else:
                second_scores = np.zeros(521)
                try:
                    silence_index = self.class_names_list.index("Silence")
                    second_scores[silence_index] = 1.0
                except ValueError:
                    pass
            per_second_results.append(self.map_to_categories(second_scores))
        return per_second_results, waveform, sample_rate

    def map_to_categories(self, yamnet_scores: np.ndarray) -> Dict[str, float]:
        category_scores = {}
        for category in CATEGORY_DISPLAY_NAMES:
            if category == "silence":
                continue
            indices = self.category_indices.get(category, [])
            category_scores[category] = float(np.max(yamnet_scores[indices])) if indices else 0.0
        silence_sensitivity_power = 3.0
        max_score = max(category_scores.values()) if category_scores else 0.0
        silence_score = (1.0 - max_score) ** silence_sensitivity_power
        try:
            raw_silence_score = float(yamnet_scores[self.class_names_list.index("Silence")])
            category_scores["silence"] = max(silence_score, raw_silence_score)
        except ValueError:
            category_scores["silence"] = max(0.0, silence_score)
        total = sum(category_scores.values())
        if total > 1e-6:
            for key in category_scores:
                category_scores[key] /= total
        else:
            for key in category_scores:
                category_scores[key] = 0.0
            category_scores["silence"] = 1.0
        return {key: float(category_scores[key]) for key in CATEGORY_DISPLAY_NAMES}

    def visualize_temporal_analysis(
        self,
        audio_path: Path,
        waveform: np.ndarray,
        sample_rate: int,
        per_second_results: List[Dict[str, float]],
        save_path: Path,
        show_plot: bool = False,
    ) -> None:
        if not per_second_results:
            return
        categories = list(CATEGORY_DISPLAY_NAMES.keys())
        labels = [CATEGORY_DISPLAY_NAMES[key] for key in categories]
        colors = [CATEGORY_COLORS[key] for key in categories]
        figure = plt.figure(figsize=(20, 12))
        axis_waveform = plt.subplot(3, 2, (1, 2))
        time_axis = np.linspace(0, len(waveform) / sample_rate, num=len(waveform))
        axis_waveform.plot(time_axis, waveform, color="steelblue", linewidth=0.5)
        axis_waveform.set_title("Waveform", fontsize=14, fontweight="bold")
        axis_waveform.set_xlabel("Time (s)")
        axis_waveform.set_ylabel("Amplitude")
        axis_waveform.grid(True, alpha=0.3)
        axis_waveform.set_xlim(0, len(waveform) / sample_rate)
        axis_spectrogram = plt.subplot(3, 2, 3)
        spectrogram_db = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)), ref=np.max)
        librosa.display.specshow(spectrogram_db, sr=sample_rate, x_axis="time", y_axis="log", ax=axis_spectrogram)
        axis_spectrogram.set_title("Log-Frequency Spectrogram", fontsize=14, fontweight="bold")
        axis_spectrogram.set_xlabel("Time (s)")
        axis_spectrogram.set_ylabel("Frequency (Hz)")
        axis_temporal = plt.subplot(3, 2, 4)
        dataframe = pd.DataFrame(per_second_results)[categories]
        plot_dataframe = dataframe.rename(columns=CATEGORY_DISPLAY_NAMES)
        plot_dataframe.plot(kind="bar", stacked=True, ax=axis_temporal, color=colors, width=1.0)
        axis_temporal.set_title("Per-Second Category Distribution", fontsize=14, fontweight="bold")
        axis_temporal.set_xlabel("Second")
        axis_temporal.set_ylabel("Probability")
        axis_temporal.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        tick_interval = max(1, len(plot_dataframe) // 10)
        tick_positions = np.arange(0, len(plot_dataframe), tick_interval)
        axis_temporal.set_xticks(tick_positions)
        axis_temporal.set_xticklabels(tick_positions)
        axis_pie = plt.subplot(3, 2, 5)
        overall_scores = dataframe.mean()
        wedges, _, autotexts = axis_pie.pie(
            overall_scores.values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.05] * len(categories),
            pctdistance=0.85,
        )
        plt.setp(autotexts, size=10, weight="bold", color="white")
        axis_pie.set_title("Average Composition", fontsize=14, fontweight="bold")
        dominant_category = str(overall_scores.idxmax())
        details_text = "\n".join(
            ["- %s: %.2f%%" % (CATEGORY_DISPLAY_NAMES[category], overall_scores[category] * 100.0) for category in categories]
        )
        summary_text = (
            "Analysis Summary\n\n"
            "Audio file: %s\n"
            "Duration: %.2f s\n\n"
            "Dominant category: %s\n"
            "Mean confidence: %.2f%%\n\n"
            "Category means:\n%s\n\n"
            "Interpretation:\n%s"
        ) % (
            audio_path.name,
            len(waveform) / sample_rate,
            CATEGORY_DISPLAY_NAMES[dominant_category],
            overall_scores[dominant_category] * 100.0,
            details_text,
            SUMMARY_LABELS[dominant_category],
        )
        axis_summary = plt.subplot(3, 2, 6)
        axis_summary.axis("off")
        axis_summary.text(
            0.05,
            0.5,
            summary_text,
            fontsize=12,
            verticalalignment="center",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "#F0F0F0", "alpha": 0.8},
        )
        plt.suptitle("Temporal Soundscape Analysis: %s" % audio_path.name, fontsize=18, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show_plot:
            plt.show()
        plt.close(figure)

    def process_files(
        self,
        inputs: Sequence[Path],
        output_csv: Path,
        visualization_dir: Path,
        show_plots: bool = False,
    ) -> pd.DataFrame:
        files = list_audio_files(inputs)
        if not files:
            raise ValueError("No supported audio files were found.")
        visualization_dir.mkdir(parents=True, exist_ok=True)
        results = []
        total_files = len(files)
        for index, audio_file in enumerate(files, start=1):
            logging.info("Processing %d/%d: %s", index, total_files, audio_file.name)
            try:
                per_second_results, waveform, sample_rate = self.classify_audio_by_second(audio_file)
                if not per_second_results:
                    continue
                for second_index, scores in enumerate(per_second_results):
                    row = {"file_name": audio_file.name, "second_index": second_index}
                    row.update(scores)
                    row["dominant_category"] = max(scores, key=scores.get)
                    results.append(row)
                figure_path = visualization_dir / ("%s_temporal_analysis.png" % audio_file.stem)
                self.visualize_temporal_analysis(
                    audio_file,
                    waveform,
                    sample_rate,
                    per_second_results,
                    figure_path,
                    show_plot=show_plots,
                )
            except Exception as exc:
                logging.error("Failed to process %s: %s", audio_file, exc)
        dataframe = pd.DataFrame(results)
        if dataframe.empty:
            raise ValueError("No classification results were generated.")
        ordered_columns = ["file_name", "second_index"] + list(CATEGORY_DISPLAY_NAMES.keys()) + ["dominant_category"]
        dataframe = dataframe[ordered_columns]
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_csv, index=False, encoding="utf-8")
        logging.info("Saved %d per-second records to %s", len(dataframe), output_csv)
        return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify soundscape composition with YAMNet at one-second resolution.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input audio files or directories.")
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/soundscape_per_second.csv"), help="Output CSV file.")
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=Path("outputs/soundscape_figures"),
        help="Directory for temporal analysis figures.",
    )
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO", help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    classifier = SoundscapeClassifier()
    classifier.process_files(args.inputs, args.output_csv, args.visualization_dir, show_plots=args.show_plots)


if __name__ == "__main__":
    main()
