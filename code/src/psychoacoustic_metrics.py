#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, filtfilt, hilbert
from mosqito.sq_metrics import loudness_zwst, roughness_dw, sharpness_din_from_loudness

try:
    from mosqito.sq_metrics import pr_ecma_st
except ImportError:
    pr_ecma_st = None


P_REF = 20e-6
EPS = 1e-12
SUPPORTED_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def a_weighting_response(freqs_hz: np.ndarray) -> np.ndarray:
    freqs = np.maximum(freqs_hz, 1e-10)
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217
    numerator = (f4 ** 2) * (freqs ** 4)
    denominator = (
        (freqs ** 2 + f1 ** 2)
        * np.sqrt((freqs ** 2 + f2 ** 2) * (freqs ** 2 + f3 ** 2))
        * (freqs ** 2 + f4 ** 2)
    )
    response = numerator / denominator
    return 20.0 * np.log10(response) + 2.0


def compute_laeq_db(signal: np.ndarray, fs: int) -> float:
    if signal.size == 0:
        raise ValueError("Empty signal supplied for LAeq computation.")
    fft_values = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    a_linear = 10 ** (a_weighting_response(freqs) / 20.0)
    weighted_fft = fft_values * a_linear
    weighted_signal = np.fft.irfft(weighted_fft, n=signal.size)
    rms = np.sqrt(np.mean(np.square(weighted_signal)))
    if rms <= 0:
        raise ValueError("Non-positive RMS encountered during LAeq computation.")
    return 20.0 * np.log10(rms / P_REF)


def compute_psychoacoustic_annoyance(
    loudness: float,
    sharpness: float,
    roughness: float,
    fluctuation: float,
) -> float:
    if loudness <= 0:
        return float("nan")
    sharpness_weight = max(0.0, 0.4 * (sharpness - 1.0))
    fluctuation_weight = max(0.0, 0.4 * (fluctuation - 0.5))
    roughness_weight = max(0.0, 0.3 * (roughness - 0.5))
    annoyance = loudness * (
        1.0 + np.sqrt(sharpness_weight ** 2 + fluctuation_weight ** 2 + roughness_weight ** 2)
    )
    return float(annoyance)


def envelope_modulation(signal_pa: np.ndarray, fs: int) -> Tuple[float, float]:
    analytic = hilbert(signal_pa)
    envelope = np.abs(analytic) + EPS
    mean_envelope = float(np.mean(envelope))
    if mean_envelope <= EPS:
        return 0.0, float("nan")
    modulation = envelope - mean_envelope
    nyquist = fs / 2.0
    cutoff = min(20.0, 0.45 * nyquist)
    if 0 < cutoff < nyquist:
        b_coeff, a_coeff = butter(4, cutoff / nyquist, btype="low")
        modulation = filtfilt(b_coeff, a_coeff, modulation)
    modulation_rms = float(np.sqrt(np.mean(np.square(modulation))))
    depth = modulation_rms / mean_envelope
    if modulation.size < 16:
        return depth, float("nan")
    freq_axis = np.fft.rfftfreq(modulation.size, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(modulation))
    valid = (freq_axis >= 0.5) & (freq_axis <= min(20.0, nyquist))
    if not np.any(valid):
        return depth, float("nan")
    dominant_index = int(np.argmax(spectrum[valid]))
    dominant_frequency = float(freq_axis[valid][dominant_index])
    return depth, dominant_frequency


def derive_vacil_scale() -> float:
    fs = 48000
    duration = 2.0
    time_axis = np.arange(0.0, duration, 1.0 / fs)
    carrier = np.sin(2.0 * np.pi * 1000.0 * time_axis)
    modulator = 0.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * time_axis))
    signal = carrier * modulator
    signal /= np.sqrt(np.mean(np.square(signal)))
    target_rms = P_REF * 10 ** (60.0 / 20.0)
    signal *= target_rms
    depth, _ = envelope_modulation(signal, fs)
    if depth <= 0.0:
        return 1.0
    return 1.0 / depth


VACIL_SCALE = derive_vacil_scale()


def compute_fluctuation_strength(signal_pa: np.ndarray, fs: int) -> Tuple[float, float, float]:
    if signal_pa.size == 0:
        return float("nan"), float("nan"), float("nan")
    depth, dominant_frequency = envelope_modulation(signal_pa, fs)
    fluctuation_vacil = depth * VACIL_SCALE
    return float(fluctuation_vacil), float(depth), float(dominant_frequency)


def list_audio_files(inputs: Sequence[Path]) -> List[Path]:
    files = []
    for item in inputs:
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)
        elif item.is_dir():
            for extension in SUPPORTED_EXTENSIONS:
                files.extend(sorted(item.rglob("*" + extension)))
        else:
            logging.warning("Skipping unsupported path: %s", item)
    return sorted({path.resolve() for path in files})


def ensure_mono(signal: np.ndarray) -> np.ndarray:
    if signal.ndim == 1:
        return signal
    return np.mean(signal, axis=1)


@dataclass
class PsychoacousticMetrics:
    filename: str
    laeq_db: float
    loudness_sone: float
    sharpness_acum: float
    roughness_asper: float
    fluctuation_vacil: float
    annoyance: float
    modulation_depth: float
    modulation_frequency_hz: float
    tonality_pr_db: Optional[float] = None


class PsychoacousticExtractor:
    def __init__(
        self,
        calibration_offset_db: float = 0.0,
        field_type: str = "free",
        max_duration: Optional[float] = None,
        include_tonality: bool = False,
    ) -> None:
        self.calibration_offset_db = calibration_offset_db
        self.field_type = field_type
        self.max_duration = max_duration
        self.include_tonality = include_tonality and pr_ecma_st is not None
        self.calibration_gain = 10 ** (calibration_offset_db / 20.0)
        if include_tonality and pr_ecma_st is None:
            logging.warning("Tonality calculation requested, but pr_ecma_st is unavailable in the installed moSQITo package.")

    def process_file(self, path: Path) -> PsychoacousticMetrics:
        logging.info("Processing %s", path)
        signal, fs = sf.read(path)
        signal = ensure_mono(np.asarray(signal, dtype=float))
        if signal.size == 0:
            raise ValueError("%s contains no samples." % path)
        if self.max_duration is not None:
            max_samples = int(self.max_duration * fs)
            if signal.size > max_samples:
                signal = signal[:max_samples]
        signal_pa = signal * self.calibration_gain
        laeq_db = compute_laeq_db(signal, fs) + self.calibration_offset_db
        loudness_values, specific_loudness, _ = loudness_zwst(signal_pa, fs=fs, field_type=self.field_type)
        loudness_array = np.atleast_1d(loudness_values)
        sharpness_values = sharpness_din_from_loudness(loudness_array, np.atleast_2d(specific_loudness))
        loudness = float(np.mean(loudness_array))
        sharpness = float(np.mean(np.atleast_1d(sharpness_values)))
        roughness_values, _, _, _ = roughness_dw(signal_pa, fs=fs, overlap=0.0)
        roughness = float(np.mean(np.atleast_1d(roughness_values)))
        fluctuation_vacil, modulation_depth, modulation_frequency = compute_fluctuation_strength(signal_pa, fs)
        annoyance = compute_psychoacoustic_annoyance(loudness, sharpness, roughness, fluctuation_vacil)
        tonality_pr_db = None
        if self.include_tonality and pr_ecma_st is not None:
            total_pr, _, _, _ = pr_ecma_st(signal_pa, fs=fs, prominence=True)
            if total_pr is not None:
                tonality_pr_db = float(np.atleast_1d(total_pr)[0])
        return PsychoacousticMetrics(
            filename=path.name,
            laeq_db=float(laeq_db),
            loudness_sone=loudness,
            sharpness_acum=sharpness,
            roughness_asper=roughness,
            fluctuation_vacil=fluctuation_vacil,
            annoyance=annoyance,
            modulation_depth=modulation_depth,
            modulation_frequency_hz=modulation_frequency,
            tonality_pr_db=tonality_pr_db,
        )

    def batch_process(self, paths: Iterable[Path]) -> List[PsychoacousticMetrics]:
        results = []
        for path in paths:
            try:
                results.append(self.process_file(path))
            except Exception as exc:
                logging.error("Failed to process %s: %s", path, exc)
        return results


def metrics_to_dataframe(metrics: Sequence[PsychoacousticMetrics]) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for item in metrics])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract psychoacoustic metrics from audio files.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input audio files or directories.")
    parser.add_argument("--output", type=Path, default=Path("outputs/psychoacoustic_metrics.csv"), help="Output CSV file.")
    parser.add_argument("--calibration-db", type=float, default=0.0, help="Calibration offset in decibels.")
    parser.add_argument("--field-type", choices=("free", "diffuse"), default="free", help="Sound field assumption for loudness.")
    parser.add_argument("--max-duration", type=float, default=None, help="Maximum analysis duration per file in seconds.")
    parser.add_argument("--include-tonality", action="store_true", help="Include prominence ratio when available.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO", help="Logging verbosity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    files = list_audio_files([path.resolve() for path in args.inputs])
    if not files:
        logging.error("No supported audio files were found.")
        raise SystemExit(1)
    extractor = PsychoacousticExtractor(
        calibration_offset_db=args.calibration_db,
        field_type=args.field_type,
        max_duration=args.max_duration,
        include_tonality=args.include_tonality,
    )
    metrics = extractor.batch_process(files)
    if not metrics:
        logging.error("No metrics were generated.")
        raise SystemExit(1)
    dataframe = metrics_to_dataframe(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False, encoding="utf-8")
    logging.info("Saved %d records to %s", len(dataframe), args.output)


if __name__ == "__main__":
    main()
