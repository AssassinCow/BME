from __future__ import annotations

import numpy as np
from scipy import signal


def longest_false_run(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    longest = current = 0
    for value in mask:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def spectral_summary(values: np.ndarray, sampling_hz: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < 16 or np.std(values) <= 1e-12:
        return {
            "dominant_frequency": 0.0,
            "spectral_entropy": 0.0,
            "power_0p1_0p5": 0.0,
            "power_0p5_2": 0.0,
            "power_2_5": 0.0,
            "power_5_10": 0.0,
            "power_10_20": 0.0,
        }
    frequencies, power = signal.welch(
        signal.detrend(values),
        fs=sampling_hz,
        nperseg=min(len(values), 512),
    )
    total = float(power.sum()) + 1e-12
    probability = power / total
    entropy = -float(np.sum(probability * np.log(probability + 1e-12)))
    entropy /= max(np.log(len(probability)), 1e-12)

    def band(low: float, high: float) -> float:
        selected = (frequencies >= low) & (frequencies < high)
        return float(power[selected].sum() / total)

    nonzero = frequencies > 0
    dominant = float(frequencies[nonzero][np.argmax(power[nonzero])]) if nonzero.any() else 0.0
    return {
        "dominant_frequency": dominant,
        "spectral_entropy": entropy,
        "power_0p1_0p5": band(0.1, 0.5),
        "power_0p5_2": band(0.5, 2.0),
        "power_2_5": band(2.0, 5.0),
        "power_5_10": band(5.0, 10.0),
        "power_10_20": band(10.0, min(20.0, sampling_hz / 2)),
    }


def ppg_quality_features(
    values: np.ndarray,
    mask: np.ndarray,
    sampling_hz: float,
) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(values) == 0:
        return np.zeros(8, dtype=np.float32), 0.0
    valid_fraction = float(mask.mean())
    valid_values = values[mask]
    if len(valid_values) < max(16, int(sampling_hz)):
        features = np.asarray(
            [valid_fraction, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32
        )
        return features, 0.0

    gap_ratio = longest_false_run(mask) / max(len(mask), 1)
    lower, upper = np.percentile(valid_values, [0.5, 99.5])
    tolerance = max((upper - lower) * 1e-6, 1e-6)
    clipping_ratio = float(
        np.mean((valid_values <= lower + tolerance) | (valid_values >= upper - tolerance))
    )
    differences = np.diff(valid_values)
    difference_median = np.median(differences)
    difference_mad = np.median(np.abs(differences - difference_median)) + 1e-12
    derivative_outlier_ratio = float(
        np.mean(np.abs(differences - difference_median) > 6.0 * difference_mad)
    )
    flatline_ratio = float(np.mean(np.abs(differences) <= tolerance))

    detrended = signal.detrend(valid_values)
    frequencies, power = signal.welch(
        detrended,
        fs=sampling_hz,
        nperseg=min(len(detrended), 1024),
    )
    total_power = float(power.sum()) + 1e-12
    pulse_band = (frequencies >= 0.5) & (frequencies <= 4.0)
    spectral_concentration = float(power[pulse_band].sum() / total_power)
    outside_power = max(total_power - float(power[pulse_band].sum()), 1e-12)
    robust_snr = float(np.clip(np.log1p(total_power / outside_power) / np.log(11.0), 0, 1))

    centered = detrended - detrended.mean()
    autocorrelation = signal.fftconvolve(centered, centered[::-1], mode="full")
    autocorrelation = autocorrelation[len(centered) - 1 :]
    autocorrelation /= max(float(autocorrelation[0]), 1e-12)
    minimum_lag = max(1, int(round(0.25 * sampling_hz)))
    maximum_lag = min(len(autocorrelation), int(round(2.0 * sampling_hz)))
    autocorrelation_peak = (
        float(np.clip(np.max(autocorrelation[minimum_lag:maximum_lag]), 0, 1))
        if maximum_lag > minimum_lag
        else 0.0
    )

    features = np.asarray(
        [
            valid_fraction,
            gap_ratio,
            clipping_ratio,
            derivative_outlier_ratio,
            spectral_concentration,
            autocorrelation_peak,
            flatline_ratio,
            robust_snr,
        ],
        dtype=np.float32,
    )
    good_components = np.asarray(
        [
            valid_fraction,
            1.0 - np.clip(gap_ratio, 0, 1),
            1.0 - np.clip(clipping_ratio, 0, 1),
            1.0 - np.clip(derivative_outlier_ratio, 0, 1),
            np.clip(spectral_concentration, 0, 1),
            np.clip(autocorrelation_peak, 0, 1),
            1.0 - np.clip(flatline_ratio, 0, 1),
            robust_snr,
        ]
    )
    quality = float(np.exp(np.mean(np.log(np.clip(good_components, 1e-4, 1.0)))))
    return features, quality


def robust_statistics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            name: 0.0
            for name in (
                "mean",
                "std",
                "median",
                "mad",
                "minimum",
                "maximum",
                "range",
                "rms",
                "energy",
                "iqr",
                "zero_crossing_rate",
                "slope",
            )
        }
    median = float(np.median(values))
    centered = values - float(np.mean(values))
    x = np.linspace(-1.0, 1.0, len(values))
    slope = float(np.dot(x, centered) / max(np.dot(x, x), 1e-12))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": median,
        "mad": float(np.median(np.abs(values - median))),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "range": float(np.ptp(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "energy": float(np.mean(np.square(values))),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "zero_crossing_rate": float(np.mean(centered[:-1] * centered[1:] < 0))
        if len(values) > 1
        else 0.0,
        "slope": slope,
    }

