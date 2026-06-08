#!/usr/bin/env python3
"""Calibrate IMU/camera XYZ and OPK corrections with Orthority.

Each cost-function evaluation:

1. applies one shared XYZ/OPK correction to the IMU data;
2. orthorectifies every configured raw image with Orthority;
3. compares every result with its manually orthorectified reference; and
4. averages the image costs.

The configuration can contain one image pair or multiple image pairs. Using
several images, especially at different headings and over non-flat terrain,
constrains the six parameters better than a single image.

Example:

    python3 calibrate_orthority_imu.py calibration.yaml

Configuration:

    flight_name: piper01
    imu: /path/to/piper01_safire.gpkg
    int_param: /path/to/piper01_int_param.yaml
    dem: /path/to/piper01_dem_1m.tif
    output: imu_camera_calibration.json
    work_dir: /tmp/orthority_imu_calibration

    initial_xyz: [0.0, 0.0, 0.0]       # metres, aircraft frame
    initial_opk: [0.0, 0.0, 0.0]       # degrees
    xyz_bounds: [[-10, 10], [-10, 10], [-10, 10]]
    opk_bounds: [[-20, 20], [-20, 20], [-30, 30]]

    pairs:
      - image: /path/to/f1-000000001.tif
        reference: /path/to/f1-000000001_modified.tif
        image_band: 1
        reference_band: 1
      - image: /path/to/f1-000000002.tif
        reference: /path/to/f1-000000002_modified.tif
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import geopandas as gpd
import numpy as np
import orthority as oty
import rasterio
import yaml
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.optimize import minimize
from scipy.stats import qmc

import imuNcOoGeojson


PARAMETER_NAMES = ("x", "y", "z", "omega", "phi", "kappa")


@dataclass(frozen=True)
class ImagePair:
    image: Path
    reference: Path
    image_band: int = 1
    reference_band: int = 1
    weight: float = 1.0


@dataclass(frozen=True)
class PreparedReference:
    pair: ImagePair
    crs: object
    transform: Affine
    width: int
    height: int
    image: np.ndarray
    gradient: np.ndarray
    valid: np.ndarray


def _read_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The calibration configuration must be a YAML mapping.")
    pose_model = config.get(
        "pose_model", imuNcOoGeojson.DEFAULT_POSE_MODEL
    )
    if pose_model not in imuNcOoGeojson.SUPPORTED_POSE_MODELS:
        raise ValueError(
            f"Unknown pose_model '{pose_model}'. Expected one of "
            f"{sorted(imuNcOoGeojson.SUPPORTED_POSE_MODELS)}."
        )
    config["pose_model"] = pose_model
    base = path.resolve().parent

    if not config.get("flight_name"):
        raise ValueError("Missing required configuration key: flight_name")

    for key in ("imu", "int_param", "dem", "output", "work_dir"):
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
        value = Path(config[key]).expanduser()
        config[key] = value if value.is_absolute() else base / value

    raw_pairs = config.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("'pairs' must contain at least one image/reference pair.")

    pairs = []
    for index, raw_pair in enumerate(raw_pairs, start=1):
        if not isinstance(raw_pair, dict):
            raise ValueError(f"Pair {index} must be a YAML mapping.")
        try:
            image = Path(raw_pair["image"]).expanduser()
            reference = Path(raw_pair["reference"]).expanduser()
        except KeyError as error:
            raise ValueError(f"Pair {index} requires image and reference.") from error
        pairs.append(
            ImagePair(
                image=image if image.is_absolute() else base / image,
                reference=reference if reference.is_absolute() else base / reference,
                image_band=int(raw_pair.get("image_band", 1)),
                reference_band=int(raw_pair.get("reference_band", 1)),
                weight=float(raw_pair.get("weight", 1.0)),
            )
        )
    config["pairs"] = pairs
    return config


def _vector(config: dict, key: str, default: Sequence[float]) -> np.ndarray:
    value = np.asarray(config.get(key, default), dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain three finite numbers.")
    return value


def _bounds(config: dict) -> tuple[np.ndarray, np.ndarray]:
    try:
        xyz = np.asarray(
            config.get("xyz_bounds", [[-10, 10], [-10, 10], [-10, 10]]),
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "xyz_bounds must contain exactly three numeric [min, max] pairs."
        ) from error
    try:
        opk = np.asarray(
            config.get("opk_bounds", [[-20, 20], [-20, 20], [-30, 30]]),
            dtype=float,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "opk_bounds must contain exactly three numeric [min, max] pairs."
        ) from error
    bounds = np.vstack((xyz, opk))
    if bounds.shape != (6, 2) or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("XYZ and OPK bounds must each contain three [min, max] pairs.")
    return bounds[:, 0], bounds[:, 1]


def _active_parameter_indices(config: dict) -> np.ndarray:
    names = config.get("optimize_parameters", list(PARAMETER_NAMES))
    if not isinstance(names, list) or not names:
        raise ValueError("optimize_parameters must be a non-empty list.")
    unknown = [name for name in names if name not in PARAMETER_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown optimize_parameters: {', '.join(map(str, unknown))}"
        )
    if len(set(names)) != len(names):
        raise ValueError("optimize_parameters contains duplicate names.")
    return np.asarray([PARAMETER_NAMES.index(name) for name in names], dtype=int)


def _starting_parameters(
    config: dict,
    default: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active_indices: np.ndarray,
) -> list[np.ndarray]:
    raw_starts = config.get("starting_corrections")
    if raw_starts is None:
        starts = [default.copy()]
    else:
        if not isinstance(raw_starts, list) or not raw_starts:
            raise ValueError("starting_corrections must be a non-empty list.")

        starts = []
        for index, raw_start in enumerate(raw_starts, start=1):
            if not isinstance(raw_start, dict):
                raise ValueError(
                    f"starting_corrections item {index} must be a mapping."
                )
            xyz = np.asarray(raw_start.get("xyz", default[:3]), dtype=float)
            opk = np.asarray(raw_start.get("opk", default[3:]), dtype=float)
            if xyz.shape != (3,) or opk.shape != (3,):
                raise ValueError(
                    f"starting_corrections item {index} requires 3 XYZ and 3 OPK values."
                )
            start = np.concatenate((xyz, opk))
            if not np.all(np.isfinite(start)):
                raise ValueError(
                    f"starting_corrections item {index} contains non-finite values."
                )
            starts.append(start)

    screening_samples = int(config.get("screening_samples", 0))
    if screening_samples < 0:
        raise ValueError("screening_samples must be greater than or equal to zero.")
    if screening_samples:
        seed = int(config.get("screening_seed", 0))
        sampler = qmc.Sobol(
            d=len(active_indices),
            scramble=True,
            seed=seed,
        )
        sample_power = int(math.ceil(math.log2(screening_samples)))
        samples = sampler.random_base2(m=sample_power)[:screening_samples]
        sampled_values = qmc.scale(
            samples,
            lower[active_indices],
            upper[active_indices],
        )
        for values in sampled_values:
            start = default.copy()
            start[active_indices] = values
            starts.append(start)
    return starts


def _load_imu(path: Path):
    imu = gpd.read_file(path)
    rename = {
        "datetime_utc": "time",
        "latitude": "LATITUDE",
        "longitude": "LONGITUDE",
        "altitude_m": "ALTITUDE",
        "roll_deg": "ROLL_smooth",
        "pitch_deg": "PITCH_smooth",
        "heading_deg": "THEAD_smooth",
    }
    imu = imu.rename(columns={key: value for key, value in rename.items() if key in imu})
    required = [
        "time",
        "LATITUDE",
        "LONGITUDE",
        "ALTITUDE",
        "ROLL_smooth",
        "PITCH_smooth",
        "THEAD_smooth",
    ]
    missing = [column for column in required if column not in imu]
    if missing:
        raise ValueError(f"IMU file is missing columns: {', '.join(missing)}")
    imu = imu.dropna(subset=required).sort_values("time").reset_index(drop=True)
    if len(imu) < 2:
        raise ValueError("The IMU file has fewer than two complete records.")
    return imu


def _gradient(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = image[valid]
    if values.size < 1000:
        raise ValueError("Fewer than 1000 valid reference pixels are available.")
    low, high = np.percentile(values, (2, 98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Image has insufficient radiometric contrast.")
    normalized = np.zeros(image.shape, dtype=np.float32)
    normalized[valid] = np.clip((image[valid] - low) / (high - low), 0, 1)
    normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=1, sigmaY=1)
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    result = cv2.magnitude(gx, gy)
    result[~valid] = 0
    return result


def _prepare_reference(pair: ImagePair, max_dimension: int) -> PreparedReference:
    with rasterio.open(pair.reference) as dataset:
        if dataset.crs is None:
            raise ValueError(f"{pair.reference} must have a CRS.")
        if pair.reference_band < 1 or pair.reference_band > dataset.count:
            raise ValueError(f"Invalid reference band for {pair.reference}.")
        scale = max(1, math.ceil(max(dataset.width, dataset.height) / max_dimension))
        width = math.ceil(dataset.width / scale)
        height = math.ceil(dataset.height / scale)
        transform = dataset.transform * Affine.scale(scale)
        image = dataset.read(
            pair.reference_band,
            out_shape=(height, width),
            resampling=Resampling.bilinear,
        ).astype(np.float32)
        valid = (
            dataset.read_masks(
                pair.reference_band,
                out_shape=(height, width),
                resampling=Resampling.nearest,
            )
            > 0
        )
        valid &= np.isfinite(image)
        return PreparedReference(
            pair=pair,
            crs=dataset.crs,
            transform=transform,
            width=width,
            height=height,
            image=image,
            gradient=_gradient(image, valid),
            valid=valid,
        )


def _candidate_on_reference(
    ortho_path: Path, prepared: PreparedReference
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair = prepared.pair
    with rasterio.open(ortho_path) as dataset:
        if pair.image_band < 1 or pair.image_band > dataset.count:
            raise ValueError(f"Invalid image band for {pair.image}.")
        image = np.full((prepared.height, prepared.width), np.nan, np.float32)
        reproject(
            source=rasterio.band(dataset, pair.image_band),
            destination=image,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            src_nodata=dataset.nodata,
            dst_transform=prepared.transform,
            dst_crs=prepared.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
            init_dest_nodata=True,
        )
        source_mask = dataset.read_masks(pair.image_band)
        valid = np.zeros((prepared.height, prepared.width), np.uint8)
        reproject(
            source=source_mask,
            destination=valid,
            src_transform=dataset.transform,
            src_crs=dataset.crs,
            src_nodata=0,
            dst_transform=prepared.transform,
            dst_crs=prepared.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
            init_dest_nodata=True,
        )
    mask = (valid > 0) & np.isfinite(image)
    return image, _gradient(image, mask), mask


def plot_validation(
    prepared: PreparedReference,
    candidate_image: np.ndarray,
    candidate_gradient: np.ndarray,
    candidate_valid: np.ndarray,
    output_path: Path,
    show: bool = False,
) -> None:
    """Plot the same difference/contour diagnostic as the old residual."""
    import matplotlib.pyplot as plt

    common = prepared.valid & candidate_valid
    difference = np.full(prepared.gradient.shape, np.nan, dtype=np.float32)
    difference[common] = (
        candidate_gradient[common] - prepared.gradient[common]
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    difference_plot = axes[0].imshow(difference, cmap="RdBu_r")
    axes[0].set_title("Corrected gradient - reference gradient")
    figure.colorbar(difference_plot, ax=axes[0], shrink=0.8)

    reference_levels = prepared.gradient[prepared.valid]
    candidate_levels = candidate_gradient[candidate_valid]
    if reference_levels.size:
        axes[1].contour(
            prepared.gradient,
            levels=[np.percentile(reference_levels, 80)],
            colors="black",
            linewidths=0.7,
        )
    if candidate_levels.size:
        axes[1].contour(
            candidate_gradient,
            levels=[np.percentile(candidate_levels, 80)],
            colors="yellow",
            linewidths=0.7,
        )
    axes[1].set_title(
        "Black=manual reference, yellow=corrected Orthority"
    )
    axes[1].set_facecolor("white")
    axes[1].set_xlim(0, prepared.width - 1)
    axes[1].set_ylim(prepared.height - 1, 0)
    axes[1].set_aspect("equal")

    for axis in axes:
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(f"Validation plot: {output_path}")
    if show:
        plt.show()
    plt.close(figure)


def _normalized_correlation(
    reference: np.ndarray, candidate: np.ndarray, valid: np.ndarray
) -> float:
    reference_values = reference[valid].astype(np.float64)
    candidate_values = candidate[valid].astype(np.float64)
    reference_values -= reference_values.mean()
    candidate_values -= candidate_values.mean()
    denominator = np.linalg.norm(reference_values) * np.linalg.norm(candidate_values)
    if denominator <= 0:
        return -1.0
    return float(np.dot(reference_values, candidate_values) / denominator)


class CalibrationObjective:
    def __init__(
        self,
        config: dict,
        imu,
        references: list[PreparedReference],
        lower: np.ndarray,
        upper: np.ndarray,
        fixed_parameters: np.ndarray | None = None,
        active_indices: np.ndarray | None = None,
    ):
        self.config = config
        self.imu = imu
        self.references = references
        self.lower = lower
        self.upper = upper
        self.fixed_parameters = (
            np.asarray(fixed_parameters, dtype=float)
            if fixed_parameters is not None
            else None
        )
        self.active_indices = (
            np.asarray(active_indices, dtype=int)
            if active_indices is not None
            else None
        )
        self.work_dir = Path(config["work_dir"])
        self.evaluation = 0
        self.best_cost = float("inf")
        self.best_parameters = None
        self.overlap_penalty = float(config.get("overlap_penalty", 0.5))
        self.keep_outputs = False
        self.show_plots = False
        self.plot_outputs = True

    def physical_parameters(self, normalized: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized, dtype=float)
        if self.active_indices is None:
            return self.lower + normalized * (self.upper - self.lower)
        parameters = self.fixed_parameters.copy()
        active = self.active_indices
        parameters[active] = (
            self.lower[active]
            + normalized * (self.upper[active] - self.lower[active])
        )
        return parameters

    def __call__(self, normalized: np.ndarray) -> float:
        self.evaluation += 1
        parameters = self.physical_parameters(np.asarray(normalized))
        correction_xyz = parameters[:3]
        correction_opk = parameters[3:]
        evaluation_dir = self.work_dir / (
            "validation" if self.keep_outputs else "current"
        )
        if evaluation_dir.exists():
            shutil.rmtree(evaluation_dir)
        evaluation_dir.mkdir(parents=True)

        try:
            images = [str(reference.pair.image) for reference in self.references]
            imuNcOoGeojson.imutogeojson(
                self.imu,
                str(evaluation_dir),
                "",
                self.config["flight_name"],
                correction_xyz,
                correction_opk,
                images,
                pose_model=self.config["pose_model"],
            )
            ext_param = (
                evaluation_dir / f"{self.config['flight_name']}_ext_param.geojson"
            )
            cameras = oty.FrameCameras(str(self.config["int_param"]), str(ext_param))

            weighted_cost = 0.0
            total_weight = 0.0
            diagnostics = []
            for index, prepared in enumerate(self.references):
                camera = cameras.get(str(prepared.pair.image))
                ortho_path = evaluation_dir / f"candidate_{index:03d}.tif"
                ortho = oty.Ortho(
                    str(prepared.pair.image),
                    str(self.config["dem"]),
                    camera=camera,
                    crs=cameras.crs,
                )
                ortho.process(str(ortho_path), overwrite=True)
                candidate_image, candidate, candidate_valid = _candidate_on_reference(
                    ortho_path, prepared
                )
                common = prepared.valid & candidate_valid
                overlap = np.count_nonzero(common) / np.count_nonzero(prepared.valid)
                if np.count_nonzero(common) < 1000:
                    pair_cost = 2.0
                    correlation = -1.0
                else:
                    correlation = _normalized_correlation(
                        prepared.gradient, candidate, common
                    )
                    pair_cost = 1.0 - correlation
                    pair_cost += self.overlap_penalty * (1.0 - overlap)
                weighted_cost += prepared.pair.weight * pair_cost
                total_weight += prepared.pair.weight
                diagnostics.append((correlation, overlap))
                if self.keep_outputs and self.plot_outputs:
                    plot_validation(
                        prepared,
                        candidate_image,
                        candidate,
                        candidate_valid,
                        evaluation_dir / f"diagnostic_pair_{index + 1:03d}.png",
                        show=self.show_plots,
                    )

            cost = weighted_cost / total_weight
        except Exception as error:
            print(f"evaluation {self.evaluation}: failed: {error}", file=sys.stderr)
            return 1.0e6

        values = " ".join(
            f"{name}={value:.4f}"
            for name, value in zip(PARAMETER_NAMES, parameters)
        )
        pair_values = " ".join(
            f"pair{index + 1}[r={correlation:.3f},o={overlap:.3f}]"
            for index, (correlation, overlap) in enumerate(diagnostics)
        )
        print(
            f"evaluation {self.evaluation}: cost={cost:.6f} {values} {pair_values}",
            flush=True,
        )
        if cost < self.best_cost:
            self.best_cost = cost
            self.best_parameters = parameters.copy()
        return float(cost)


def _load_calibration(
    path: Path, expected_pose_model: str | None = None
) -> np.ndarray:
    with path.open("r", encoding="utf-8") as calibration_file:
        calibration = json.load(calibration_file)
    pose_model = calibration.get("pose_model")
    if pose_model not in imuNcOoGeojson.SUPPORTED_POSE_MODELS:
        raise ValueError(
            f"{path} has no supported pose_model. Re-run the calibration."
        )
    if expected_pose_model is not None and pose_model != expected_pose_model:
        raise ValueError(
            f"{path} uses pose_model '{pose_model}', but the configuration "
            f"requests '{expected_pose_model}'."
        )
    try:
        correction_xyz = np.asarray(calibration["correction_xyz"], dtype=float)
        correction_opk = np.asarray(calibration["correction_opk"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{path} has no valid correction_xyz/correction_opk."
        ) from error
    if correction_xyz.shape != (3,) or correction_opk.shape != (3,):
        raise ValueError("Calibration corrections must each contain three values.")
    return np.concatenate((correction_xyz, correction_opk))


def _write_result(
    config: dict,
    result,
    objective: CalibrationObjective,
    screening_results: list[dict] | None = None,
    selected_screening_start: int | None = None,
    full_optimization_results: list[dict] | None = None,
) -> None:
    parameters = objective.physical_parameters(result.x)
    payload = {
        "pose_model": config["pose_model"],
        "optimized_parameters": config.get(
            "optimize_parameters", list(PARAMETER_NAMES)
        ),
        "correction_xyz": parameters[:3].tolist(),
        "correction_opk": parameters[3:].tolist(),
        "xyz_units": "metres in aircraft frame",
        "opk_units": "degrees",
        "cost": float(result.fun),
        "success": bool(result.success),
        "message": str(result.message),
        "evaluations": int(result.nfev),
        "selected_screening_start": selected_screening_start,
        "screening_results": screening_results or [],
        "full_optimization_results": full_optimization_results or [],
        "image_count": len(config["pairs"]),
        "pairs": [
            {
                "image": str(pair.image.resolve()),
                "reference": str(pair.reference.resolve()),
                "weight": pair.weight,
            }
            for pair in config["pairs"]
        ],
    }
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Calibration written to {output}")
    print(f"correction_xyz = {payload['correction_xyz']}")
    print(f"correction_opk = {payload['correction_opk']}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimize one shared XYZ/OPK IMU-camera correction using one or "
            "more manual orthorectification references."
        )
    )
    parser.add_argument("config", type=Path, help="Calibration YAML file.")
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Override max_iterations from the YAML file.",
    )
    parser.add_argument(
        "--evaluate",
        type=Path,
        help=(
            "Evaluate a calibration JSON without optimizing. Corrected "
            "orthorectifications are kept in work_dir/validation."
        ),
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display validation plots interactively in addition to saving PNGs.",
    )
    return parser


def _initial_steps(config: dict) -> np.ndarray:
    return np.concatenate(
        (
            _vector(config, "initial_step_xyz", [0.5, 0.5, 0.5]),
            _vector(config, "initial_step_opk", [0.5, 0.5, 0.5]),
        )
    )


def _initial_simplex(
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    config: dict,
    active_indices: np.ndarray | None = None,
) -> np.ndarray:
    physical_steps = _initial_steps(config)
    if active_indices is not None:
        physical_steps = physical_steps[active_indices]
        lower = lower[active_indices]
        upper = upper[active_indices]
    if np.any(physical_steps <= 0):
        raise ValueError("Initial XYZ and OPK step sizes must be positive.")
    normalized_steps = physical_steps / (upper - lower)
    simplex = [initial]
    for index, step in enumerate(normalized_steps):
        vertex = initial.copy()
        direction = 1.0 if vertex[index] + step <= 1.0 else -1.0
        vertex[index] += direction * step
        simplex.append(vertex)
    return np.asarray(simplex)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = _read_config(args.config)
        for path_key in ("imu", "int_param", "dem"):
            if not Path(config[path_key]).is_file():
                raise ValueError(f"{path_key} file not found: {config[path_key]}")
        for pair in config["pairs"]:
            if not pair.image.is_file() or not pair.reference.is_file():
                raise ValueError(
                    f"Image pair not found: {pair.image}, {pair.reference}"
                )
            if pair.weight <= 0:
                raise ValueError("Image-pair weights must be positive.")

        max_dimension = int(config.get("max_dimension", 2048))
        if max_dimension < 256:
            raise ValueError("max_dimension must be at least 256.")
        references = [
            _prepare_reference(pair, max_dimension) for pair in config["pairs"]
        ]
        imu = _load_imu(Path(config["imu"]))
        lower, upper = _bounds(config)
        active_indices = _active_parameter_indices(config)
        initial = np.concatenate(
            (
                _vector(config, "initial_xyz", [0, 0, 0]),
                _vector(config, "initial_opk", [0, 0, 0]),
            )
        )
        if np.any(initial < lower) or np.any(initial > upper):
            raise ValueError("Initial parameters must lie inside their bounds.")
        starts = _starting_parameters(
            config,
            initial,
            lower,
            upper,
            active_indices,
        )
        for index, start in enumerate(starts, start=1):
            if np.any(start < lower) or np.any(start > upper):
                raise ValueError(
                    f"Starting correction {index} lies outside configured bounds."
                )

        work_dir = Path(config["work_dir"])
        work_dir.mkdir(parents=True, exist_ok=True)
        objective = CalibrationObjective(
            config,
            imu,
            references,
            lower,
            upper,
            fixed_parameters=initial,
            active_indices=active_indices,
        )
        if args.evaluate:
            parameters = _load_calibration(
                args.evaluate, expected_pose_model=config["pose_model"]
            )
            if np.any(parameters < lower) or np.any(parameters > upper):
                raise ValueError("Calibration parameters lie outside YAML bounds.")
            objective.keep_outputs = True
            objective.show_plots = args.show_plots
            objective.fixed_parameters = parameters
            normalized = (
                (parameters[active_indices] - lower[active_indices])
                / (upper[active_indices] - lower[active_indices])
            )
            cost = objective(normalized)
            if cost >= 1.0e6:
                raise ValueError("Calibration evaluation failed.")
            print(f"Validation cost: {cost:.6f}")
            print(f"Validation rasters: {work_dir / 'validation'}")
            return 0

        max_iterations = (
            args.maxiter
            if args.maxiter is not None
            else int(config.get("max_iterations", 300))
        )
        screening_iterations = int(config.get("screening_iterations", 10))
        if screening_iterations <= 0:
            raise ValueError("screening_iterations must be positive.")
        start_results = []
        screened_candidates = []
        for start_index, start in enumerate(starts, start=1):
            print(
                f"\nScreening start {start_index}/{len(starts)}: "
                + " ".join(
                    f"{name}={value:.4f}"
                    for name, value in zip(PARAMETER_NAMES, start)
                )
            )
            normalized_start = (
                (start[active_indices] - lower[active_indices])
                / (upper[active_indices] - lower[active_indices])
            )
            objective = CalibrationObjective(
                config,
                imu,
                references,
                lower,
                upper,
                fixed_parameters=start,
                active_indices=active_indices,
            )
            result = minimize(
                objective,
                normalized_start,
                method="Nelder-Mead",
                bounds=[(0.0, 1.0)] * len(active_indices),
                options={
                    "initial_simplex": _initial_simplex(
                        normalized_start,
                        lower,
                        upper,
                        config,
                        active_indices=active_indices,
                    ),
                    "maxiter": screening_iterations,
                    "xatol": float(config.get("parameter_tolerance", 1.0e-3)),
                    "fatol": float(config.get("cost_tolerance", 1.0e-4)),
                    "adaptive": True,
                    "disp": True,
                },
            )
            parameters = objective.physical_parameters(result.x)
            start_results.append(
                {
                    "start_index": start_index,
                    "initial_xyz": start[:3].tolist(),
                    "initial_opk": start[3:].tolist(),
                    "correction_xyz": parameters[:3].tolist(),
                    "correction_opk": parameters[3:].tolist(),
                    "cost": float(result.fun),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "evaluations": int(result.nfev),
                }
            )
            print(
                f"Screening start {start_index} result: cost={result.fun:.6f}, "
                f"success={result.success}"
            )
            if objective.best_parameters is not None and np.isfinite(result.fun):
                screened_candidates.append(
                    {
                        "start_index": start_index,
                        "cost": float(result.fun),
                        "parameters": parameters.copy(),
                    }
                )

        if not screened_candidates:
            raise ValueError("Every screening optimization failed.")

        full_optimization_starts = int(config.get("full_optimization_starts", 1))
        if full_optimization_starts <= 0:
            raise ValueError("full_optimization_starts must be positive.")
        finalists = sorted(
            screened_candidates,
            key=lambda candidate: candidate["cost"],
        )[:full_optimization_starts]
        print(
            f"\nSelected {len(finalists)} screening finalists from "
            f"{len(screened_candidates)} successful starts."
        )

        best_full_result = None
        best_full_objective = None
        best_start = None
        full_results = []
        for finalist_index, finalist in enumerate(finalists, start=1):
            full_start = finalist["parameters"]
            normalized_start = (
                (full_start[active_indices] - lower[active_indices])
                / (upper[active_indices] - lower[active_indices])
            )
            objective = CalibrationObjective(
                config,
                imu,
                references,
                lower,
                upper,
                fixed_parameters=full_start,
                active_indices=active_indices,
            )
            print(
                f"\nFull optimization {finalist_index}/{len(finalists)} "
                f"from screening start {finalist['start_index']} "
                f"(screening cost={finalist['cost']:.6f}): "
                + " ".join(
                    f"{name}={value:.4f}"
                    for name, value in zip(PARAMETER_NAMES, full_start)
                )
            )
            result = minimize(
                objective,
                normalized_start,
                method="Nelder-Mead",
                bounds=[(0.0, 1.0)] * len(active_indices),
                options={
                    "initial_simplex": _initial_simplex(
                        normalized_start,
                        lower,
                        upper,
                        config,
                        active_indices=active_indices,
                    ),
                    "maxiter": max_iterations,
                    "xatol": float(config.get("parameter_tolerance", 1.0e-3)),
                    "fatol": float(config.get("cost_tolerance", 1.0e-4)),
                    "adaptive": True,
                    "disp": True,
                },
            )
            parameters = objective.physical_parameters(result.x)
            full_results.append(
                {
                    "rank": finalist_index,
                    "screening_start_index": finalist["start_index"],
                    "screening_cost": finalist["cost"],
                    "correction_xyz": parameters[:3].tolist(),
                    "correction_opk": parameters[3:].tolist(),
                    "cost": float(result.fun),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "evaluations": int(result.nfev),
                }
            )
            print(
                f"Full optimization {finalist_index} result: "
                f"cost={result.fun:.6f}, success={result.success}"
            )
            if (
                objective.best_parameters is not None
                and np.isfinite(result.fun)
                and (
                    best_full_result is None
                    or result.fun < best_full_result.fun
                )
            ):
                best_full_result = result
                best_full_objective = objective
                best_start = finalist["start_index"]

        if best_full_result is None or best_full_objective is None:
            raise ValueError("Every full optimization failed.")

        result = best_full_result
        objective = best_full_objective
        print(
            f"\nSelected full optimization from screening start {best_start} "
            f"with cost {result.fun:.6f}."
        )
        objective.keep_outputs = True
        objective.show_plots = args.show_plots
        final_cost = objective(result.x)
        if final_cost >= 1.0e6:
            raise ValueError("Final calibration validation failed.")
        _write_result(
            config,
            result,
            objective,
            screening_results=start_results,
            selected_screening_start=best_start,
            full_optimization_results=full_results,
        )
        print(
            "Optimized parameters: "
            + ", ".join(PARAMETER_NAMES[index] for index in active_indices)
        )
        print(f"Validation rasters and plots: {work_dir / 'validation'}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
