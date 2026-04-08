"""
Utilities for XAS deep learning stage (stage 06).
Uses shared config and project_root paths from the APS_XAS toolchain.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml


def get_tools_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def get_project_root() -> Path:
    # .../Tools/APS_XAS/xas_deep_learning -> .../Tools/APS_XAS -> .../Tools -> .../zzy_llm
    tools_dir = get_tools_dir()
    zzy_llm_dir = tools_dir.parent.parent
    return zzy_llm_dir / "project_root"


def get_config_path(config_path: Optional[str | Path] = None) -> Path:
    if config_path is None:
        return get_tools_dir() / "xas_config" / "xas_ml_settings.yaml"
    return Path(config_path)


def load_deep_learning_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    # Use ConfigLoader if available
    try:
        from xas_ml_modules.config_utils import ConfigLoader
        loader = ConfigLoader(get_config_path(config_path))
        cfg = loader.get_all()
    except Exception:
        cfg_path = get_config_path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return cfg.get("deep_learning", {})


def load_plot_settings() -> Dict[str, Any]:
    cfg_path = get_tools_dir() / "xas_config" / "xas_plot_settings.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def apply_plot_style() -> Dict[str, Any]:
    plot_cfg = load_plot_settings()
    if not plot_cfg:
        return {}

    use_pub = bool(plot_cfg.get("use_publication_quality", True))
    base = plot_cfg.get("publication" if use_pub else "plot_settings", {})
    fonts = base.get("fonts", {})
    dpi = base.get("dpi", 300)

    try:
        import matplotlib as mpl
        mpl.rcParams["figure.dpi"] = dpi
        mpl.rcParams["savefig.dpi"] = dpi
        mpl.rcParams["axes.titlesize"] = fonts.get("title_size", 18)
        mpl.rcParams["axes.labelsize"] = fonts.get("axis_label_size", 16)
        mpl.rcParams["xtick.labelsize"] = fonts.get("tick_label_size", 14)
        mpl.rcParams["ytick.labelsize"] = fonts.get("tick_label_size", 14)
        mpl.rcParams["legend.fontsize"] = fonts.get("legend_size", 16)
    except Exception:
        pass

    return {
        "dpi": dpi,
        "fonts": fonts,
    }


def get_output_dirs(base_dir: Optional[Path] = None) -> Dict[str, Path]:
    if base_dir is None:
        base_dir = get_project_root() / "xas_results" / "06_deep_learning"
    base_dir = Path(base_dir)
    dirs = {
        "base": base_dir,
        "datasets": base_dir / "datasets",
        "models": base_dir / "models",
        "plots": base_dir / "plots",
        "reports": base_dir / "reports",
        "logs": base_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "deep_learning.log"
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_config_snapshot(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
