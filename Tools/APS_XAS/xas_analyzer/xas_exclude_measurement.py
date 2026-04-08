"""
XAS Data Exclusion Module

Provides functionality to remove specific measurements/outliers from XAS data analysis.
Supports both manual exclusion and automatic exclusion based on quality thresholds.

This module is separate from spectrum_quality_check to maintain clear separation:
- Quality Check = Diagnosis (what's wrong?)
- Data Exclusion = Remediation (what to do about it?)
"""

import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
import pandas as pd


class XASDataExcluder:
    """
    Handles exclusion of bad measurements from XAS datasets.

    Can exclude manually specified measurements or automatically exclude
    based on quality check results.
    """

    def __init__(self):
        """Initialize the excluder with empty exclusion tracking."""
        self.excluded_data: Dict[str, List[int]] = {}
        self.exclusion_log: List[Dict[str, Any]] = []

    def exclude_measurements_manual(
        self,
        data: pd.DataFrame,
        measurements: Union[List[int], np.ndarray],
        experiment: Union[str, int] = "all",
        experiments_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Manually exclude specific measurements from dataset.

        Parameters
        ----------
        data : pd.DataFrame
            XAS data with Experiment and Measurement columns
        measurements : List[int] or np.ndarray
            Measurement numbers to exclude
        experiment : str or int, optional
            Experiment name/index to exclude from. Use "all" for all experiments.
        experiments_list : List[str], optional
            List of all experiment names (for index lookup)

        Returns
        -------
        pd.DataFrame
            Filtered dataset with measurements excluded
        """
        measurements = np.array(measurements)

        # Determine which experiments to process
        if experiment == "all":
            target_experiments = data["Experiment"].unique() if experiments_list is None else experiments_list
        elif isinstance(experiment, str):
            if experiments_list and experiment not in experiments_list:
                raise ValueError(f"Invalid experiment name: {experiment}")
            target_experiments = [experiment]
        elif isinstance(experiment, int):
            if experiments_list is None:
                raise ValueError("experiments_list required when using experiment index")
            target_experiments = [experiments_list[experiment]]
        else:
            raise ValueError("Invalid experiment specification")

        filtered_data = data.copy()

        for exp in target_experiments:
            # Initialize exclusion tracking for this experiment
            if exp not in self.excluded_data:
                self.excluded_data[exp] = []

            # Find measurements to exclude for this experiment
            exp_measurements = measurements
            if experiment != "all":
                # If specific experiment, exclude these measurements
                pass  # measurements already set
            else:
                # If "all", exclude these measurements from all experiments
                pass  # measurements already set

            # Create exclusion mask
            exclude_mask = (
                (filtered_data["Experiment"] == exp) &
                (filtered_data["Measurement"].isin(exp_measurements))
            )

            # Log exclusions
            excluded_count = exclude_mask.sum()
            if excluded_count > 0:
                self.excluded_data[exp].extend(exp_measurements.tolist())
                self.exclusion_log.append({
                    "experiment": exp,
                    "measurements": exp_measurements.tolist(),
                    "count": excluded_count,
                    "reason": "manual_exclusion",
                    "timestamp": pd.Timestamp.now()
                })

            # Apply exclusion
            filtered_data = filtered_data[~exclude_mask]

        return filtered_data

    def exclude_measurements_auto(
        self,
        data: pd.DataFrame,
        quality_results: Dict[str, Dict[str, Any]],
        confidence_threshold: float = 0.5,
        exclude_invalid: bool = True,
        exclude_low_confidence: bool = False
    ) -> pd.DataFrame:
        """
        Automatically exclude measurements based on quality check results.

        Parameters
        ----------
        data : pd.DataFrame
            XAS data with Experiment and Measurement columns
        quality_results : dict
            Quality check results keyed by experiment-measurement
        confidence_threshold : float, optional
            Minimum confidence to keep (default: 0.5)
        exclude_invalid : bool, optional
            Whether to exclude 'invalid' classifications (default: True)
        exclude_low_confidence : bool, optional
            Whether to exclude low confidence spectra (default: False)

        Returns
        -------
        pd.DataFrame
            Filtered dataset with bad measurements excluded
        """
        filtered_data = data.copy()
        exclusions_made = []

        for key, quality in quality_results.items():
            # Parse key to get experiment and measurement
            # Assuming key format like "experiment_001_measurement_005"
            if "_" not in key:
                continue

            parts = key.split("_measurement_")
            if len(parts) != 2:
                continue

            experiment = parts[0].replace("experiment_", "")
            measurement = int(parts[1])

            exclude = False
            reason = ""

            # Check classification
            if exclude_invalid and quality.get("classification") == "invalid":
                exclude = True
                reason = f"invalid_classification_{quality.get('classification')}"

            # Check confidence
            elif exclude_low_confidence and quality.get("confidence", 1.0) < confidence_threshold:
                exclude = True
                reason = f"low_confidence_{quality.get('confidence', 0):.2f}"

            if exclude:
                # Apply exclusion
                exclude_mask = (
                    (filtered_data["Experiment"] == experiment) &
                    (filtered_data["Measurement"] == measurement)
                )

                if exclude_mask.sum() > 0:
                    # Track exclusion
                    if experiment not in self.excluded_data:
                        self.excluded_data[experiment] = []
                    self.excluded_data[experiment].append(measurement)

                    self.exclusion_log.append({
                        "experiment": experiment,
                        "measurements": [measurement],
                        "count": 1,
                        "reason": reason,
                        "quality_flags": quality.get("flags", []),
                        "confidence": quality.get("confidence", 0),
                        "timestamp": pd.Timestamp.now()
                    })

                    # Remove the data
                    filtered_data = filtered_data[~exclude_mask]
                    exclusions_made.append(f"{experiment}_measurement_{measurement}")

        if exclusions_made:
            print(f"Auto-excluded {len(exclusions_made)} measurements: {exclusions_made}")

        return filtered_data

    def get_exclusion_summary(self) -> Dict[str, Any]:
        """
        Get summary of all exclusions made.

        Returns
        -------
        dict
            Summary statistics of exclusions
        """
        total_exclusions = sum(len(measurements) for measurements in self.excluded_data.values())

        return {
            "total_exclusions": total_exclusions,
            "experiments_affected": len(self.excluded_data),
            "exclusion_log": self.exclusion_log,
            "excluded_by_experiment": self.excluded_data
        }


# Convenience functions for integration
def exclude_bad_measurements_manual(
    data: pd.DataFrame,
    measurements: Union[List[int], np.ndarray],
    experiment: Union[str, int] = "all",
    experiments_list: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function for manual measurement exclusion.
    """
    excluder = XASDataExcluder()
    return excluder.exclude_measurements_manual(data, measurements, experiment, experiments_list)


def exclude_bad_measurements_auto(
    data: pd.DataFrame,
    quality_results: Dict[str, Dict[str, Any]],
    confidence_threshold: float = 0.5,
    exclude_invalid: bool = True,
    exclude_low_confidence: bool = False
) -> pd.DataFrame:
    """
    Convenience function for automatic measurement exclusion based on quality.
    """
    excluder = XASDataExcluder()
    return excluder.exclude_measurements_auto(
        data, quality_results, confidence_threshold, exclude_invalid, exclude_low_confidence
    )