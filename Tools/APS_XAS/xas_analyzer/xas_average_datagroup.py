"""
XAS Data Averaging Module

Provides functionality to average multiple XAS measurements for operando experiments.
This module is currently DISABLED and not integrated into the main workflow.

When enabled, it will support:
1. Simple averaging of all measurements within each experiment
2. Periodic averaging with configurable grouping

Note: This module is experimental and should be thoroughly tested before production use.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List
from tqdm import tqdm

# DISABLED: Set to False to enable averaging functionality
AVERAGING_ENABLED = False

class XASAveragingMixin:
    """
    Mixin class providing XAS data averaging functionality.

    This mixin can be added to XAS data processing classes to enable
    averaging of multiple measurements from operando experiments.
    """

    def average_measurements(
        self,
        measurements_to_average: Union[str, List[int], np.ndarray, range] = "all",
        standards: bool = False,
        averaging_enabled: bool = AVERAGING_ENABLED
    ) -> None:
        """
        Average data points for each experiment.

        Parameters
        ----------
        measurements_to_average : Union[str, List[int], np.ndarray, range], optional
            Measurements to average. Use "all" for all measurements. Defaults to "all".
        standards : bool, optional
            Whether to average standards data. Defaults to False.
        averaging_enabled : bool, optional
            Whether averaging is enabled. Defaults to AVERAGING_ENABLED flag.

        Returns
        -------
        None
            Modifies data in-place if averaging is enabled.
        """
        if not averaging_enabled:
            print("Data averaging is currently disabled. Set AVERAGING_ENABLED=True to enable.")
            return

        if not hasattr(self, 'data') or not hasattr(self, 'experiments'):
            raise AttributeError("XASAveragingMixin requires 'data' and 'experiments' attributes")

        dataframe = self.standards if standards else self.data
        experiments = self.standard_experiments if standards else self.experiments

        print(f"Averaging {'standards' if standards else 'data'} across {len(experiments)} experiments...")

        avg_measurements = []
        for experiment in tqdm(experiments, desc="Averaging data", leave=False):
            experiment_filter = dataframe["Experiment"] == experiment

            if measurements_to_average == "all":
                measurements = dataframe["Measurement"][experiment_filter].unique()
            else:
                measurements = np.array(measurements_to_average)

            if len(measurements) == 0:
                continue

            # Initialize accumulators
            first_measurement = measurements[0]
            first_filter = (dataframe["Experiment"] == experiment) & (
                dataframe["Measurement"] == first_measurement
            )
            df_avg = dataframe[first_filter].copy()

            # Initialize arrays for averaging
            energy_shape = len(df_avg)
            i0_avg = np.zeros(energy_shape, dtype=np.float64)
            i1_avg = np.zeros(energy_shape, dtype=np.float64)
            mu_avg = np.zeros(energy_shape, dtype=np.float64)

            # Accumulate data from all measurements
            n_measurements = 0
            for measurement in measurements:
                measurement_filter = (dataframe["Experiment"] == experiment) & (
                    dataframe["Measurement"] == measurement
                )

                if measurement_filter.sum() == 0:  # Skip if measurement doesn't exist
                    continue

                i0_avg += dataframe["I0"][measurement_filter].to_numpy()
                i1_avg += dataframe["I1"][measurement_filter].to_numpy()
                mu_avg += dataframe["mu"][measurement_filter].to_numpy()
                n_measurements += 1

            if n_measurements == 0:
                continue

            # Calculate averages
            df_avg["I0"] = i0_avg / n_measurements
            df_avg["I1"] = i1_avg / n_measurements
            df_avg["mu"] = mu_avg / n_measurements

            # Average temperature data if available
            temp_filter = (dataframe["Experiment"] == experiment) & (
                dataframe["Measurement"].isin(measurements)
            )
            if "Temperature" in dataframe.columns:
                df_avg["Temperature"] = dataframe["Temperature"][temp_filter].mean()
                df_avg["Temperature (std)"] = dataframe["Temperature"][temp_filter].std()

            avg_measurements.append(df_avg)

        if avg_measurements:
            if standards:
                self.standards = pd.concat(avg_measurements, ignore_index=True)
            else:
                self.data = pd.concat(avg_measurements, ignore_index=True)

        print(f"Averaging complete. Processed {len(avg_measurements)} experiments.")

    def average_measurements_periodic(
        self,
        period: Optional[int] = None,
        n_periods: Optional[int] = None,
        standards: bool = False,
        averaging_enabled: bool = AVERAGING_ENABLED
    ) -> None:
        """
        Average data points for each experiment using periodic grouping of measurements.

        Parameters
        ----------
        period : int, optional
            Number of measurements to group for averaging. Determines number of periods automatically.
        n_periods : int, optional
            Number of periods to group for averaging. Determines measurements per period automatically.
        standards : bool, optional
            Whether to average standards data. Defaults to False.
        averaging_enabled : bool, optional
            Whether averaging is enabled. Defaults to AVERAGING_ENABLED flag.

        Raises
        ------
        ValueError
            If both period and n_periods are provided, or neither is provided.
        """
        if not averaging_enabled:
            print("Periodic data averaging is currently disabled. Set AVERAGING_ENABLED=True to enable.")
            return

        if not hasattr(self, 'data') or not hasattr(self, 'experiments'):
            raise AttributeError("XASAveragingMixin requires 'data' and 'experiments' attributes")

        # Validate input parameters
        if (period is not None and n_periods is not None) or (period is None and n_periods is None):
            raise ValueError("Exactly one of 'period' or 'n_periods' must be specified")

        dataframe = self.standards if standards else self.data
        experiments = self.standard_experiments if standards else self.experiments

        print(f"Periodic averaging {'standards' if standards else 'data'} across {len(experiments)} experiments...")

        avg_measurements = []

        for experiment in tqdm(experiments, desc="Periodic averaging", leave=False):
            experiment_filter = dataframe["Experiment"] == experiment
            measurements = dataframe["Measurement"][experiment_filter].unique()
            n_total_measurements = len(measurements)

            if n_total_measurements == 0:
                continue

            # Calculate grouping parameters
            if period is not None:
                n_measurements_per_period = period
                n_periods_calc = int(np.ceil(n_total_measurements / period))
            else:  # n_periods is not None
                n_periods_calc = n_periods
                n_measurements_per_period = n_total_measurements // n_periods

            # Group measurements into periods
            for period_idx in range(n_periods_calc):
                start_idx = period_idx * n_measurements_per_period
                end_idx = min(start_idx + n_measurements_per_period, n_total_measurements)
                period_measurements = measurements[start_idx:end_idx]

                if len(period_measurements) == 0:
                    continue

                # Initialize with first measurement in period
                first_measurement = period_measurements[0]
                first_filter = (dataframe["Experiment"] == experiment) & (
                    dataframe["Measurement"] == first_measurement
                )
                df_avg = dataframe[first_filter].copy()

                # Initialize arrays for averaging
                energy_shape = len(df_avg)
                energy_avg = np.zeros(energy_shape, dtype=np.float64)
                i0_avg = np.zeros(energy_shape, dtype=np.float64)
                i1_avg = np.zeros(energy_shape, dtype=np.float64)
                mu_avg = np.zeros(energy_shape, dtype=np.float64)

                # Accumulate data from measurements in this period
                n_measurements = 0
                for measurement in period_measurements:
                    measurement_filter = (dataframe["Experiment"] == experiment) & (
                        dataframe["Measurement"] == measurement
                    )

                    if measurement_filter.sum() == 0:
                        continue

                    energy_avg += dataframe["Energy"][measurement_filter].to_numpy()
                    i0_avg += dataframe["I0"][measurement_filter].to_numpy()
                    i1_avg += dataframe["I1"][measurement_filter].to_numpy()
                    mu_avg += dataframe["mu"][measurement_filter].to_numpy()
                    n_measurements += 1

                if n_measurements == 0:
                    continue

                # Calculate averages
                df_avg["Energy"] = energy_avg / n_measurements
                df_avg["I0"] = i0_avg / n_measurements
                df_avg["I1"] = i1_avg / n_measurements
                df_avg["mu"] = mu_avg / n_measurements
                df_avg["Measurement"] = period_idx + 1  # Renumber measurements

                # Average temperature data if available
                temp_filter = (dataframe["Experiment"] == experiment) & (
                    dataframe["Measurement"].isin(period_measurements)
                )
                if "Temperature" in dataframe.columns:
                    df_avg["Temperature"] = dataframe["Temperature"][temp_filter].mean()
                    df_avg["Temperature (std)"] = dataframe["Temperature"][temp_filter].std()

                avg_measurements.append(df_avg)

        if avg_measurements:
            if standards:
                self.standards = pd.concat(avg_measurements, ignore_index=True)
            else:
                self.data = pd.concat(avg_measurements, ignore_index=True)

        print(f"Periodic averaging complete. Created {len(avg_measurements)} averaged periods.")



def average_spectra(energy_list: List[np.ndarray], mu_list: List[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    # Average multiple spectra onto a common energy grid (uses first spectrum as grid).
    if not energy_list or not mu_list or len(energy_list) != len(mu_list):
        raise ValueError("energy_list and mu_list must be non-empty and the same length")

    base_energy = energy_list[0]
    if base_energy is None:
        raise ValueError("Base energy grid is None")

    acc = np.zeros_like(base_energy, dtype=np.float64)
    n = 0
    for energy, mu in zip(energy_list, mu_list):
        if energy is None or mu is None:
            continue
        if len(energy) != len(base_energy) or not np.allclose(energy, base_energy):
            mu_interp = np.interp(base_energy, energy, mu)
            acc += mu_interp
        else:
            acc += mu
        n += 1

    if n == 0:
        raise ValueError("No valid spectra to average")

    return base_energy, acc / n


# Example usage (when enabled):
"""
# To enable averaging, set AVERAGING_ENABLED = True at the top of this file

# Then in your XAS processor class:
class XASProcessorWithAveraging(XASAveragingMixin, XASProcessor):
    pass

# Usage:
processor = XASProcessorWithAveraging()
processor.average_measurements(measurements_to_average="all")
# or
processor.average_measurements_periodic(period=5)  # Average every 5 measurements
"""
    