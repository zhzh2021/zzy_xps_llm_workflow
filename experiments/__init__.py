"""Experiment package shim.

This package exists to satisfy legacy imports and packaging steps that
expect a zzy_llm.experiments namespace. Actual experiment modules (exp_*)
live under zzy_llm/Tools. See zzy_llm/experiments/utils.py for a manifest
export helper that bridges to the Tools-backed discovery.
"""

