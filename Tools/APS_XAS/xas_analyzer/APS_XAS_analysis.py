
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from larch import Group
from larch.xafs import (
    pre_edge,
    autobk,
    xftf
)
from larch.io import read_ascii, read_xdi
# pip install xraylarch numpy pandas h5py matplotlib

# ----------------------------
# Directory setup
# ----------------------------
data_dir = Path("aps_xas_raw/")   #  directory with XDI/.dat/.txt
#once user drop the data in a directory, automaticcally generate  output directory
output_dir = Path("aps_xas_out/")
output_dir.mkdir(exist_ok=True)

# summary table
summary = []

# ----------------------------
# File import helper
# ----------------------------
def load_xas(file_path):
    """
    Load XAS data from file.
    Supports XDI and generic ASCII with energy + intensity columns.
    """
    ext = file_path.suffix.lower()

    if ext == '.xdi':
        data = read_xdi(str(file_path))
        energy = data.energy
        mu = data.mutrans   # transmission μ(E)
    else:
        # try generic ASCII: first 2–3 columns: energy, i0, it
        raw = np.loadtxt(file_path)
        energy = raw[:, 0]
        i0, it = raw[:, 1], raw[:, 2]
        mu = np.log(i0 / it)

    return energy, mu

# ----------------------------
# Loop files
# ----------------------------
for file in sorted(data_dir.iterdir()):
    if not file.is_file():
        continue

    print(f"Processing {file.name} ...")

    # load
    energy, mu = load_xas(file)

    # prepare group
    g = Group(energy=energy, mu=mu)

    # XANES processing
    pre_edge(g, pre1=-150, pre2=-30, norm1=150, norm2=800)

    # EXAFS background removal
    autobk(g, rbkg=1.0, kmin=0, kmax=14)

    # Fourier transform EXAFS
    xftf(g, kmin=2, kmax=12)

    # save processed variables
    prefix = output_dir / file.stem

    # normalized XANES
    np.savetxt(
        prefix.with_suffix("_xanes.csv"),
        np.column_stack([g.energy, g.norm]),
        delimiter=",",
        header="energy_eV,mu_norm",
        comments=""
    )

    # χ(k)
    np.savetxt(
        prefix.with_suffix("_chik.csv"),
        np.column_stack([g.k, g.chi]),
        delimiter=",",
        header="k,chi",
        comments=""
    )

    # χ(R)
    np.savetxt(
        prefix.with_suffix("_chiR.csv"),
        np.column_stack([g.r, np.abs(g.chir)]),
        delimiter=",",
        header="R,abs_chiR",
        comments=""
    )

    # Extract key features for ML
    features = {
        "sample": file.stem,
        "e0": float(g.e0),
        "edge_step": float(g.edge_step),
        "white_line_peak": float(np.max(g.norm)),
        "FT_peak_R": float(g.r[np.argmax(np.abs(g.chir))]),
        "FT_peak_amp": float(np.max(np.abs(g.chir)))
    }
    summary.append(features)

# write summary table
df = pd.DataFrame(summary)
df.to_csv(output_dir / "aps_xas_summary.csv", index=False)
print("Batch processing complete. Summary saved to aps_xas_summary.csv")
