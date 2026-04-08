import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lmfit.models import PseudoVoigtModel, LinearModel


========================================================================


1. LOAD YOUR REAL DATA


========================================================================



# Resolve data path relative to this file so the script works on any machine.
# Override by setting the ZZY_LLM_HOME environment variable to your project root.
import os as _os
_project_root = Path(
    _os.environ.get("ZZY_LLM_HOME") or Path(__file__).resolve().parents[3] / "project_root"
)
data_path = str(_project_root / "01_converted_csv" / "Li1s_5scans_lowSNR.csv")




data = np.loadtxt(data_path, delimiter=',')
x = data[:, 0]      # Energy values (binding energy in eV)
y_noisy = data[:, 1]  # Intensity values (counts)




print(f"Loaded data: {len(x)} points")
print(f"Energy range: {x.min():.2f} to {x.max():.2f} eV")
print(f"Intensity range: {y_noisy.min():.1f} to {y_noisy.max():.1f} counts\n")



========================================================================


2. DEFINE THE MODEL FOR FITTING


========================================================================


We assume there are two peaks and a linear background.


Adjust the number of peaks if your Li 1s spectrum has more components.



peak1_fit = PseudoVoigtModel(prefix='p1_')
peak2_fit = PseudoVoigtModel(prefix='p2_')
background_fit = LinearModel()




model_fit = peak1_fit + peak2_fit + background_fit



========================================================================


3. SET UP UNCONSTRAINED PARAMETERS


========================================================================


This is the critical step. We provide vague initial guesses and


very wide bounds, allowing the fitter to find unphysical solutions.



params_fit = model_fit.make_params()



Peak 1: Unconstrained parameters



params_fit['p1_center'].set(value=55, min=50, max=65)      # Can be anywhere in the scan range
params_fit['p1_sigma'].set(value=1.0, min=0.1, max=10.0)   # Can be extremely wide (unphysical)
params_fit['p1_amplitude'].set(value=100, min=0)           # Positive amplitude



Peak 2: Unconstrained parameters



params_fit['p2_center'].set(value=58, min=50, max=65)      # Can be anywhere in the scan range
params_fit['p2_sigma'].set(value=1.0, min=0.1, max=10.0)   # Can be extremely wide (unphysical)
params_fit['p2_amplitude'].set(value=50, min=0)            # Positive amplitude



Linear background: Unconstrained



params_fit['slope'].set(value=0, min=-100, max=100)
params_fit['intercept'].set(value=np.mean(y_noisy), min=0)



========================================================================


4. PERFORM THE UNCONSTRAINED FIT


========================================================================



print("Performing unconstrained fit...\n")
result = model_fit.fit(y_noisy, params_fit, x=x)



========================================================================


5. ANALYZE AND REPORT THE RESULTS


========================================================================



print("=" * 70)
print("UNCONSTRAINED FIT REPORT")
print("=" * 70)
print(result.fit_report())
print("=" * 70)



Extract key parameters to highlight unphysical results



p1_center = result.params['p1_center'].value
p1_sigma = result.params['p1_sigma'].value
p1_fwhm = p1_sigma * 2.355  # Convert sigma to FWHM




p2_center = result.params['p2_center'].value
p2_sigma = result.params['p2_sigma'].value
p2_fwhm = p2_sigma * 2.355




print("\n" + "=" * 70)
print("KEY RESULTS (Check for Unphysical Values)")
print("=" * 70)
print(f"Peak 1: Center = {p1_center:.2f} eV, FWHM = {p1_fwhm:.2f} eV")
print(f"Peak 2: Center = {p2_center:.2f} eV, FWHM = {p2_fwhm:.2f} eV")
print(f"Reduced Chi-Square: {result.redchi:.4f}")
print("=" * 70)



========================================================================


6. PLOT THE RESULTS TO VISUALIZE THE FAILURE


========================================================================



plt.figure(figsize=(12, 7))



Plot the data



plt.plot(x, y_noisy, 'o', label='Low-SNR Data (5 scans)',
         markersize=5, color='gray', alpha=0.6)



Plot the best fit



plt.plot(x, result.best_fit, '-', label='Best Unconstrained Fit',
         linewidth=2.5, color='red')



Plot the individual components the model "found"



components = result.eval_components(x=x)
plt.plot(x, components['p1_'], '--', label=f'Found Peak 1 (FWHM={p1_fwhm:.2f} eV)',
         linewidth=2, color='blue')
plt.plot(x, components['p2_'], '--', label=f'Found Peak 2 (FWHM={p2_fwhm:.2f} eV)',
         linewidth=2, color='green')
plt.plot(x, components['linear'], ':', label='Found Background',
         linewidth=2, color='orange')



Formatting



plt.title('Failure of Unconstrained Fit on Low-SNR Li 1s Data', fontsize=14, fontweight='bold')
plt.xlabel('Binding Energy (eV)', fontsize=12)
plt.ylabel('Intensity (Counts)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()



Save the figure (optional)


plt.savefig('unconstrained_fit_failure.png', dpi=300)



plt.show()


