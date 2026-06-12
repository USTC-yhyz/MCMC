# Endmember Mixing Model for Cosmochemical Samples

This repository contains Python scripts for endmember unmixing of cosmochemical samples using isotopic (ε⁵⁴Cr, ε⁵⁰Ti) and elemental (Mg, Cr, Ti, Al, Ni) data. The approach combines Bayesian MCMC to infer endmember compositions and mixing fractions, and a deterministic calculation to compute fractions from posterior mean endmembers.

## Contents

- **`MCMC.py`** – Bayesian MCMC sampling (emcee) to estimate endmember compositions and compare models (2, 3, 4 endmembers) using WAIC. Includes NC isotope line priors, CAI priors, posterior predictive checks, and corner plots.
- **`MCMCcal.py`** – Compute mixing fractions for each sample using the posterior mean endmember compositions (or any user-defined endmembers). Outputs a CSV with fitting errors and percentage contributions.

## Requirements

Both scripts require Python 3.7+ and the following packages:

```
numpy
pandas
scipy
emcee
corner
matplotlib
tqdm
multiprocessing   # (standard library)
```

Install the dependencies using:

```bash
pip install numpy pandas scipy emcee corner matplotlib tqdm
```

## Input Data Format

The scripts expect an Excel file (`.xlsx`) with the following columns (adjust the path in the script):

| Column | Description |
|--------|-------------|
| `Sample` | Sample identifier |
| `Level1` or `Type` | Sample classification (e.g., chondrite group). A `Type` column will be used if present; otherwise `Level1` is used. |
| `ε54Cr` | Measured ε⁵⁴Cr value |
| `ε50Ti` | Measured ε⁵⁰Ti value |
| `Mg`, `Cr`, `Ti`, `Al`, `Ni` | Elemental concentrations (in any consistent units) |
| `ε54Cr2SE`, `ε50Ti2SE` (optional) | 2 standard error of the isotopic measurements. If missing, a default error of 0.10 (2SE) is used. |

The `MCMC.py` script also uses a built‑in table of NC chondrite data for the NC isotope line prior. You can modify this table inside the script if needed.

## Usage

### 1. Bayesian MCMC analysis (`MCMC.py`)

This script runs four endmember models (two, three, and two four‑endmember configurations) and compares them via WAIC. It produces posterior summaries, WAIC vectors, and diagnostic plots.

1. Place your data file (named `data4.xlsx` by default) in the same directory.
2. Adjust `FILE_PATH` and other global parameters if necessary:
   ```python
   FILE_PATH = 'data.xlsx'
   N_WALKERS = 200
   N_STEPS_BURN = 2000
   N_STEPS_SAMPLE = 3000
   THIN = 5
   ```
3. Run the script:
   ```bash
   python MCMC.py
   ```
4. Outputs:
   - `MCMC_Full_Comparison_weighted_<model>.h5` – MCMC backend files (can be reused to resume chains)
   - `MCMC_Full_Comparison_weighted_<model>_posterior_summary_extended.csv` – Posterior summary statistics (min, mean, max, std, 5%, 95%)
   - `MCMC_Full_Comparison_weighted_<model>_waic_vector.csv` – Per‑observation WAIC contributions
   - `MCMC_Full_Comparison_weighted_WAIC_comparison.pdf` – Bar chart comparing model WAIC values
   - `MCMC_Full_Comparison_weighted_<best_model>_posterior_predictive.pdf` – Predicted vs. observed ε⁵⁴Cr and ε⁵⁰Ti for the best model
   - `MCMC_Full_Comparison_weighted_<best_model>_corner.pdf` – Corner plot of key parameters for the best model
   - `MCMC_Full_Comparison_weighted_<best_model>_postpred.npz` – Posterior predictive samples (compressed)

   The best model is selected based on the lowest WAIC.

### 2. Computing fraction contributions (`MCMCcal.py`)

Use the posterior mean endmember compositions (or any other set) to calculate the mixing fractions for each sample.

1. Place your data file (named `data.xlsx` by default).
2. Verify or modify the endmember compositions in the script. By default, the script uses the posterior means from “this study”. You can add additional endmember sets from other studies by uncommenting and defining similar arrays.
3. Run:
   ```bash
   python MCMCcal.py
   ```
4. Output:
   - `endmember_fractions.csv` – CSV file containing for each sample:
     - `Sample`, `Type`, `Source` (endmember set name), `Fitting Error` (weighted sum of squared residuals), and the percentage contributions of each endmember (e.g., `NCAvg`, `CAIUltra`, `CAINorm`, `CI`).

## Model Details

Both scripts use the same forward model and optimization:

- **Endmembers** are parameterized by the seven quantities: ε⁵⁴Cr, ε⁵⁰Ti, and the concentrations of Mg, Cr, Ti, Al, Ni.
- **Mixing** is linear in the elemental concentrations; isotopic ratios are computed from mass balances.
- **Fraction solving** is performed via constrained optimization (SLSQP) minimizing a weighted sum of squared residuals that includes isotopic and elemental ratio constraints. The objective function is identical in both scripts to ensure consistency.

### MCMC Priors

In `MCMC.py`, several priors are applied:
- **NC isotope line prior**: The ε⁵⁰Ti of any NC endmember is assumed to follow a linear relation ε⁵⁰Ti = a·ε⁵⁴Cr + b, fitted from NC chondrite literature data, with Gaussian scatter.
- **CAI prior**: The ε⁵⁰Ti of certain CAI endmembers is constrained to Gaussian priors centered on literature values (e.g., 9.0 ± 1.0 for CAI_Avg).
- **Parameter bounds**: Hard limits are set for all endmember parameters based on reasonable ranges.

The four‑endmember model `four_nc` splits the NC component into two endmembers (NC_L and NC_H) that are required to have similar isotopic compositions (additional Gaussian constraint with σ = 0.1).

## Configuration for Other Studies

If you want to use endmembers from other publications, add them as new arrays in the `sources` dictionary of `MCMCcal.py`. For `MCMC.py`, you can modify the model configurations (`TWO_BOUNDS`, `THREE_BOUNDS`, etc.) and the NC line fitting data to adapt to your own prior knowledge.

## License

This project is provided under the MIT License. Feel free to use, modify, and distribute.

## Citation

If you use these scripts in your research, please cite the corresponding study (to be added by the authors) and the used libraries (emcee, corner, etc.).

---

For any questions or issues, please open an issue in the repository.
