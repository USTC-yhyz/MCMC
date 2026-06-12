import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ======================== Configuration ========================
FILE_PATH = 'data.xlsx'
OUTPUT_FILE = 'endmember_fractions.csv'

# Names of the four endmembers
EM_NAMES = ['NCAvg', 'CAIUltra', 'CAINorm', 'CI']
N_EMS = 4

# ======================== Endmember definitions (posterior means from this study) ========================
# Each endmember in order: [ε54Cr, ε50Ti, Mg, Cr, Ti, Al, Ni]
em_this_study = np.array([
    [-2.556339012, -4.401194468, 1.304367927, 1.021545158, 0.686805239, 0.947062562, 0.189590503],  # NCAvg
    [ 5.191623036,  9.433936608, 0.553925245, 0.423797533, 9.684130127, 7.603160141, 0.662339942],  # CAIUltra
    [ 5.767272076,  3.396739004, 1.546373269, 1.24393382, 12.55868301, 2.978387564, 0.407383174],   # CAINorm
    [ 1.654005135,  1.833357532, 1.047845132, 1.020182245, 0.895608778, 0.933664106, 1.086728211]    # CI
])

# To add endmembers from other studies, define similar arrays and include them in the sources dictionary
# em_schrader = np.array([...])   # Schrader et al. 2025
# em_zhu = np.array([...])        # Zhu et al. 2025

sources = {
    'this study': em_this_study,
    # 'Schrader et al. 2025': em_schrader,
    # 'Zhu et al. 2025': em_zhu,
}

# ======================== Data loading ========================
def load_data(filepath):
    df = pd.read_excel(filepath)
    # Filter valid samples (adjust filtering criteria as needed)
    df = df[(df['Level1'].astype(str).str.startswith('C')) & (df['Level1'] != 'Clast')].copy()

    # Check for a 'Type' column; if missing, use 'Level1' as the type
    if 'Type' not in df.columns:
        if 'Level1' in df.columns:
            df['Type'] = df['Level1']
        else:
            df['Type'] = 'Unknown'

    cols = ['ε54Cr', 'ε50Ti', 'Mg', 'Cr', 'Ti', 'Al', 'Ni']
    df_calc = df.dropna(subset=cols).copy()
    obs_vals = df_calc[cols].values

    # Read measurement errors
    try:
        e54_2se = df_calc['ε54Cr2SE'].values
        e50_2se = df_calc['ε50Ti2SE'].values
        print("✅ Using actual measurement errors")
    except KeyError:
        e54_2se = np.full(len(df_calc), 0.10)
        e50_2se = np.full(len(df_calc), 0.10)
    sigma_e54 = np.abs(e54_2se) / 2.0
    sigma_e50 = np.abs(e50_2se) / 2.0

    rel_error = 0.05
    sigma_elem = np.abs(obs_vals[:, 2:]) * rel_error

    obs_sigmas = np.zeros_like(obs_vals)
    obs_sigmas[:, 0] = sigma_e54
    obs_sigmas[:, 1] = sigma_e50
    obs_sigmas[:, 2:] = sigma_elem
    obs_sigmas = np.maximum(obs_sigmas, 1e-9)

    print(f"✅ Loaded {len(df_calc)} samples")
    return obs_vals, obs_sigmas, df_calc[['Sample', 'Type']].reset_index(drop=True)

obs_vals, obs_sigmas, meta_df = load_data(FILE_PATH)
n_samples = len(obs_vals)

# ======================== Unmixing function (exactly matching the MCMC model) ========================
def predict_mixture(f, ems):
    mix_Mg = np.dot(f, ems[:, 2])
    mix_Cr = np.dot(f, ems[:, 3])
    mix_Ti = np.dot(f, ems[:, 4])
    mix_Al = np.dot(f, ems[:, 5])
    mix_Ni = np.dot(f, ems[:, 6])
    mix_e54 = np.dot(f, ems[:, 3] * ems[:, 0]) / mix_Cr
    mix_e50 = np.dot(f, ems[:, 4] * ems[:, 1]) / mix_Ti
    return np.array([mix_e54, mix_e50, mix_Cr, mix_Ti, mix_Mg, mix_Al, mix_Ni])

def solve_fractions(obs, ems, sigma):
    sigma = np.asarray(sigma)
    obs_e54, obs_e50 = obs[0], obs[1]
    R_CrTi_obs = obs[3] / obs[4]
    R_MgCr_obs = obs[2] / obs[3]
    R_AlMg_obs = obs[5] / obs[2]
    R_NiMg_obs = obs[6] / obs[2]

    rel_Cr = sigma[3] / obs[3]; rel_Ti = sigma[4] / obs[4]
    sigma_CrTi = R_CrTi_obs * np.sqrt(rel_Cr**2 + rel_Ti**2)
    rel_Mg = sigma[2] / obs[2]
    sigma_MgCr = R_MgCr_obs * np.sqrt(rel_Mg**2 + rel_Cr**2)
    rel_Al = sigma[5] / obs[5]
    sigma_AlMg = R_AlMg_obs * np.sqrt(rel_Al**2 + rel_Mg**2)
    rel_Ni = sigma[6] / obs[6]
    sigma_NiMg = R_NiMg_obs * np.sqrt(rel_Ni**2 + rel_Mg**2)

    def obj(f):
        p = predict_mixture(f, ems)
        err = ((p[0]-obs_e54)/sigma[0])**2 + ((p[1]-obs_e50)/sigma[1])**2
        err += ((p[2]/p[3] - R_CrTi_obs) / sigma_CrTi)**2
        err += ((p[4]/p[2] - R_MgCr_obs) / sigma_MgCr)**2
        err += ((p[5]/p[4] - R_AlMg_obs) / sigma_AlMg)**2
        err += ((p[6]/p[4] - R_NiMg_obs) / sigma_NiMg)**2
        return err

    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bnds = [(0, 1)] * ems.shape[0]
    init = np.ones(ems.shape[0]) / ems.shape[0]
    res = minimize(obj, init, method='SLSQP', bounds=bnds, constraints=cons)
    return res.fun, res.x

# ======================== Compute and output ========================
output_rows = []
for src_name, ems in sources.items():
    for i in range(n_samples):
        sample = meta_df['Sample'].iloc[i]
        typ = meta_df['Type'].iloc[i]
        ob = obs_vals[i]
        sig = obs_sigmas[i]
        error_val, f_opt = solve_fractions(ob, ems, sig)

        # Convert fractions to percentage strings with two decimal places (e.g., 4.00%, 0.07%)
        f_percent = [f"{val*100:.2f}%" for val in f_opt]
        # Example: 4% and 0.07% are both shown with two decimals; adjust if exactly integer formatting is desired.

        row = [sample, typ, src_name, f"{error_val:.4f}"] + f_percent
        output_rows.append(row)

# Column names
cols = ['Sample', 'Type', 'Source', 'Fitting Error'] + EM_NAMES
df_out = pd.DataFrame(output_rows, columns=cols)
df_out.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Results saved to {OUTPUT_FILE}")
print(df_out.head(10))