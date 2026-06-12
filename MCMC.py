import numpy as np
import pandas as pd
from scipy.optimize import minimize
import emcee
import corner
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm
import multiprocessing as mp

# ==========================================
# 0. Global configuration
# ==========================================
FILE_PATH = 'data4.xlsx'               # Modify according to actual path
OUTPUT_PREFIX = 'MCMC_Full_Comparison_weighted'
N_WALKERS = 200
N_STEPS_BURN = 2000
N_STEPS_SAMPLE = 3000
THIN = 5

# ==========================================
# 1. Data loading (including measurement errors)
# ==========================================
def load_data(filepath):
    df = pd.read_excel(filepath)
    df = df[(df['Level1'].astype(str).str.startswith('C')) & (df['Level1'] != 'Clast')].copy()
    cols = ['ε54Cr', 'ε50Ti', 'Mg', 'Cr', 'Ti', 'Al', 'Ni']
    df_calc = df.dropna(subset=cols).copy()
    obs_vals = df_calc[cols].values

    try:
        e54_2se = df_calc['ε54Cr2SE'].values
        e50_2se = df_calc['ε50Ti2SE'].values
        print("✅ Successfully read ε54Cr2SE and ε50Ti2SE")
    except KeyError:
        print("⚠️  ε54Cr2SE or ε50Ti2SE columns not found, using default SE=0.05")
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

    print(f"✅ Loaded {len(df_calc)} valid samples with measurement errors")
    return obs_vals, obs_sigmas, df_calc[['Sample', 'Level1', 'Source']].reset_index(drop=True)

obs_vals, obs_sigmas, meta_df = load_data(FILE_PATH)
n_samples = len(obs_vals)

# ==========================================
# 1.5 Fit the NC isotope linear relation (ε⁵⁰Ti = a*ε⁵⁴Cr + b)
# ==========================================
nc_raw = {
    'ε54Cr':  [-0.38, -0.32, -0.38,  0.02,  0.02, -0.84, -0.68, -0.80, -0.36, -0.42, -0.55, -0.31, -0.35, -0.43, -0.74, -0.78, -0.65, -0.67],
    '2SE_54': [ 0.02,  0.16,  0.09,  0.04,  0.07,  0.07,  0.09,  0.09,  0.07,  0.09,  0.11,  0.10,  0.06,  0.13,  0.11,  0.06,  0.05,  0.05],
    'ε50Ti':  [-0.50, -0.63, -0.67, -0.11, -0.28, -1.85, -2.24, -2.05, -1.12, -1.29, -1.02, -1.16, -1.16, -1.18, -1.35, -1.28, -1.23, -1.24],
    '2SE_50': [ 0.05,  0.03,  0.06,  0.05,  0.17,  0.20,  0.44,  0.24,  0.23,  0.10,  0.10,  0.06,  0.06,  0.08,  0.14,  0.10,  0.07,  0.05]
}
df_nc_prior = pd.DataFrame(nc_raw)
x_nc = df_nc_prior['ε54Cr'].values
y_nc = df_nc_prior['ε50Ti'].values
e_y = df_nc_prior['2SE_50'].values / 2.0
w_nc = 1.0 / e_y**2
coeff_nc = np.polyfit(x_nc, y_nc, 1, w=w_nc)
NC_LINE_A = coeff_nc[0]
NC_LINE_B = coeff_nc[1]
y_pred_nc = NC_LINE_A * x_nc + NC_LINE_B
resid_nc = y_nc - y_pred_nc
NC_LINE_SIGMA = np.sqrt(np.sum(w_nc * resid_nc**2) / np.sum(w_nc))
print(f"✅ NC isotope linear relation: ε⁵⁰Ti = {NC_LINE_A:.3f} * ε⁵⁴Cr + {NC_LINE_B:.3f}, σ = {NC_LINE_SIGMA:.3f}")

# ==========================================
# 2. Model definitions (end-member names, parameter bounds, CAI priors)
# ==========================================
def make_labels(names):
    return [f'{n}_{el}' for n in names for el in ['ε54','ε50','Mg','Cr','Ti','Al','Ni']]

# --- Two, three, four end-member definitions ---
TWO_NAMES = ['NC_Avg', 'CAI_Avg']
TWO_BOUNDS = np.array([
    [-4, 0], [-6, 0], [0.5, 1.5], [0.5, 1.5], [0.5, 1.3], [0.5, 1.5], [0.1, 2.0],
    [2, 10], [5, 14], [0.3, 1.5], [0.01, 1.0], [10, 45], [10, 45], [0.01, 1.0]
])

THREE_NAMES = ['NC_Avg', 'CAI_Avg', 'CI']
THREE_BOUNDS = np.array([
    [-4, 0], [-6, 0], [0.5, 1.5], [0.5, 1.5], [0.5, 1.3], [0.5, 1.5], [0.1, 2.0],
    [2, 10], [5, 14], [0.3, 1.5], [0.01, 1.0], [5, 20], [5, 20], [0.01, 1.0],
    [1.5, 1.7], [1.8, 2.0], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2]
])

FOUR_NC_NAMES = ['NC_L', 'NC_H', 'CAI_Avg', 'CI']
FOUR_NC_BOUNDS = np.array([
    [-4, 0], [-6, 0], [0.5, 1.5], [0.5, 1.5], [0.5, 1.3], [0.5, 1.5], [0.01, 0.8],
    [-4, 0], [-6, 0], [1e-3, 0.5], [0.5, 5.0], [1e-3, 0.1], [1e-3, 0.5], [5.0, 40.0],
    [2, 10], [5, 14], [0.3, 1.5], [0.01, 1.0], [5, 20], [5, 20], [0.01, 1.0],
    [1.5, 1.7], [1.8, 2.0], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2]
])

FOUR_CAI_NAMES = ['NC_Avg', 'CAI_Ultra', 'CAI_Norm', 'CI']
FOUR_CAI_BOUNDS = np.array([
    [-4, 0], [-6, 0], [0.5, 1.5], [0.5, 1.5], [0.5, 1.3], [0.5, 1.5], [0.1, 2.0],
    [2, 10], [7, 15], [0.2, 1.5], [0.01, 0.8], [5, 55], [5, 55], [0.01, 1.0],
    [0.5, 7], [2, 10], [0.5, 1.8], [0.05, 1.5], [2, 25], [2, 25], [0.01, 1.0],
    [1.5, 1.7], [1.8, 2.0], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2], [0.8, 1.2]
])

# ========== CAI prior definitions ==========
cai_avg_priors = [{'em_idx': 1, 'param_idx': 1, 'mean': 9.0, 'sigma': 1.0}]
cai_avg_four_priors = [{'em_idx': 2, 'param_idx': 1, 'mean': 9.0, 'sigma': 1.0}]
four_cai_ultra_norm_priors = [
    {'em_idx': 1, 'param_idx': 1, 'mean': 9.0, 'sigma': 1.0},
    {'em_idx': 1, 'param_idx': 4, 'mean': 14.88, 'sigma': 2.0},
    {'em_idx': 2, 'param_idx': 1, 'mean': 4.0, 'sigma': 1.0},
    {'em_idx': 2, 'param_idx': 4, 'mean': 14.88, 'sigma': 2.0}
]

# ========== Model configurations (only two, three, and four end-member models retained) ==========
MODEL_CONFIGS = {
    'two':      {'names': TWO_NAMES,      'bounds': TWO_BOUNDS,      'n_ems': 2, 'nc_idxs': [0],      'ca_priors': cai_avg_priors},
    'three':    {'names': THREE_NAMES,    'bounds': THREE_BOUNDS,    'n_ems': 3, 'nc_idxs': [0],      'ca_priors': cai_avg_priors},
    'four_nc':  {'names': FOUR_NC_NAMES,  'bounds': FOUR_NC_BOUNDS,  'n_ems': 4, 'nc_idxs': [0, 1],   'ca_priors': cai_avg_four_priors},
    'four_cai': {'names': FOUR_CAI_NAMES, 'bounds': FOUR_CAI_BOUNDS, 'n_ems': 4, 'nc_idxs': [0],      'ca_priors': four_cai_ultra_norm_priors}
}
# Add labels to each configuration
for key in MODEL_CONFIGS:
    MODEL_CONFIGS[key]['labels'] = make_labels(MODEL_CONFIGS[key]['names'])

# ==========================================
# 3. Helper functions
# ==========================================
def params_to_ems_generic(theta, n_ems):
    return theta.reshape(n_ems, 7)

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
    # ---- Fix: add defensive check to ensure sigma has correct shape ----
    sigma = np.asarray(sigma)
    if sigma.ndim != 1 or len(sigma) != 7:
        raise ValueError(f"sigma must be a 1D array of length 7, got shape {sigma.shape}")
    # ---------------------------------------------------------
    n_ems = ems.shape[0]
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
    bnds = [(0, 1)] * n_ems
    init = np.ones(n_ems) / n_ems
    res = minimize(obj, init, method='SLSQP', bounds=bnds, constraints=cons)
    return res.fun, res.x

def nc_line_logprior(ems, nc_idxs, a, b, sigma):
    lp = 0.0
    for idx in nc_idxs:
        e54 = ems[idx, 0]
        e50 = ems[idx, 1]
        resid = e50 - (a * e54 + b)
        lp -= 0.5 * (resid / sigma) ** 2
    return lp

def em_param_prior(ems, priors):
    lp = 0.0
    for pr in priors:
        val = ems[pr['em_idx'], pr['param_idx']]
        lp -= 0.5 * ((val - pr['mean']) / pr['sigma']) ** 2
    return lp

def log_prior(theta, bounds):
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        return -np.inf
    return 0.0

def _get_ems(theta, cfg):
    return params_to_ems_generic(theta, cfg['n_ems'])

def log_likelihood(obs_set, ems, sigmas):
    total = 0.0
    for i, ob in enumerate(obs_set):
        err, _ = solve_fractions(ob, ems, sigmas[i])
        total -= 0.5 * err
    return total

def log_probability(theta, obs, bounds, cfg):
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf

    ems = _get_ems(theta, cfg)

    # NC isotope line prior
    nc_idxs = cfg.get('nc_idxs', None)
    if nc_idxs is not None:
        lp += nc_line_logprior(ems, nc_idxs, NC_LINE_A, NC_LINE_B, NC_LINE_SIGMA)

    # CAI end-member priors
    ca_priors = cfg.get('ca_priors', None)
    if ca_priors is not None:
        lp += em_param_prior(ems, ca_priors)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(obs, ems, cfg['obs_sigmas'])

# ==========================================
# 4. MCMC running (with functionality to skip already computed models)
# ==========================================
def run_mcmc(obs, cfg, model_type, n_walkers, n_burn, n_sample, thin):
    bounds = cfg['bounds']
    ndim = len(bounds)
    filename = f"{OUTPUT_PREFIX}_{model_type}.h5"
    backend = emcee.backends.HDFBackend(filename)

    n_expected = n_sample // thin
    # Try to load from existing file; skip if iterations are sufficient
    try:
        if backend.iteration > 0:
            if backend.iteration >= n_expected:
                print(f"⏩ Skipping {model_type}: already has {backend.iteration} steps (≥{n_expected})")
                return backend
            else:
                print(f"⚠️ {model_type} incomplete ({backend.iteration}/{n_expected}), re-initializing...")
                backend.reset(n_walkers, ndim)
        else:
            backend.reset(n_walkers, ndim)
    except Exception:
        backend.reset(n_walkers, ndim)

    p0 = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_walkers, ndim))
    with mp.Pool() as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers, ndim, log_probability,
            args=[obs, bounds, cfg],
            backend=backend,
            moves=emcee.moves.StretchMove(a=2.0),
            pool=pool
        )
        print(f"\n🔥 Burn-in ({model_type})...")
        state = sampler.run_mcmc(p0, n_burn, progress=True)
        sampler.reset()
        print(f"📈 Sampling ({model_type})...")
        sampler.run_mcmc(state, n_sample, thin=thin, progress=True)
    return backend

# ==========================================
# 5. WAIC computation
# ==========================================
def compute_waic(flat_samples, obs, bounds, cfg, nsamp=200):
    subset = flat_samples[np.random.choice(len(flat_samples), nsamp, replace=False)]
    n_obs = len(obs)
    sigmas = cfg['obs_sigmas']
    lppd = np.zeros(n_obs)
    p_waic = np.zeros(n_obs)
    for theta in tqdm(subset, desc=f"WAIC"):
        ems = _get_ems(theta, cfg)
        for i, ob in enumerate(obs):
            err, _ = solve_fractions(ob, ems, sigmas[i])
            logL = -0.5 * err
            lppd[i] += logL
            p_waic[i] += logL**2
    lppd /= nsamp
    p_waic = p_waic / nsamp - lppd**2
    waic_vec = -2.0 * (lppd - p_waic)
    waic = np.sum(waic_vec)
    waic_se = np.sqrt(n_obs * np.var(waic_vec))
    return waic, waic_se, waic_vec

# ==========================================
# 6. Posterior predictive check and plotting (corrected)
# ==========================================
def posterior_predictive(flat_samples, obs, cfg, nsamp=100):
    sigmas = cfg['obs_sigmas']                # shape: (n_samples, 7)
    idx_sub = np.random.choice(len(flat_samples), nsamp, replace=False)
    preds = []
    for idx in idx_sub:
        theta = flat_samples[idx]
        ems = _get_ems(theta, cfg)
        pred_obs = []
        for i, ob in enumerate(obs):
            _, f_opt = solve_fractions(ob, ems, sigmas[i])
            pred_obs.append(predict_mixture(f_opt, ems)[:2])
        preds.append(pred_obs)
    return np.array(preds)

def plot_waic_comparison(model_tags, waic_vals, waic_se, save_path):
    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(len(model_tags))
    ax.bar(x, waic_vals, yerr=waic_se, capsize=5,
           color=plt.cm.viridis(np.linspace(0,1,len(model_tags))))
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('_','-') for t in model_tags], rotation=15)
    ax.set_ylabel('WAIC')
    ax.set_title('Model Comparison (WAIC)')
    for i, (v, se) in enumerate(zip(waic_vals, waic_se)):
        ax.text(i, v + se + 5, f'{v:.0f}\n±{se:.0f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 7. Main workflow
# ==========================================
def main():
    print("🚀 Full model MCMC comparison (weighted, models: two, three, four_nc, four_cai)")
    model_tags = list(MODEL_CONFIGS.keys())
    results = {}

    # Add observation errors to each model (shared)
    for tag in model_tags:
        MODEL_CONFIGS[tag]['obs_sigmas'] = obs_sigmas

    for tag in model_tags:
        cfg = MODEL_CONFIGS[tag]
        print(f"\n{'='*40}\n▶ Running {tag} model ({cfg['n_ems']} end-members)")
        backend = run_mcmc(obs_vals, cfg, tag, N_WALKERS, N_STEPS_BURN, N_STEPS_SAMPLE, THIN)
        flat = backend.get_chain(flat=True)
        results[tag] = {'flat': flat, 'cfg': cfg}

        # Generate posterior summary if it doesn't exist
        summary_path = f"{OUTPUT_PREFIX}_{tag}_posterior_summary_extended.csv"
        if not os.path.exists(summary_path):
            flat_df = pd.DataFrame(flat, columns=cfg['labels'])
            summary = flat_df.describe(percentiles=[0.05, 0.95]).T
            summary = summary[['min', 'mean', 'max', 'std', '5%', '95%']]
            summary.columns = ['Min', 'Mean', 'Max', 'Std', '5%', '95%']
            summary.to_csv(summary_path)
            print(f"  Posterior summary saved to {summary_path}")
        else:
            print(f"  Posterior summary already exists, skipping write")

    print("\n📊 Computing weighted WAIC (200 posterior samples per model)...")
    waic_results = {}
    for tag in model_tags:
        flat = results[tag]['flat']
        cfg = results[tag]['cfg']
        waic, waic_se, waic_vec = compute_waic(flat, obs_vals, cfg['bounds'], cfg, nsamp=200)
        waic_results[tag] = (waic, waic_se)
        pd.Series(waic_vec, name='waic_i').to_csv(f"{OUTPUT_PREFIX}_{tag}_waic_vector.csv", index=False)
        print(f"  {tag}: WAIC = {waic:.2f} ± {waic_se:.2f}")

    sorted_tags = sorted(model_tags, key=lambda t: waic_results[t][0])
    best_tag = sorted_tags[0]
    print(f"\n🔎 ΔWAIC (relative to {best_tag}):")
    ref_waic, ref_se = waic_results[best_tag]
    for t in sorted_tags:
        if t == best_tag: continue
        diff = waic_results[t][0] - ref_waic
        diff_se = np.sqrt(waic_results[t][1]**2 + ref_se**2)
        sig = "significant" if diff > 2*diff_se else "not significant"
        print(f"  {t} minus {best_tag}: {diff:.2f} ± {diff_se:.2f} ({sig})")

    plot_waic_comparison(sorted_tags, [waic_results[t][0] for t in sorted_tags],
                         [waic_results[t][1] for t in sorted_tags],
                         f"{OUTPUT_PREFIX}_WAIC_comparison.pdf")
    print("📈 WAIC comparison plot saved")

    print(f"\n🎨 Best model ({best_tag}) posterior predictive check")
    best_flat = results[best_tag]['flat']
    best_cfg = results[best_tag]['cfg']
    preds = posterior_predictive(best_flat, obs_vals, best_cfg, nsamp=50)
    np.savez_compressed(f"{OUTPUT_PREFIX}_{best_tag}_postpred.npz", preds=preds)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    obs_e54 = obs_vals[:, 0]; obs_e50 = obs_vals[:, 1]
    pred_mean_e54 = np.mean(preds[:,:,0], axis=0)
    pred_low_e54 = np.percentile(preds[:,:,0], 2.5, axis=0)
    pred_high_e54 = np.percentile(preds[:,:,0], 97.5, axis=0)
    pred_mean_e50 = np.mean(preds[:,:,1], axis=0)
    pred_low_e50 = np.percentile(preds[:,:,1], 2.5, axis=0)
    pred_high_e50 = np.percentile(preds[:,:,1], 97.5, axis=0)
    idx = np.arange(n_samples)
    axes[0].fill_between(idx, pred_low_e54, pred_high_e54, alpha=0.3, label='95% CI')
    axes[0].plot(idx, pred_mean_e54, 'r-', label='Mean pred')
    axes[0].scatter(idx, obs_e54, c='k', s=10, label='Observed')
    axes[0].set_ylabel('ε⁵⁴Cr'); axes[0].legend()
    axes[1].fill_between(idx, pred_low_e50, pred_high_e50, alpha=0.3, label='95% CI')
    axes[1].plot(idx, pred_mean_e50, 'r-', label='Mean pred')
    axes[1].scatter(idx, obs_e50, c='k', s=10, label='Observed')
    axes[1].set_ylabel('ε⁵⁰Ti'); axes[1].legend()
    plt.suptitle(f"Posterior Predictive Check – {best_tag.replace('_','-')}")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PREFIX}_{best_tag}_posterior_predictive.pdf")
    plt.close()

    # Corner plot
    n_ems = best_cfg['n_ems']
    reshaped = best_flat.reshape(-1, n_ems, 7)
    data_list, lab_list = [], []
    for i in range(n_ems):
        e54 = reshaped[:, i, 0]
        e50 = reshaped[:, i, 1]
        nimgo = np.log10(reshaped[:, i, 6] / reshaped[:, i, 2])
        data_list.extend([e54, e50, nimgo])
        lab_list.extend([f'ε⁵⁴Cr_{best_cfg["names"][i]}',
                         f'ε⁵⁰Ti_{best_cfg["names"][i]}',
                         f'log Ni/Mg_{best_cfg["names"][i]}'])
    corner_data = np.column_stack(data_list[:10])
    corner_labels = lab_list[:10]
    fig = corner.corner(corner_data, labels=corner_labels, show_titles=True,
                        quantiles=[0.16, 0.5, 0.84])
    plt.savefig(f"{OUTPUT_PREFIX}_{best_tag}_corner.pdf")
    plt.close()
    print(f"📊 Corner plot saved to {OUTPUT_PREFIX}_{best_tag}_corner.pdf")
    print("\n✅ All analyses complete!")

if __name__ == "__main__":
    main()