import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({'font.size': 24})


def load_data(matnames_path, bandgap_dir, diele_dir):
    """Read a matname list and return Eg (DDH) and total dielectric constant (PBEsol, trace/3) as a DataFrame"""
    with open(matnames_path) as f:
        matnames = [line.strip() for line in f if line.strip()]

    df = {"matname": [], "bandgap": [], "totd": []}
    for matname in matnames:
        bandgap = float(np.loadtxt(f"{bandgap_dir}/{matname}/bandgap.txt"))
        totd_ten = np.loadtxt(f"{diele_dir}/{matname}/totd.txt")
        totd_sca = float(np.trace(totd_ten)) / 3.

        df["matname"].append(matname)
        df["bandgap"].append(bandgap)
        df["totd"].append(totd_sca)

    df = pd.DataFrame(df)
    df["fom"] = df["bandgap"] * df["totd"]
    return df


""" Training data (928 materials) """
train_df = load_data(
    matnames_path="database/matnames_train.txt",
    bandgap_dir="database/bandgap_ddh",
    diele_dir="database/dielectric_pbesol",
)
print(f"train: {len(train_df)}")

""" Screened materials (31 materials) """
screened_df = load_data(
    matnames_path="results_screening/matnames_screened31.txt",
    bandgap_dir="results_screening/bandgap_ddh",
    diele_dir="results_screening/dielectric_pbesol",
)
print(f"screened: {len(screened_df)}")

save_stem = "results_screening/map_eg_ddh_vs_totd_pbesol"
train_df.to_csv(f"{save_stem}_train.csv", index=False)
screened_df.to_csv(f"{save_stem}_screened.csv", index=False)

""" Plot the map """
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)

ax.scatter(
    train_df["totd"], train_df["bandgap"],
    s=50, color="gold", alpha=0.5, edgecolor="none",
)
ax.scatter(
    screened_df["totd"], screened_df["bandgap"],
    s=50, color="crimson", alpha=0.8, marker="D", edgecolor="black",
)

""" Annotate the best-FoM candidate with its composition """
idx_best = screened_df["fom"].idxmax()
best_row = screened_df.loc[idx_best]
best_mpid, best_formula = best_row["matname"].split("_")
print(f"best candidate: {best_row['matname']} (FoM = {best_row['fom']:.1f})")
best_formula_sub = re.sub(r"(\d+)", r"$_{\1}$", best_formula)
ax.annotate(
    best_formula_sub,
    (best_row["totd"], best_row["bandgap"]),
    textcoords="offset points",
    xytext=(0, 12),
    ha='center',
    va='bottom',
    fontsize=20,
    color='black',
)

ax.set_xlabel(r"$\epsilon^\mathrm{total}_\mathrm{ave}$")
ax.set_ylabel(r"$E_g^{\mathrm{DDH}}$ [eV]")

ax.set_xlim(0, 700)
ax.set_ylim(0, 12)
ax.xaxis.set_major_locator(MultipleLocator(200))

train_marker = mlines.Line2D([], [], color='none', marker='o', markersize=10,
                             markerfacecolor='gold', markeredgecolor="none", label='Train')
screen_marker = mlines.Line2D([], [], color='none', marker='D', markersize=10,
                              markerfacecolor='crimson', markeredgecolor="black", label='Candidates')
ax.legend(handles=[train_marker, screen_marker], fontsize=18, frameon=False)

plt.tight_layout()
save_path = f"{save_stem}.png"
fig.savefig(save_path)
plt.close(fig)
print(f"saved to {save_path}")
