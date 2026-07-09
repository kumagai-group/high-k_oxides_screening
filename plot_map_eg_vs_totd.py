import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({'font.size': 24})


def add_fom_contours(ax, contour_Cs, cvmin, cvmax, x_max, label_y0, cmap="viridis_r"):
    """Draw iso-FoM curves (y = C / x) colored consistently with the colormap"""
    norm = mcolors.Normalize(vmin=cvmin, vmax=cvmax)
    cmap = plt.get_cmap(cmap)
    x_line = np.linspace(1., x_max, 400)
    for i, C in enumerate(contour_Cs):
        y_line = C / x_line
        color = cmap(norm(C))
        ax.plot(x_line, y_line, color=color, linestyle="-", linewidth=1.5, zorder=0)
        # stagger label heights so that neighboring labels do not overlap
        label_y = label_y0 - 0.7 * i
        ax.text(
            C / label_y, label_y, f"{C}",
            color=color, fontsize=12, fontweight='bold',
            ha='left', va='bottom',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=1),
        )


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

""" Plot the map (colored by FoM = Eg * total dielectric constant) """
cvmin = 0.
cvmax = float(max(train_df["fom"].max(), screened_df["fom"].max()))

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)

add_fom_contours(
    ax, contour_Cs=[500, 1000, 1500, 2000, 2500],
    cvmin=cvmin, cvmax=cvmax, x_max=700., label_y0=11.3,
)

sc = ax.scatter(
    train_df["totd"], train_df["bandgap"],
    s=50, c=train_df["fom"], cmap="viridis_r", vmin=cvmin, vmax=cvmax,
    alpha=0.5, edgecolor="none",
)
ax.scatter(
    screened_df["totd"], screened_df["bandgap"],
    s=50, c=screened_df["fom"], cmap="viridis_r", vmin=cvmin, vmax=cvmax,
    alpha=0.8, marker="D", edgecolor="black",
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
                              markerfacecolor='gold', markeredgecolor="black", label='Candidates')
ax.legend(handles=[train_marker, screen_marker], fontsize=18, frameon=False)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"$\mathrm{FoM}$ ($=E_{\mathrm{g}}^{\mathrm{DDH}} * \epsilon^{\mathrm{total}}_{\mathrm{ave}}$)", fontsize=20)
try:
    cbar.solids.set_alpha(1)
except AttributeError:
    for coll in cbar.ax.collections:
        coll.set_alpha(1)

plt.tight_layout()
save_path = f"{save_stem}.png"
fig.savefig(save_path)
plt.close(fig)
print(f"saved to {save_path}")
