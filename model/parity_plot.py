import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

FONTSIZE = 24
SIZE=100
ALPHA=0.2


class ParityPlot:
    def __init__(
            self, 
            x_data: np.ndarray, 
            y_data: np.ndarray
    ):
        self.x_data = x_data
        self.y_data = y_data

    def calc_r2(self):
        # r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = r2_score(self.x_data, self.y_data)

        return r2
    
    def calc_mae(self):
        # mae = np.mean(np.abs(self.x_data - self.y_data))
        mae = mean_absolute_error(self.x_data, self.y_data)
        return mae

    def calc_mse(self):
        mse = mean_squared_error(self.x_data, self.y_data)
        # mse = np.mean((self.x_data - self.y_data) ** 2)
        return mse

    def calc_rmse(self):
        rmse = np.sqrt(self.calc_mse())
        return rmse

    def plot_parity_scatter(self, ax, x_vals, y_vals, label=None, color=None, size=20, alpha=1., is_box=True):
        """パリティプロット用の散布図と軸範囲更新"""
        if is_box:
            ax.set_aspect('equal', adjustable='box')
        ax.scatter(x_vals, y_vals, s=size, label=label, c=color, alpha=alpha, edgecolor="none")
        return x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()
    
    def plot_parity_hexbin(self, ax, x_vals, y_vals, fontsize=20, gridsize=50, cmap='Greens', mincnt=1):
        """
        パリティプロット用の六角形ビン密度プロット
        """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        from matplotlib.colors import LogNorm

        ax.set_aspect('equal', adjustable='box')
        # hb = ax.hexbin(x_vals, y_vals, gridsize=gridsize, cmap=cmap, mincnt=mincnt)
        hb = ax.hexbin(
            x_vals, 
            y_vals, 
            gridsize=50, 
            cmap=cmap, 
            norm=LogNorm(), 
        )
        cax = inset_axes(
            ax, 
            width="35%", 
            height="3%", 
            loc='upper left',
            borderpad=2.5, 
        )
        # カラーバーの作成
        cb = plt.colorbar(hb, cax=cax, orientation='horizontal')
        cb.ax.tick_params(labelsize=20)
        cb.ax.text(1.1, 0.4, 'Counts', fontsize=fontsize,
                va='center', ha='left', transform=cb.ax.transAxes)
        return x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()

    def add_diagonal_line(self, ax, x_min, x_max, y_min, y_max):
        """パリティプロットの対角線を引く"""
        min_val, max_val = min(x_min, y_min), max(x_max, y_max)
        ax.plot([min_val, max_val], [min_val, max_val],
                color='grey', linestyle='--', label='Diagonal')

    def annotate_r2(self, ax, r2_values, fontsize=10):
        """R2値のテキストをグラフ上に描画"""
        if not isinstance(r2_values, list):
            r2_values = [r2_values]
        r2_text = "\n".join([r"$R^2$"+f" = {r2:.2f}" for r2 in r2_values])
        ax.text(
            0.95, 0.05, r2_text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='bottom', horizontalalignment='right',
            # bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

    def annotate_score(self, ax, scores, fontsize=10):
        score_text = "\n".join([tag+f" = {val:.3f}" for (tag, val) in scores])
        ax.text(
            0.95, 0.05, score_text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='bottom', horizontalalignment='right',
            # bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )

    def plot_with_hist(
        self, 
        color=None, 
        alpha=1., 
        is_box=True, 
        x_label="true", y_label="pred", 
        fontsize=10, 
        fixed_range=None, 
        ticks_width=None, 
        save_path="parity_plot.png"
    ):
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(4, 4, wspace=0.0001, hspace=0.0)

        """ 散布図 """
        ax_scatter = fig.add_subplot(gs[1:4, 0:3])   # 散布図

        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
        _x_min, _x_max, _y_min, _y_max = self.plot_parity_scatter(
            ax=ax_scatter, 
            x_vals=self.x_data, 
            y_vals=self.y_data, 
            alpha=alpha, 
            is_box=is_box
        )

        x_min, x_max = min(x_min, _x_min), max(x_max, _x_max)
        y_min, y_max = min(y_min, _y_min), max(y_max, _y_max)

        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)
        print(min_val, max_val)
        if fixed_range is not None:
            min_val, max_val = fixed_range

        r2 = self.calc_r2()
        self.annotate_r2(ax_scatter, [r2], fontsize=fontsize)
        self.add_diagonal_line(ax_scatter, min_val, max_val, min_val, max_val)

        ax_scatter.set_aspect('equal')
    
        ax_scatter.set_xlabel(x_label, fontsize=fontsize)
        ax_scatter.set_ylabel(y_label, fontsize=fontsize)
        ax_scatter.tick_params(axis='both', labelsize=fontsize)
        
        if ticks_width is None:
            ticks_width = (max_val - min_val) / 2
        ticks = np.arange(min_val, max_val + .5, ticks_width)
        ax_scatter.set_xticks(ticks)
        ax_scatter.set_yticks(ticks)

        """ ヒストグラム """
        ax_histx = fig.add_subplot(gs[0, 0:3], sharex=ax_scatter)  # x軸ヒストグラム
        ax_histy = fig.add_subplot(gs[1:4, 3])  # y軸ヒストグラム

        num_bins = 30
        bins = np.linspace(min_val, max_val, num_bins + 1)

        ax_histx.hist(self.x_data, bins=bins, density=True, color="gray", edgecolor='black')
        ax_histy.hist(self.y_data, bins=bins, density=True, color="gray", edgecolor='black', orientation='horizontal')

        ax_histx.tick_params(
            axis='both',
            which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False
        )
        ax_histy.tick_params(
            axis='both',
            which='both',
            bottom=False, top=False, left=False, right=False,
            labelbottom=False, labelleft=False
        )

        for ax in [ax_histx, ax_histy]:
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def plot_without_hist(
        self, 
        color=None, 
        cmap="Blues", 
        size=20, 
        alpha=1., 
        is_box=True, 
        x_label="true", y_label="pred", 
        fontsize=20, 
        fixed_range=None, 
        fixed_ticks=None, 
        ticks_width=None, 
        path_save="parity_plot.png", 
        show_2nd_axis=False, 
        secax_xlabel="", 
        secax_ylabel="", 
    ):
        fig = plt.figure(figsize=(8, 8))

        """ 散布図 """
        ax_scatter = fig.add_subplot()   # 散布図

        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
        _x_min, _x_max, _y_min, _y_max = self.plot_parity_scatter(
            ax=ax_scatter, 
            x_vals=self.x_data, 
            y_vals=self.y_data, 
            color=color, 
            size=size, 
            alpha=alpha, 
            is_box=is_box, 
        )
        # _x_min, _x_max, _y_min, _y_max = self.plot_parity_hexbin(
        #     ax=ax_scatter, 
        #     x_vals=self.x_data, 
        #     y_vals=self.y_data, 
        #     fontsize=fontsize, 
        #     cmap=cmap, 
        # )

        x_min, x_max = min(x_min, _x_min), max(x_max, _x_max)
        y_min, y_max = min(y_min, _y_min), max(y_max, _y_max)

        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)
        # print(min_val, max_val)
        if fixed_range is not None:
            min_val, max_val = fixed_range

        mae = self.calc_mae()
        rmse = self.calc_rmse()
        r2 = self.calc_r2()
        # print("MAE=", mae, ", R2=", r2)
        scores = [
            # ("MAE", mae), 
            ("RMSE", rmse), 
            (r"$R^2$", r2)]
        self.annotate_score(ax_scatter, scores, fontsize=fontsize)
        self.add_diagonal_line(ax_scatter, -1000, 1000, -1000, 1000)

        ax_scatter.set_aspect('equal')
    
        ax_scatter.set_xlabel(x_label, fontsize=fontsize)
        ax_scatter.set_ylabel(y_label, fontsize=fontsize)
        ax_scatter.tick_params(axis='both', labelsize=fontsize)
        
        if ticks_width is None:
            ticks_width = (max_val - min_val) / 2
        tick_start = np.floor(min_val / ticks_width) * ticks_width
        tick_end = np.ceil(max_val / ticks_width) * ticks_width
        if fixed_ticks is None:
            fixed_ticks = (min_val, max_val)
        ticks = np.arange(fixed_ticks[0], fixed_ticks[1]+0.1, ticks_width)
        ax_scatter.set_xticks(ticks)
        ax_scatter.set_yticks(ticks)
        range_x = abs(max_val - min_val)
        range_y = abs(max_val - min_val)
        ax_scatter.set_xlim(min_val-(0.05*range_x), max_val+(0.05*range_x))
        ax_scatter.set_ylim(min_val-(0.05*range_y), max_val+(0.05*range_y))
        if show_2nd_axis:
            def format_func(value, _):
                if value < 1:
                    return f"{value:.1f}"
                else:
                    return f"{value:.0f}"

            # 主目盛（log軸の値を10^xで線形軸に変換）
            secax_x = ax_scatter.secondary_xaxis('top', functions=(lambda x: 10**x, lambda x: np.log10(x)))
            secax_y = ax_scatter.secondary_yaxis('right', functions=(lambda y: 10**y, lambda y: np.log10(y)))

            # 主目盛値（log軸のtickを10^xで変換して線形軸に表示）
            # linear_tick_vals = [10**val for val in np.arange(fixed_ticks[0], fixed_ticks[1]+0.1, ticks_width)]
            linear_tick_vals = [10**val for val in np.arange(-2, 4+0.1, 1)]
            secax_x.set_xticks(linear_tick_vals)
            secax_y.set_yticks(linear_tick_vals)
            secax_x.set_xlabel(secax_xlabel, fontsize=20)
            secax_y.set_ylabel(secax_ylabel, fontsize=20)
            secax_x.set_xticklabels([format_func(val, None) for val in linear_tick_vals], rotation=45, fontsize=20)
            secax_y.set_yticklabels([format_func(val, None) for val in linear_tick_vals], fontsize=20)

            # 副目盛値（各主目盛区間内で対数的に分割）
            minor_tick_vals = []
            # log_tick_vals = np.arange(fixed_ticks[0], fixed_ticks[1]+0.1, ticks_width)
            log_tick_vals = np.arange(-2, 4+0.1, 1)
            for i in range(len(log_tick_vals)-1):
                start = log_tick_vals[i]
                end = log_tick_vals[i+1]
                base = 10 ** start
                next_base = 10 ** end
                for m in range(2, 10):
                    tick = base * m
                    if tick < next_base:
                        minor_tick_vals.append(tick)
            secax_x.set_xticks(minor_tick_vals, minor=True)
            secax_y.set_yticks(minor_tick_vals, minor=True)

        plt.tight_layout()
        fig.savefig(path_save)
        plt.close(fig)
    
    def plot_without_hist_double(
        self, 
        x_data2, 
        y_data2, 
        color=None, 
        cmap="Blues", 
        size=20, 
        alpha=1., 
        is_box=True, 
        x_label="true", y_label="pred", 
        fontsize=10, 
        fixed_range=None, 
        fixed_ticks=None, 
        ticks_width=None, 
        save_path="parity_plot.png", 
        show_2nd_axis=True, 
        secax_xlabel="", 
        secax_ylabel="", 
    ):
        fig = plt.figure(figsize=(8, 8))

        """ 散布図 """
        ax_scatter = fig.add_subplot()   # 散布図

        x_min, x_max, y_min, y_max = float('inf'), float('-inf'), float('inf'), float('-inf')
        _x_min, _x_max, _y_min, _y_max = self.plot_parity_scatter(
            ax=ax_scatter, 
            x_vals=self.x_data, 
            y_vals=self.y_data, 
            color=color, 
            size=size, 
            alpha=alpha, 
            is_box=is_box, 
        )
        self.plot_parity_scatter(
            ax=ax_scatter, 
            x_vals=x_data2, 
            y_vals=y_data2, 
            color="orange", 
            size=size, 
            alpha=alpha, 
            is_box=is_box, 
        )
        # _x_min, _x_max, _y_min, _y_max = self.plot_parity_hexbin(
        #     ax=ax_scatter, 
        #     x_vals=self.x_data, 
        #     y_vals=self.y_data, 
        #     fontsize=fontsize, 
        #     cmap=cmap, 
        # )

        x_min, x_max = min(x_min, _x_min), max(x_max, _x_max)
        y_min, y_max = min(y_min, _y_min), max(y_max, _y_max)

        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)
        print(min_val, max_val)
        if fixed_range is not None:
            min_val, max_val = fixed_range

        mae = self.calc_mae()
        rmse = self.calc_rmse()
        r2 = self.calc_r2()
        print("MAE=", mae, ", R2=", r2)
        scores = [
            # ("MAE", mae), 
            ("RMSE", rmse), 
            (r"$R^2$", r2)]
        self.annotate_score(ax_scatter, scores, fontsize=fontsize)
        self.add_diagonal_line(ax_scatter, -1000, 1000, -1000, 1000)

        ax_scatter.set_aspect('equal')
    
        ax_scatter.set_xlabel(x_label, fontsize=fontsize)
        ax_scatter.set_ylabel(y_label, fontsize=fontsize)
        ax_scatter.tick_params(axis='both', labelsize=fontsize)
        
        if ticks_width is None:
            ticks_width = (max_val - min_val) / 2
        tick_start = np.floor(min_val / ticks_width) * ticks_width
        tick_end = np.ceil(max_val / ticks_width) * ticks_width
        if fixed_ticks is None:
            fixed_ticks = (min_val, max_val)
        ticks = np.arange(fixed_ticks[0], fixed_ticks[1]+0.1, ticks_width)
        ax_scatter.set_xticks(ticks)
        ax_scatter.set_yticks(ticks)
        range_x = abs(max_val - min_val)
        range_y = abs(max_val - min_val)
        ax_scatter.set_xlim(min_val-(0.05*range_x), max_val+(0.05*range_x))
        ax_scatter.set_ylim(min_val-(0.05*range_y), max_val+(0.05*range_y))
        if show_2nd_axis:
            def format_func(value, _):
                if value < 1:
                    return f"{value:.1f}"
                else:
                    return f"{value:.0f}"

            # 主目盛（log軸の値を10^xで線形軸に変換）
            secax_x = ax_scatter.secondary_xaxis('top', functions=(lambda x: 10**x, lambda x: np.log10(x)))
            secax_y = ax_scatter.secondary_yaxis('right', functions=(lambda y: 10**y, lambda y: np.log10(y)))

            # 主目盛値（log軸のtickを10^xで変換して線形軸に表示）
            # linear_tick_vals = [10**val for val in np.arange(fixed_ticks[0], fixed_ticks[1]+0.1, ticks_width)]
            linear_tick_vals = [10**val for val in np.arange(-2, 4+0.1, 1)]
            secax_x.set_xticks(linear_tick_vals)
            secax_y.set_yticks(linear_tick_vals)
            secax_x.set_xlabel(secax_xlabel, fontsize=20)
            secax_y.set_ylabel(secax_ylabel, fontsize=20)
            secax_x.set_xticklabels([format_func(val, None) for val in linear_tick_vals], rotation=45, fontsize=20)
            secax_y.set_yticklabels([format_func(val, None) for val in linear_tick_vals], fontsize=20)

            # 副目盛値（各主目盛区間内で対数的に分割）
            minor_tick_vals = []
            # log_tick_vals = np.arange(fixed_ticks[0], fixed_ticks[1]+0.1, ticks_width)
            log_tick_vals = np.arange(-2, 4+0.1, 1)
            for i in range(len(log_tick_vals)-1):
                start = log_tick_vals[i]
                end = log_tick_vals[i+1]
                base = 10 ** start
                next_base = 10 ** end
                for m in range(2, 10):
                    tick = base * m
                    if tick < next_base:
                        minor_tick_vals.append(tick)
            secax_x.set_xticks(minor_tick_vals, minor=True)
            secax_y.set_yticks(minor_tick_vals, minor=True)

        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def save_as_csv(self, save_path):
        df = {}
        df["true"] = self.x_data
        df["pred"] = self.y_data

        import pandas as pd
        df = pd.DataFrame(df)
        df.to_csv(save_path.split(".")[0]+".csv", index=None)


def get_pp(
        x_data, 
        y_data, 
        cmap, 
        color, 
        x_label, 
        y_label, 
        fixed_range, 
        fixed_ticks, 
        ticks_width, 
        save_dir, 
        save_fname, 
        show_2nd_axis, 
        secax_xlabel, 
        secax_ylabel, 
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{save_fname}"
    pp_bec_eig = ParityPlot(
        x_data=x_data, 
        y_data=y_data, 
    )
    pp_bec_eig.plot_without_hist(
        cmap=cmap, 
        color=color, 
        size=SIZE, 
        alpha=ALPHA, 
        x_label=x_label, 
        y_label=y_label, 
        fontsize=FONTSIZE, 
        fixed_range=fixed_range, 
        fixed_ticks=fixed_ticks, 
        ticks_width=ticks_width, 
        save_path=save_path, 
        show_2nd_axis=show_2nd_axis, 
        secax_xlabel=secax_xlabel, 
        secax_ylabel=secax_ylabel, 
    )
    pp_bec_eig.save_as_csv(save_path.split(".")[0]+".csv")
    print(f" saved to {save_path}")

    return (pp_bec_eig.calc_r2(), pp_bec_eig.calc_rmse())


def process_cv_metrics(sca_r2_list, sca_rmse_list, eig_r2_list, eig_rmse_list):
    metrics_df = pd.DataFrame({
        "fold": list(range(1, 6)),
        "sca_r2": sca_r2_list,
        "sca_rmse": sca_rmse_list,
        "eig_r2": eig_r2_list,
        "eig_rmse": eig_rmse_list
    })
    metrics_df.loc["mean"] = [
        "mean",
        np.mean(sca_r2_list),
        np.mean(sca_rmse_list),
        np.mean(eig_r2_list),
        np.mean(eig_rmse_list)
    ]
    metrics_df.loc["std"] = [
        "std",
        np.std(sca_r2_list),
        np.std(sca_rmse_list),
        np.std(eig_r2_list),
        np.std(eig_rmse_list)
    ]
    return metrics_df


def remove_u0_pred(df):
    remove_indices = df.index[df["pred"] <= 0].tolist()
    for idx in remove_indices:
        print(f"exclude (pred<=0): {df.loc[idx, 'matname']}")
    df = df.drop(index=remove_indices).reset_index(drop=True)

    return df
