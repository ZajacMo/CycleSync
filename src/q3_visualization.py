import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "data" / "output" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree

try:
    from src.font_config import configure_matplotlib_chinese_fonts
except ImportError:
    from font_config import configure_matplotlib_chinese_fonts

from src.fence_optimization import (
    CLEANED_DATA_PATH,
    OUTPUT_DIR,
    R_EARTH,
    DEFAULT_EPS_METERS,
    DEFAULT_MIN_SAMPLES,
    DEFAULT_R_FENCE,
    load_data,
    generate_candidates,
)


def configure_plot_style():
    sns.set_theme(context="paper", style="ticks", font_scale=1.1)
    configure_matplotlib_chinese_fonts()
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.titlepad": 10,
        }
    )


def project_to_local_meters(lon_deg, lat_deg, lon0_deg, lat0_deg):
    # 以研究区域中心点为基准做局部等距近似：把经纬度映射到“米”尺度平面坐标（便于标尺/等比例显示）
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    lon0 = np.radians(lon0_deg)
    lat0 = np.radians(lat0_deg)
    x = R_EARTH * (lon - lon0) * np.cos(lat0)
    y = R_EARTH * (lat - lat0)
    return x, y


def weighted_quantile(values, weights, q):
    # 加权分位数：用于“按需求权重”的 P50/P95 推行距离
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return np.interp(q, cw, v)


def add_scalebar(ax, length_axis, label):
    # 标尺长度以当前坐标轴单位计（本脚本绘图单位为 km）
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    pad_x = 0.04 * (x1 - x0)
    pad_y = 0.05 * (y1 - y0)

    x_start = x0 + pad_x
    x_end = x_start + length_axis
    y = y0 + pad_y

    ax.plot([x_start, x_end], [y, y], color="0.2", lw=2.2, solid_capstyle="butt")
    ax.plot([x_start, x_start], [y - 0.015 * (y1 - y0), y + 0.015 * (y1 - y0)], color="0.2", lw=2.2)
    ax.plot([x_end, x_end], [y - 0.015 * (y1 - y0), y + 0.015 * (y1 - y0)], color="0.2", lw=2.2)
    ax.text(x_start, y + 0.025 * (y1 - y0), label, ha="left", va="bottom", color="0.2")


def compute_nearest_fence_distances(demand_df, fences_df):
    # 需求点 i 的“推行距离”定义为到最近围栏点的球面距离 d(i, argmin_j d_ij)
    demand_rad = np.radians(demand_df[["y", "x"]].to_numpy())
    fences_rad = np.radians(fences_df[["y", "x"]].to_numpy())

    tree = BallTree(fences_rad, metric="haversine")
    dist_rad, idx = tree.query(demand_rad, k=1)
    return dist_rad[:, 0] * R_EARTH, idx[:, 0]


def plot_layout_density_with_fences(parking_points, fences, out_dir, r_fence_m, title_suffix):
    lon0 = float(parking_points["x"].mean())
    lat0 = float(parking_points["y"].mean())

    px, py = project_to_local_meters(parking_points["x"].to_numpy(), parking_points["y"].to_numpy(), lon0, lat0)
    fx, fy = project_to_local_meters(fences["x"].to_numpy(), fences["y"].to_numpy(), lon0, lat0)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    # hexbin：展示“停放事件密度”的空间分布；对数分箱缓解极端热点的压缩
    hb = ax.hexbin(px / 1000, py / 1000, gridsize=150, bins="log", cmap="viridis", mincnt=1)
    ax.scatter(
        fx / 1000,
        fy / 1000,
        s=14,
        marker="x",
        linewidths=0.6,
        color="black",
        alpha=0.75,
        label="电子围栏位置",
    )

    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label(r"$\log_{10}$(停放事件数)")

    ax.set_title(f"问题3：电子围栏布局（停放密度 × 选址）{title_suffix}")
    ax.set_xlabel("东西向相对距离 (km)")
    ax.set_ylabel("南北向相对距离 (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=True, framealpha=0.9, loc="upper right")
    ax.grid(False)

    add_scalebar(ax, length_axis=2.0, label="2 km")

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    circle_center = (x1 - 0.12 * (x1 - x0), y0 + 0.12 * (y1 - y0))
    r_km = r_fence_m / 1000
    ax.add_patch(plt.Circle(circle_center, r_km, fill=False, lw=1.4, ec="0.2"))
    ax.text(circle_center[0], circle_center[1] + 1.2 * r_km, f"R = {int(r_fence_m)} m", ha="center", va="bottom", color="0.2")

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "q3_viz_layout_density.png")
    fig.savefig(Path(out_dir) / "q3_viz_layout_density.pdf")
    plt.close(fig)


def plot_weighted_ecdf(dist_m, weights, out_dir, r_fence_m, title_suffix):
    # 加权 ECDF：y 轴为累计需求权重比例，衡量“用户便利性”（更小推行距离更好）
    order = np.argsort(dist_m)
    d = dist_m[order]
    w = weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]

    p50 = weighted_quantile(dist_m, weights, 0.5)
    p95 = weighted_quantile(dist_m, weights, 0.95)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(d, cw, color=sns.color_palette("viridis", 4)[2], lw=2.2)
    ax.axvline(p50, color="0.25", lw=1.2, ls="--", label=f"P50 = {p50:.0f} m")
    ax.axvline(p95, color="0.25", lw=1.2, ls=":", label=f"P95 = {p95:.0f} m")
    ax.axvline(r_fence_m, color="0.45", lw=1.2, ls="-.", label=f"覆盖半径 R = {int(r_fence_m)} m")

    ax.set_title(f"问题3：推行距离加权分布（ECDF）{title_suffix}")
    ax.set_xlabel("到最近电子围栏的距离 (m)")
    ax.set_ylabel("累计覆盖比例（按需求权重）")
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, framealpha=0.9, loc="lower right")
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "q3_viz_push_distance_ecdf.png")
    fig.savefig(Path(out_dir) / "q3_viz_push_distance_ecdf.pdf")
    plt.close(fig)


def plot_fence_load_distribution(load_weights, out_dir, title_suffix):
    # 围栏承载 load_j：以“需求点就近分配”统计每个围栏吸纳的需求权重（用于识别潜在过载）
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.log10(1 + load_weights)
    sns.histplot(x, bins=40, color=sns.color_palette("viridis", 6)[3], edgecolor="white", ax=ax)
    ax.set_title(f"问题3：围栏承载（就近分配的需求权重）分布{title_suffix}")
    ax.set_xlabel(r"$\log_{10}(1 + \mathrm{load}_j)$")
    ax.set_ylabel("围栏数量")
    sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "q3_viz_fence_load_distribution.png")
    fig.savefig(Path(out_dir) / "q3_viz_fence_load_distribution.pdf")
    plt.close(fig)

def render_q3_figures(
    cleaned_data_path,
    fences_csv,
    out_dir,
    eps_meters,
    min_samples,
    r_fence_m,
):
    configure_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parking_points = load_data(cleaned_data_path)
    fences = pd.read_csv(fences_csv)

    demand_points = generate_candidates(parking_points, eps_meters=eps_meters, min_samples=min_samples)
    dist_m, nearest_idx = compute_nearest_fence_distances(demand_points, fences)

    weights = demand_points["weight"].to_numpy()
    # 加权覆盖率：CR = sum_{i: d_i<=R} w_i / sum_i w_i（与文档中的全覆盖/α口径一致）
    cr_weighted = float(weights[dist_m <= r_fence_m].sum() / weights.sum())

    # 就近分配：围栏 j 的承载为被分配到 j 的需求权重之和（非容量约束，仅作诊断可视化）
    load_weights = np.bincount(nearest_idx, weights=weights, minlength=len(fences))

    title_suffix = f"\n(eps={eps_meters}m, min_samples={min_samples}, R={int(r_fence_m)}m, |S|={len(fences)}, CR={cr_weighted:.3f})"

    plot_layout_density_with_fences(parking_points, fences, out_dir, r_fence_m=r_fence_m, title_suffix=title_suffix)
    plot_weighted_ecdf(dist_m, weights, out_dir, r_fence_m=r_fence_m, title_suffix=title_suffix)
    plot_fence_load_distribution(load_weights, out_dir, title_suffix=title_suffix)



def parse_args():
    parser = argparse.ArgumentParser(description="问题3可视化：围栏布局与诊断图表")
    parser.add_argument("--cleaned", default=CLEANED_DATA_PATH, help="清洗后的订单数据 CSV")
    parser.add_argument("--fences", default=str(Path(OUTPUT_DIR) / "q3_fences.csv"), help="电子围栏选址结果 CSV")
    parser.add_argument("--out-dir", default=OUTPUT_DIR, help="输出目录")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS_METERS, help="候选/需求聚类 DBSCAN 半径 eps (m)")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES, help="候选/需求聚类 DBSCAN 最小样本数")
    parser.add_argument("--r", type=float, default=DEFAULT_R_FENCE, help="覆盖半径 R (m)")
    parser.add_argument("--s-rf", type=float, default=50.0, help="热力图切片：r_f (m)")
    parser.add_argument("--s-m", type=int, default=500, help="热力图切片：M")
    return parser.parse_args()


def main():
    args = parse_args()
    render_q3_figures(
        cleaned_data_path=args.cleaned,
        fences_csv=args.fences,
        out_dir=args.out_dir,
        eps_meters=args.eps,
        min_samples=args.min_samples,
        r_fence_m=args.r,
    )


if __name__ == "__main__":
    main()
