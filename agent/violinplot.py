from mailcap import subst

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ptitprince as pt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = .8
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["xtick.minor.visible"] = False
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["xtick.top"] = False
plt.rcParams["ytick.right"] = False
colors = ["#2FBE8F", "#459DFF", "#FF5B98", "#FFCC37", "#6A4C93", "#F2600C", "#A3A500", "#00A3E0"]
data = pd.read_csv("../result/compare1/compare11.csv")

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100, facecolor='w')

# 云雨图
dx = "algorithm"
dy = "completion_rate"
ort = "v"
pal = colors
sigma = 0.2

pt.RainCloud(
    x=dx,
    y=dy,
    data=data,
    palette=pal,
    bw=sigma,
    width_viol=0.6,
    ax=ax,
    orient=ort,
    move=0.2
)
ax.set_xlabel("Algorithm", fontsize=15)
ax.set_ylabel("Completion Rate", fontsize=15)
"""
order = data["algorithm"].unique()

# 云图
sns.violinplot(
    x="algorithm", y="completion_rate",
    data=data,
    order=order,
    palette=colors,
    cut=0,
    inner=None,
    linewidth=1.2,
    width=0.8,
    ax=ax
)
# 箱体图
sns.boxplot(
    x="algorithm", y="completion_rate",
    data=data,
    order=order,
    width=0.2,
    showcaps=True,
    showfliers=False,
    boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=1.2),
    medianprops=dict(color="black", linewidth=1.2),
    ax=ax)
# 雨图
sns.stripplot(
    x="algorithm", y="completion_rate",
    data=data,
    order=order,
    color="k",
    size=3,
    jitter=0.15,
    alpha=0.6,
    ax=ax
)
# 折线图

means = data.groupby("algorithm")["completion_rate"].mean().reindex(order)
x_pos = np.arange(len(order))
ax.plot(
    x_pos, means.values,
    color="red",
    linewidth=2,
    marker="o",
    markersize=5,
    zorder=10
)

ax.set_xlabel("Algorithm", fontsize=15)
ax.set_ylabel("Completion Rate", fontsize=15)
ax.set_xticklabels(order, fontsize=12)
"""

plt.tight_layout()

fig.savefig('../photo/compare1/1200_150.png', bbox_inches='tight', dpi=300)
plt.show()