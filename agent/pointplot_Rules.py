import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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
colors = ["#2FBE8F", "#459DFF", "#FF5B98", "#FFCC37", "#6A4C93", "#F2600C"]
data = pd.read_csv("../result/compare2/compare2_1.csv")
algos = sorted(data["algorithm"].unique())
marker_list = ['o', '^', 's', 'D', 'X' ,'P' ]
dash_list = [(1,0), (5,2), (2,2), (3,1,1,1), (5,1,1,1), (1,1)]
marker_map = dict(zip(algos, marker_list))
dash_map = dict(zip(algos, dash_list))
fig, ax = plt.subplots(figsize=(10, 5), dpi=100, facecolor='w')

sns.lineplot(
    x="instances", y="avg_decision_time",
    hue='algorithm',
    style='algorithm',
    data=data,
    palette=colors,
    linewidth=1.5,
    markers=marker_map,
    #markers=True,
    dashes=dash_map,
    #dashes=True,
    ax=ax
)
ax.set_xlabel("")
ax.set_ylabel("Avg Decision Time (s)", fontsize=15)
leg = ax.legend(
    title="Algorithm",
    fontsize=10,
    loc='upper right',
    bbox_to_anchor=(1, 1),
    borderaxespad=0,
    frameon=False
)
#图例透明
#leg.get_frame().set_alpha(0.0)
"""#图例线条透明
for handle in leg.legend_handles:
    handle.set_alpha(0.7)
#图例内容透明
for text in leg.get_texts():
    text.set_alpha(0.7)
"""
plt.xticks(rotation=60)
plt.tight_layout()

fig.savefig('../photo/compare2_3.png', bbox_inches='tight', dpi=300)
# plt.show()