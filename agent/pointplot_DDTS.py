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
colors = ["#2FBE8F", "#459DFF", "#FF5B98", "#FFCC37", "#6A4C93", "#00A3E0", "#F2600C", "#A3A500"]
data = pd.read_csv("../result/compare1/compareDDT_S.csv")
marker_styles = ['o', '^', 's', 'D', 'P', 'X' ]
line_styles = [(1,0), (5,2), (2,2), (3,1,1,1), (5,1,1,1), (1,1)]
data["S_label"] = data["S"].apply(lambda x: f"S={x:.0f}" if pd.notnull(x) else np.nan)
data = data.sort_values("S")
fig, ax = plt.subplots(figsize=(5, 4), dpi=100, facecolor='w')

sns.lineplot(
    x="DDT", y="avg_decision_time",
    hue='S_label',
    style="S_label",
    data=data,
    palette=colors,
    linewidth=1.5,
    markers=marker_styles,
    #markers=True,
    dashes=line_styles,
    #dashes=True,
    ax=ax
)
ax.set_xlabel("DDT", fontsize=12)
ax.set_ylabel("Avg Decision Time (s)", fontsize=12)
# ax.set_xticks([50, 100, 150])
ax.set_xticks([500, 600, 900, 1200, 1600])
leg = ax.legend(
    fontsize=10,
    loc='upper left',
    bbox_to_anchor=(0, 1),
    borderaxespad=0,
    frameon=False
)
#图例透明
#leg.get_frame().set_alpha(0.0)
# #图例线条透明
for handle in leg.legend_handles:
    handle.set_alpha(0.7)
#图例内容透明
for text in leg.get_texts():
    text.set_alpha(0.7)
#  ax.legend(title="Algorithm", fontsize=10)
# plt.xticks(rotation=60)
plt.tight_layout()

fig.savefig('../photo/compareDDT.png', bbox_inches='tight', dpi=300)
# plt.show()