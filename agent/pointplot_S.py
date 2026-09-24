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
data["DDT_label"] = data["DDT"].apply(lambda x: f"DDT={x:.0f}" if pd.notnull(x) else np.nan)
data = data.sort_values("DDT")
fig, ax = plt.subplots(figsize=(5, 4), dpi=100, facecolor='w')

sns.lineplot(
    x="S", y="avg_decision_time",
    hue='DDT_label',
    style="DDT_label",
    data=data,
    palette=colors,
    linewidth=1.5,
    markers=marker_styles,
    dashes=line_styles,
    ax=ax
)
ax.set_xlabel("S", fontsize=12)
ax.set_ylabel("Avg Decision Time (s)", fontsize=12)
ax.set_xticks([50, 100, 150])
leg = ax.legend(
    fontsize=10,
    loc='upper right',
    bbox_to_anchor=(1, 1),
    borderaxespad=0,
    frameon=False
)
for handle in leg.legend_handles:
    handle.set_alpha(0.7)
for text in leg.get_texts():
    text.set_alpha(0.7)
plt.tight_layout()

fig.savefig('../photo/compareS.png', bbox_inches='tight', dpi=300)
# plt.show()
