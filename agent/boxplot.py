from mailcap import subst

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
colors = ["#2FBE8F", "#459DFF", "#FF5B98", "#FFCC37", "#6A4C93", "#F2600C", "#A3A500", "#00A3E0"]
data = pd.read_csv("../result/compare3/compare3_15.csv")

# fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100, facecolor='w')
fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100, facecolor='w')
# instances = ['DDT500_num50']#'DDT500_num100','DDT500_num150', 'DDT600_num50'

sns.boxplot(x="algorithm", y="completion_rate", hue='algorithm', data=data, palette=colors, saturation=1, width=.7, linewidth=1.2, ax=ax)
ax.set_xlabel("Algorithm", fontsize=15)
ax.set_ylabel("Completion Rate", fontsize=15)
ax.legend(title="Algorithm", fontsize=10)
plt.tight_layout()

fig.savefig('../photo/compare3_15.png', bbox_inches='tight', dpi=300)
plt.show()