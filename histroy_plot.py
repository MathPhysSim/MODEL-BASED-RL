import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


fmri = sns.load_dataset("fmri")
print(fmri)
ax = sns.lineplot(x="index", y="signal", data=fmri)

plt.show()