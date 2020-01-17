import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

# specify your path of directory


# call listdir() method
# path is a directory of which you want to list

# This would print all the files and directories
emps =[]
index = []




data_folder = 'data_25_25_pure_policy_50_acquisition_long/'

directories = os.listdir(data_folder)
directories.sort()

nr=0
for file in directories:
    if 'Awake_test' in file and not 'init' in  file:
        nr+=1
        try:
            pickle_off = open(data_folder + file, "rb")
            emp = pickle.load(pickle_off)
            emps.append(emp)
            pickle_off.close()
            print('Added ', file)
        except:
            print('Failed on ',file )
        # ep = int(file[11:file.find('_',11)])
        # it = int(file[file.find('_',11)+1:file.find('.',file.find('_',11))])
        # index.append(ep*it)
        # print(ep, it)
# print(emps)
df_awake = pd.concat(emps, keys=range(nr)).reset_index()

emps =[]
index = []
nr = 0
for file in directories:
    if 'Awake_model' in file:
        nr+=1
        try:
            pickle_off = open(data_folder + file, "rb")
            emp = pickle.load(pickle_off)
            emps.append(emp)
            pickle_off.close()
            print('Added ', file)
        except:
            print('Failed on ',file )
        # ep = int(file[11:file.find('_',11)])
        # it = int(file[file.find('_',11)+1:file.find('.',file.find('_',11))])
        # index.append(ep*it)
        # print(ep, it)

pickle_off = open(data_folder + 'ep_data.pkl', "rb")
ep_data = pd.DataFrame(pickle.load(pickle_off))
ep_data.columns = ['ep', 'it', 'nr data points']
# pickle_off.close()

df_model = pd.concat(emps, keys=range(nr)).reset_index()
# print(df)

fig, axs = plt.subplots(3, sharex=True)

ax1=axs[0].twinx()
sns.lineplot(x="level_0", y="Ep. rews", data=df_awake, ax=ax1, color="blue")
ax = sns.lineplot(x="level_0", y="Ep. rews", hue="level_1", data=df_model, ax=axs[0])

ax1=axs[1].twinx()
sns.lineplot(x="level_0", y="Ep. lens", data=df_awake, ax=ax1, color="blue")
ax = sns.lineplot(x="level_0", y="Ep. lens", hue="level_1", data=df_model, ax=axs[1])
plt.grid(True)
# sns.lineplot(x="level_0", y="Ep. lens", data=df_model, ax=ax1 , hue="level_1")

# df_model.groupby('level_0').mean()['Ep. success'].plot(ax=axs[1])
# sns.lineplot(x="level_0", y="Ep. lens", hue="level_1", data=df_awake, ax=ax1)
# ax = sns.lineplot(x="level_0", y="Ep. rews", data=df_awake, ax=ax)

df_awake.groupby('level_0').mean()['Ep. success'].plot(ax=axs[2])
ax = axs[2].twinx()
ep_data['nr data points'].plot(ax=ax,drawstyle="steps" , color='lime')
plt.grid(True)

plt.savefig(data_folder+'progress_model.pdf')
plt.show()

# # #
# # # fmri = sns.load_dataset("fmri")
# # print(df)
fig, axs = plt.subplots(2, sharex=True)
sns.lineplot(x="level_0", y="Ep. lens", data=df_awake, ax=axs[0])
plt.grid(True)
ax=axs[0].twinx()
sns.lineplot(x="level_0", y="Ep. rews", data=df_awake, ax=ax, color="coral")
plt.grid(True)

df_awake.groupby('level_0').mean()['Ep. success'].plot(ax=axs[1])
ax = axs[1].twinx()
ep_data['nr data points'].plot(ax=ax,drawstyle="steps" , color='lime')
plt.grid(True)
# #
plt.savefig(data_folder+'progress_awake.pdf')
# plt.show()
#
plt.show()