import h5py
import pandas as pd
import numpy as np
import os
#
dir = "../../DATAAWAKE/"
element_state_list = ['BPM.430028_horizontal',
                      'BPM.430039_horizontal',
                      'BPM.430103_horizontal',
                      'BPM.430129_horizontal',
                      'BPM.430203_horizontal',
                      'BPM.430308_horizontal',
                      'BPM.412343_horizontal',
                      'BPM.412345_horizontal',
                      'BPM.412347_horizontal',
                      'BPM.412349_horizontal',
                      'BPM.412351_horizontal',
                      'BPM.430028_vertical',
                      'BPM.430039_vertical',
                      'BPM.430103_vertical',
                      'BPM.430129_vertical',
                      'BPM.430203_vertical',
                      'BPM.430308_vertical',
                      'BPM.412343_vertical',
                      'BPM.412345_vertical',
                      'BPM.412347_vertical',
                      'BPM.412349_vertical',
                      'BPM.412351_vertical']


element_action_list = ['horizontal412344/K',
       'horizontal412345/K',
       'horizontal412347/K',
       'horizontal412349/K',
       'horizontal412353/K',
       'horizontal430029/K',
       'horizontal430040/K',
       'horizontal430104/K',
       'horizontal430130/K',
       'horizontal430204/K',
       'horizontal430309/K',
       'vertical412344/K',
       'vertical412345/K',
       'vertical412347/K',
       'vertical412349/K',
       'vertical412353/K',
       'vertical430029/K',
       'vertical430040/K',
       'vertical430104/K',
       'vertical430130/K',
       'vertical430204/K',
       'vertical430309/K']

# element_action_list = element_action_list[:11]
# element_state_list = element_state_list[:11]

count = 0
data_all = pd.DataFrame()

for file in os.listdir(dir)[:16]:
    file_name = dir + file
    # print(file_name)
    for i in range(2,10):
        key = 'default_' + str(i)
        # frameb = pd.read_hdf(file_name, key=key + '/corr_val').T
        # frameb.index = element_action_list
        # frameb = frameb.loc[element_actiond_list[:11]]
        # print(frameb)
        try:
            key = 'default_' + str(i)
            framea = pd.read_hdf(file_name, key=key + '/bpm_meas').T
            framea.index = element_state_list
            framea = framea.iloc[element_state_list[:11]]
            # print(framea)
            frameb = pd.read_hdf(file_name, key=key + '/corr_val').T
            # frameb.index = element_action_list
            # frameb = frameb.loc[element_action_list[:11]]
            print(frameb)
            frame = pd.concat([framea, frameb])
            frame.loc[frameb.index, :] = frameb.values

            print(frame)
            # del frame['set']
            data_all = pd.concat([data_all, frame], axis=1)
            count += 1

        except:
            pass
print(data_all.index)

# data_all.index[0:len(element_state_list)] = element_state_list
data_all.columns = [i for i in range(data_all.shape[-1])]
data_all.to_hdf('awake_data.h5', key='data')





data_all = pd.read_hdf('awake_data.h5', key='data')

data_all[data_all.iloc[:22,:]<1e-20]=np.nan
#
#
# data_all.dropna().to_hdf('awake_data_clean.h5', key='data')
# for i in range(3):
#     print(pd.read_hdf('DATA/AWAKEcommunicator_2019-09-19 17:24:17.062747.h5', key='nr'+str(i)+'/corr_val').T)
# file_name = dir+'AWAKEcommunicator_2019-09-19 17:24:17.062747.h5'

# print(pd.read_hdf(file_name, key='nr'+str(0)+'/bpm_meas').T)

# data_all = pd.read_hdf('awake_data_clean.h5', key='data')

print(data_all)