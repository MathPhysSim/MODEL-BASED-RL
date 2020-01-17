import os
import pickle

dir = 'data copy/'
directories = os.listdir(dir)
directories.sort()
pickle_off = open('data copy/Awake_model_0_0.pkl', "rb")
ep_data = pickle.load(pickle_off)

nr = 0
emps = []
for file in directories:
    if 'Awake_test' in file:
        nr += 1
        try:
            pickle_off = open(dir + file, "rb")
            emps.append(pickle.load(pickle_off))
            # pickle_off.close()
        except:
            print('Failed on ', file)
