import os

dir = 'data/'
directories = os.listdir(dir)
prefix = 'Awake_model'
for file in directories:
    if prefix in file:
        name_split = file.split('_')
        ep = int(name_split[2])
        it = int(name_split[-1].split('.')[0])
        # print(ep, it)
        if it < 10:
            text = '00' + str(it)
        elif it < 100:
            text = '0' + str(it)
        name = prefix + '_' + str(ep) + '_' + text + '.pkl'
        print(name)
        os.rename(dir+file, dir+name)
