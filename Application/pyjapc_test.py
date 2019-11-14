import pyjapc
import numpy as np

japc = pyjapc.PyJapc('', noSet=True, incaAcceleratorName=None)
japc.rbacLogin()
print(japc.rbacGetToken())

acquisition_names = ['TT43.BPM.430028/Acquisition', 'TT43.BPM.430039/Acquisition', 'TT43.BPM.430103/Acquisition',
                     'TT43.BPM.430129/Acquisition', 'TT43.BPM.430203/Acquisition', 'TT43.BPM.430308/Acquisition',
                     'TT41.BPM.412343/Acquisition', 'TT41.BPM.412345/Acquisition', 'TT41.BPM.412347/Acquisition',
                     'TT41.BPM.412349/Acquisition', 'TT41.BPM.412351/Acquisition']

# for device in acquisition_names:
#     print('device', device)
#     print(japc.getParam(device, noPyConversion=True, timingSelectorOverride='').getValue())
# data_in, headers_in = japc.getParam(acquisition_names, getHeader=True)
# transmissions = np.array([float(data['sigmaAvePos']) / 1 for data in data_in])
# acqStamps_new = [header['acqStamp'] for header in headers_in]
#
acquisition_names = ['rmi://virtual_awake/logical.RCIBH.430029/K', 'rmi://virtual_awake/logical.RCIBH.430040/K',
                     'rmi://virtual_awake/logical.RCIBH.430104/K', 'rmi://virtual_awake/logical.RCIBH.430130/K',
                     'rmi://virtual_awake/logical.RCIBH.430204/K', 'rmi://virtual_awake/logical.RCIBH.430309/K',
                     'rmi://virtual_awake/logical.RCIBH.412344/K', 'rmi://virtual_awake/logical.RCIBH.412345/K',
                     'rmi://virtual_awake/logical.RCIBH.412347/K', 'rmi://virtual_awake/logical.RCIBH.412349/K',
                     'rmi://virtual_awake/logical.RCIBV.430029/K', 'rmi://virtual_awake/logical.RCIBH.412353/K',
                     'rmi://virtual_awake/logical.RCIBV.430040/K', 'rmi://virtual_awake/logical.RCIBV.430104/K',
                     'rmi://virtual_awake/logical.RCIBV.430130/K', 'rmi://virtual_awake/logical.RCIBV.430204/K',
                     'rmi://virtual_awake/logical.RCIBV.430309/K', 'rmi://virtual_awake/logical.RCIBV.412344/K',
                     'rmi://virtual_awake/logical.RCIBV.412345/K', 'rmi://virtual_awake/logical.RCIBV.412347/K',
                     'rmi://virtual_awake/logical.RCIBV.412349/K', 'rmi://virtual_awake/logical.RCIBV.412353/K']

mal_device = ['rmi://virtual_awake/logical.RCIBV.412353/K', 'rmi://virtual_awake/logical.RCIBH.412353/K']

for device in mal_device:
    # print(mal_device)
    acquisition_names.remove(device)

for device in acquisition_names:
    print('device', device)
    value = japc.getParam(device, noPyConversion=True, timingSelectorOverride='').getValue()
    japc.setParam(device, value)
