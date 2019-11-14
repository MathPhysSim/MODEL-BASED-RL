import time
import sys
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

import pyjapc
from tqdm import tqdm


class AWAKEcommunicator():
    int_injected = 'TT43.BPM.430010/Acquisition#sigmaAvePos'

    def __init__(self, twiss_file, **kwargs):
        twiss_header_names = pd.read_csv(twiss_file, skiprows=45,
                                         nrows=0, delim_whitespace=True).columns[1:]
        self.twiss = pd.read_csv(twiss_file, skiprows=47, delim_whitespace=True,
                                 names=twiss_header_names, header=None)

        self.bpms = self.get_bpms()
        self.correctors_h = self.get_correctors(plane='H')
        self.correctors_v = self.get_correctors(plane='V')
        self.measurement_data = pd.DataFrame()
        self.session_name = 'AWAKEcommunicator_' + str(datetime.datetime.now()) + '.h5'
        print('Logging to: ', self.session_name)
        self.twiss.to_hdf(self.session_name, key='twiss')

        self.current_set_corrector_settings = None
        # print('Stored in:', self.session_name)
        # print('BPMS', self.bpms)
        # print('correctors_h', self.correctors_h)
        # print('correctors_v', self.correctors_v)

        # self.japc = pyjapc.PyJapc('SPS.USER.SFTPRO2', noSet=True)
        self.devices_mal = ['rmi://virtual_awake/logical.RCIBV.412353/K', 'rmi://virtual_awake/logical.RCIBH.412353/K']
        # self.japc = pyjapc.PyJapc('SPS.USER.SFTPRO2', noSet=noSet)

        if 'noSet' in kwargs:
            noSet = kwargs.get('noSet')
        else:
            noSet=True

        if 'japc' in kwargs:
            self.japc = kwargs.get('japc')
        else:
            self.japc = pyjapc.PyJapc('', noSet=False, incaAcceleratorName=None)
        self.japc.rbacLogin()    
        self.tSelector = 'SPS.USER.AWAKE1'
        self.BDL = 0.00072
        self.cConst = 0.2998
        self.max_I = 10
        self.mom_str = 'rmi://virtual_sps/TT43BEAM/MOMENTUM'
        self.momentum = self.japc.getParam(self.mom_str, timingSelectorOverride='')

        self.COUNTER = 0

    ''' Check and wait for magnet current to be set '''

    def wait4magnet(self, magnet, k_value):
        corr_constant = (self.momentum * self.max_I) / (self.BDL * self.cConst)
        i_setting = (corr_constant * k_value)
        i_tolerance = 0.01
        magnet = magnet[:-1]+'I'
        i_actual = self.japc.getParam(magnet, timingSelectorOverride=self.get_ts)
        return np.abs(i_setting - i_actual) > i_tolerance

    def get_bpms(self):
        # bpms = self.twiss[self.twiss.NAME.apply(lambda x: x[:3]) == 'BPM']
        bpms = self.twiss[self.twiss.KEYWORD == 'MONITOR']
        return pd.concat([bpms, self.generate_japc_acquisition_bpm_names(bpms)], axis=1)

    def generate_japc_acquisition_bpm_names(self, bpms):
        data = pd.Series([], name='japcName')
        for index, row in bpms.iterrows():
            bpm_name = row.NAME
            if '.43' in bpm_name:
                acquisiton_name = 'TT43.' + bpm_name + '/Acquisition'
            else:
                acquisiton_name = 'TT41.' + bpm_name + '/Acquisition'
            data.loc[index] = acquisiton_name
        return data

    def get_correctors(self, plane='V'):
        correctors = self.twiss[self.twiss.KEYWORD == "KICKER"]
        return pd.concat([correctors,
                          self.generate_japc_acquisition_corrector_names(correctors=correctors, plane=plane)], axis=1)

    def generate_japc_acquisition_corrector_names(self, correctors, plane='V'):
        data = pd.Series([], name='japcName')
        for index, row in correctors.iterrows():
            bpm_name = row.NAME
            prefix = 'rmi://virtual_awake/logical.RCIB'
            suffix = '/K'
            acquisition_name = prefix + plane + '.' + bpm_name.split('.')[-1] + suffix
            data.loc[index] = acquisition_name
        return data

    def take_bpm_measurement(self, number_measurements, **kwargs):
        acquisition_names = self.bpms['japcName'].values.tolist()

        data_h = pd.DataFrame(columns=acquisition_names)
        data_v = pd.DataFrame(columns=acquisition_names)

        counter_time = 0
        max_time = 10
        time_over = False
        self.unique_measurement_id = 0
        shape_size = 0

        with tqdm(total=number_measurements, unit='measurements') as pbar:
            while not (time_over):
                try:
                    data_in, headers_in = self.japc.getParam(acquisition_names, getHeader=True)
                    transmissions = np.array([float(data['sigmaAvePos']) / 1 for data in data_in])
                    acqStamps_new = [header['acqStamp'] for header in headers_in]

                    if not any(transmissions == -1) and all(transmissions >= -0.4):
                        if data_h.shape[0] < 1:
                            data_h_temp = pd.DataFrame(np.array([data['horPos'] for data in data_in]),
                                                       index=acquisition_names).T.dropna()
                            data_h = pd.concat([data_h, data_h_temp])

                            data_v_temp = pd.DataFrame(np.array([data['verPos'] for data in data_in]),
                                                       index=acquisition_names).T.dropna()
                            data_v = pd.concat([data_v, data_v_temp])

                            acqStamps_old = acqStamps_new
                            counter_time = 0
                        elif all(np.array([acqStamps_new[i] != acqStamps_old[i]
                                           for i in range(len(acquisition_names))])):

                            data_h_temp = pd.DataFrame(np.array([data['horPos'] for data in data_in]),
                                                       index=acquisition_names).T.dropna()
                            data_h = pd.concat([data_h, data_h_temp], ignore_index=True)

                            data_v_temp = pd.DataFrame(np.array([data['verPos'] for data in data_in]),
                                                       index=acquisition_names).T.dropna()
                            data_v = pd.concat([data_v, data_v_temp], ignore_index=True)

                            acqStamps_old = acqStamps_new
                            counter_time = 0
                    pbar.update(data_h.shape[0] - shape_size)
                    shape_size = data_h.shape[0]

                    if (data_h.shape[0] >= number_measurements) and (data_v.shape[0] >= number_measurements):
                        break

                except:
                    e = sys.exc_info()[0]
                    print(e)
                    print('Failed to receive. Retrying...')

                time.sleep(1)
                counter_time += 1
            time_over = counter_time >= max_time
        try:

            return_data = pd.concat([data_h, data_v], keys=['horizontal', 'vertical'], axis=1)

            if 'measurement_id' in kwargs:
                measurement_id = kwargs.get('measurement_id')
            else:
                measurement_id = self.COUNTER
                self.COUNTER+=1

            self.store_to_disc_continuous(return_data, measurement_id=measurement_id)
            print('Data acquisition finished')
            return return_data
        except:
            e = sys.exc_info()[0]
            print('Failed')
            print(e)

    def store_to_disc_continuous(self, measured_data=None, measurement_id='new', **kwargs):
        if not (type(measurement_id) is str):
            measurement_id = 'default_' + str(measurement_id)
        if not (measured_data is None):
            measured_data.to_hdf(self.session_name, key=measurement_id + '/bpm_meas')
        if not (self.current_set_corrector_settings is None):
            self.current_set_corrector_settings.to_hdf(self.session_name, key=measurement_id + '/corr_set')

        current_corrector_values = (self.get_corrector_values(pd.concat([self.get_corrector_japc_names(plane='H'),
                                                self.get_corrector_japc_names(plane='V')]).values.tolist(), **kwargs))

        # current_corrector_values.columns = list([name+'_hor' for name in self.correctors_h['NAME']]
        # +[name+'_ver' for name in self.correctors_v['NAME']])
        # # print(list([name+'_hor' for name in self.correctors_h['NAME']]
        # # +[name+'_ver' for name in self.correctors_v['NAME']]))
        # print(current_corrector_values.T)
        current_corrector_values.to_hdf(self.session_name, key= measurement_id + '/corr_val')

    def push_to_disc(self, data, **kwargs):
        if 'key' in kwargs:
            data.to_hdf(self.session_name, key=kwargs.get('key'))
        else:
            data.to_hdf(self.session_name, key='scan_data')

    def get_corrector_values(self, element_list, **kwargs):
        data = pd.DataFrame(columns=element_list)
        for device in element_list:
            data.loc['get', device] = self._getVirtualParameter(device=device, **kwargs)
        return data

    def set_corrector_values(self, element_value_dict, **kwargs):
        self.current_set_corrector_settings = pd.DataFrame(element_value_dict, index=['set'])
        for device, value in element_value_dict.items():
            self._setVirtualParameter(device=device, value=value, **kwargs)
        count = 0
        # while any([self.wait4magnet(device, value) for device, value in element_value_dict.items()]):
        #     time.sleep(.25)
        #     if count > 100:
        #         break
        #     count += 1

    def get_corrector_japc_names(self, plane='V'):
        if plane == 'V':
            return self.correctors_v['japcName']
        else:
            return self.correctors_h['japcName']

    def get_corrector_names(self, plane='V'):
        if plane == 'V':
            return self.correctors_v['NAME']
        else:
            return self.correctors_h['NAME']

    def get_bpm_names(self, plane='V'):
        if plane == 'V':
            return [value + '_vertical' for index, value in self.bpms['NAME'].items()]
        else:
            return [value + '_horizontal' for index, value in self.bpms['NAME'].items()]

    def get_stats(self, number_measurement):
        data = self.take_bpm_measurement(number_measurements=number_measurement)

    def _setVirtualParameter(self, device, value, test=False):
        if test:
            print(device, ' test: ', value)
        elif device in self.devices_mal:
            pass
        else:
            java_return = self.japc.getParam(device, noPyConversion=True, timingSelectorOverride='').getValue()
            java_return.double = value
            self.japc.setParam(device, java_return, timingSelectorOverride='')

    def _getVirtualParameter(self, device, test=False):
        if test:
            return 0
        elif device in self.devices_mal:
            return np.nan
        else:
            java_return = self.japc.getParam(device, noPyConversion=True, timingSelectorOverride='').getValue()
            return java_return.double


if __name__ == '__main__':
    awake_communicator = AWAKEcommunicator('../InfrastructuralData/electron_tt43.out', noSet=True)
    data = awake_communicator.take_bpm_measurement(35)

    # print(data.T)
    print(data.describe().loc['mean'].values)

    names_correctors_h = awake_communicator.get_corrector_japc_names(plane='H').iloc[:]
    print(names_correctors_h)
#    awake_communicator._setVirtualParameter(device = 'rmi://virtual_AWAKE/TT43BEAM/MOMENTUM',
#                                             value=1.)
    for i in range(3):
        elements_values = dict([(i, 0.0001*np.random.rand()) for i in names_correctors_h])
        awake_communicator.set_corrector_values(element_value_dict=elements_values, test=False)
        out = awake_communicator.take_bpm_measurement(number_measurements=25, measurement_id='nr' + str(i))
        print('ouput', out)
