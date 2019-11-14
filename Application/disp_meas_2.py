import pyjapc
import matplotlib.pyplot as plt
import pickle
import datetime
import numpy as np
import time
import datetime
import sys
#import acc_library as al
"""
The script is controlled using command line arguments.
[1] "meas" or "pickle_file_name.p" => if meas then measurement is done, otherwise analysis only
[2] Best momentum guess in MeV/c
[3] Number of steps to do
[4] Number of measurements per momentum step to take
[5] Full range to span in %, e.g. 2 means that the scan will be done +/- 1% the best momentum guess given

"""

def errorbar(x, y, yerr=None, xerr=None, fmt='o', ecolor=None, elinewidth=0.5, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, mfc='w', **kwargs):

    return plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt=fmt, ecolor=ecolor, elinewidth=elinewidth, capsize=capsize, barsabove=barsabove, lolims=lolims, uplims=uplims, xlolims=xlolims, xuplims=xuplims, errorevery=errorevery, capthick=capthick, mfc=mfc, **kwargs)

def setVirtualParameter(device, value):
    ele = japc.getParam(device, noPyConversion=True, timingSelectorOverride='').getValue()
    ele.double = value
    japc.setParam(device, ele, timingSelectorOverride='')
    print(device + ' set to: ' + str(value))
    

def dict_fromkeys(list_ini, array_ini):
    return {key: array_ini.copy() for key in list_ini}

def get_twiss_dic(file_name):
    """
    Gets a twiss and a list of twiss columns as input and gives a matrix where each
    column represents a twiss column in the given order

    @param file_name: twiss file name
    @return: matrix of the asked twiss columns
    """

    beam_info = {}

    with open(file_name, 'r') as ff:
        for i in range(1, 45):
            line = ff.readline().split()
            try:
                beam_info[line[1]] = float(line[3])
            except:
                beam_info[line[1]] = line[3]

        next(ff)

        names = next(ff).split()
        del names[0]

        next(ff)

        data = np.genfromtxt(file_name, skip_header=47, names=names, dtype=None)
    return data, beam_info

def getBPM_pos(bpm_name):
    if '.43' in bpm_name:
        tl_name = tt43
    else:
        tl_name = tt41
    while(True):
        try:
            i_initial = japc.getParam(int_injected)
            bpm_data = japc.getParam(tl_name + bpm_name + '/Acquisition')
            transmission = float(bpm_data[sigmaAve]) / 1 # i_initial
        except:
            transmission = -1

        if transmission >= -0.4:
            print(bpm_name + ' position = ' + str(bpm_data['hor' + avePos]) + ' ver = ' + str(bpm_data['ver' + avePos]))
            return (bpm_data['hor' + avePos], bpm_data['ver' + avePos])

def getMeanStd(bpms, numMeas):
    
    # array => H | V
    dic_meas = dict_fromkeys(bpms, np.zeros((numMeas, 2)))
    for i in range(numMeas):
        
        for bpm_name in bpms:
        
            dic_meas[bpm_name][i, :] = getBPM_pos(bpm_name)
        time.sleep(1.3)
    
    # array =>      | H | V
    #          -------------
    #          Mean |.....
    #          -------------
    #          Std  |.....
    #          -------------
    dic_reas = dict_fromkeys(bpms, np.zeros((2, 2)))
    for bpm_name in bpms:
        dic_reas[bpm_name][0, :] = np.mean(dic_meas[bpm_name], axis=0)
        dic_reas[bpm_name][1, :] = np.std(dic_meas[bpm_name], axis=0)

    return dic_reas


twiss, _ = get_twiss_dic('../InfrastructuralData/electron_tt43.out')
names = []

for i in range(len(twiss['NAME'])):
    names.append(twiss['NAME'][i].decode('UTF-8'))

bpms = []
for ele in names:
    if 'BPM' in ele:
        bpms.append(ele.strip('"'))

bpm_h_achr = 'BPM.430308'

bpm_v_achr_1 =  'BPM.430103'
bpm_v_achr_2 =  'BPM.430129'


if sys.argv[1] == 'meas':
    mom_nominal = float(sys.argv[2]) * 1e-3

    measNum = int(sys.argv[4])

    steps = int(sys.argv[3])

    max_percent = float(sys.argv[5]) / 2.
    percent_vector = np.linspace(-1e-2 * max_percent, 1e-2 * max_percent, steps)
    mom_vector = mom_nominal * (1 + percent_vector)

# # of meas | Mean | Std
# -----------------------
#  1...     | <>   | std()

    results = {'h': None, 'v': None}
    results['h'] = dict_fromkeys(bpms, np.zeros((steps, 2)))
    results['v'] = dict_fromkeys(bpms, np.zeros((steps, 2)))

    japc = pyjapc.PyJapc('SPS.USER.SFTPRO2')

    momentum = 'rmi://virtual_sps/TT43BEAM/MOMENTUM'

# for transmission sigmaAvePos

    acquisition = '/Acquisition'
    avePos = 'AvePos'
    sigmaAve = 'sigmaAvePos'
    tt43 = 'TT43.'
    tt41 = 'TT41.'


    int_injected = 'TT43.BPM.430010/Acquisition#sigmaAvePos'

    # japc.rbacLogin()


    for i, mom in enumerate(mom_vector):
        
        setVirtualParameter(momentum, mom)
        time.sleep(20)

        dic_res = getMeanStd(bpms, measNum)

        for bpm_name in bpms:
            results['h'][bpm_name][i] = dic_res[bpm_name][:, 0]
            results['v'][bpm_name][i] = dic_res[bpm_name][:, 1]

    print(results)

    now = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')

    pickle.dump((mom_vector, results), open('results_' + now + '.p', 'wb'), 2)

else:
    mom_vector, results = pickle.load(open(sys.argv[1], 'rb'))

plt.figure(1)
plt.title('Momentum matching')

a = []
b = []

for bpm, bety in zip([bpm_v_achr_1, bpm_v_achr_2], [1.1895, 13.171]):

    p = []
    not_nan = ~np.isnan(results['v'][bpm][:, 0])
    p0 = mom_vector[not_nan]
    y1 = results['v'][bpm][:, 0][not_nan] / np.sqrt(bety)
    
    coef = np.polyfit(p0, y1, 1)
    a.append(coef[-1])
    b.append(coef[-2])
    f = np.poly1d(coef)

    plt.figure(1)
    pl0 = errorbar(mom_vector, results['v'][bpm][:, 0], results['v'][bpm][:, 1], fmt='-o', label=bpm)

    plt.figure(2)
    if bpm == bpm_v_achr_1:
        plt.plot(mom_vector[not_nan], f(mom_vector[not_nan]), c=pl0[0].get_color())
        errorbar(mom_vector, results['v'][bpm][:, 0] / np.sqrt(bety), np.abs(results['v'][bpm][:, 1] / np.sqrt(bety)), c=pl0[0].get_color(), fmt='o', label=bpm)
    else:
        plt.plot(mom_vector[not_nan], -1 * f(mom_vector[not_nan]), c=pl0[0].get_color())
        errorbar(mom_vector, -1 * results['v'][bpm][:, 0] / np.sqrt(bety), np.abs(results['v'][bpm][:, 1] / np.sqrt(bety)), c=pl0[0].get_color(), fmt='o', label=bpm)

momentum_est = -1 * (a[0] + a[1]) / (b[1] + b[0])

setVirtualParameter(momentum, momentum_est)

print('Momentum of the line: %.3f MeV/c'%(momentum_est * 1e3))
plt.axvline(momentum_est, ls='--', label='p = %.3f MeV/c'%(momentum_est * 1e3))
plt.legend(frameon=True)
plt.ylabel('y / sqrt(beta y) / m1/2')
plt.xlabel('Momentum line / GeV')

plt.figure(1)

plt.axhline(0, ls='--')
plt.xlabel('Momentum line / GeV')
plt.ylabel(r'V Trajectory / mm')

plt.legend()
plt.show()


