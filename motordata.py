import pandas as pd
from openpyxl import load_workbook
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from os import path


data = pd.read_excel('propstuff/dbt_inventory.xlsx', sheet_name='Motor')
#data.head(26) # -1 for max?

eta_m = lambda omega, kv, R, i0, v: (1 - (i0*R)/(v - omega / kv))*(omega / (v*kv))
Q_m = lambda omega, kv, R, i0, v: ((v - (omega / kv)) * (1 / R) - i0) * (1 / kv)
Pshaft = lambda omega, kv, R, i0, v: ((v - (omega / kv)) * (1 / R) - i0) * (omega / kv)

num = 26 ## number of entries in the excel file
motor_names = np.array(data['Model'])[0:num]
lines = np.array([i for i in range(num)])
kv = np.array(data['KV [RPM/V]'])[0:num] *(2*np.pi/60) # equation requires rad/s/volt
R = np.array(data['Resistance [Ohms]'])[0:num]
i0 = np.array(data['No load current at 10 V [A]'])[0:num]

rgb = ['blue', 'red', 'black', 'green', 'cyan']

functions = {'eta': eta_m, 'Q': Q_m, 'Pshaft': Pshaft}

def store(database, sheet):
    f = 'propstuff/motordata.xlsx'
    if not path.exists(f):
        writer = pd.ExcelWriter(f, engine='xlsxwriter')
        df = pd.DataFrame(data, index=[0])
        df.to_excel(writer, sheet_name= sheet, index=False)
        writer.save()
    else:
        df = pd.DataFrame(data, index=[0])
        writer = pd.ExcelWriter(f, engine='openpyxl')
        writer.book = load_workbook(f)
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        reader = pd.read_excel(f)
        df.to_excel(writer, index=False, header=False, startrow=len(reader)+1, sheet_name=sheet)
        writer.close()

def prepare_m(Q_required, omega_required, max_volts):

    v_required = lambda kv, R, i0, Q_required, omega_required: \
        (kv * Q_required + i0)*R + omega_required/kv

    database = {}

    for line in lines[0:num]: ## to keep it brief for now!!!
        if kv[line] is str:
            pass
        elif R[line] is str:
            pass
        elif i0[line] is str:
            pass
        else:
            volts = v_required(kv=kv[line], R=R[line], i0=i0[line],\
                    Q_required=Q_required, omega_required=omega_required)

            if volts[0] < max_volts:
                database[motor_names[line]] = {}
                database[motor_names[line]]['voltage_required'] = volts[0]

                for fun in functions:
                    #plt.figure(figsize=(8,8))
                    #plt.title(motor_names[line])
                
                    omega = np.linspace(0, kv[line]*(volts[0] - i0[line]*R[line]), 1000)

                    func = np.array(functions[fun](omega=omega, kv = kv[line], R=R[line], i0=i0[line], v=volts[0]))

                    #plt.plot(omega, func, color=rgb[j], label= 'v = ' + str(v[j]))
                    #plt.xlabel('rps')
                    #plt.ylabel(fun)

                    #database[motor_names[line]]['omega'] = omega
                    #database[motor_names[line]][fun] = func

                    if fun == 'eta':
                        eta_max = np.nanmax(func)
                        rps_max = omega[np.where(func == eta_max)][0]
                        database[motor_names[line]]['eta_max_m'] = eta_max
                        database[motor_names[line]]['rpm_max_m'] = rps_max / (2*np.pi/60)
                        #plt.text(rps_max, np.nanmax(func), \
                                #'omega = {:.2f}'.format(rps_max) + '\n' + 'eta = {:.2f}'.format(eta_max))
                    #plt.legend()
        #store(database, motor_names[line]) needs fixing
    return database
        