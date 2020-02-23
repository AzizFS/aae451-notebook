#%%
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import prop_uiuc
import pandas as pd
from openpyxl import load_workbook
from os import path
import subprocess as sb
import motordata as md

prop_db = prop_uiuc.parse_database(False)
#%%

class SymTable(dict):
    '''
    A bookkeeping class for symbolic variables.
    '''
    
    def declare(self, name: str, desc: str, value: float, lower: float, upper: float, unit: str):
        sym = ca.MX.sym(name)
        self[sym] = {
            'name': name,
            'desc': desc,
            'value': value,
            'lower': lower,
            'upper': upper,
            'unit': unit
        }
        return sym


syms = SymTable()

# variables
r_prop_in = syms.declare(name='r_prop_in', desc='prop radius, in', value=1, lower=0.01, upper=0.3, unit='in')
rpm_prop = syms.declare(name='rpm_prop', desc='prop rpm', value=0, lower=0, upper=100, unit='rad/s')
V_inf = syms.declare(name='V_inf', desc='Freestream velocity', value=1, lower=0, upper=10, unit='m/s') ## should value be changed?
rho = syms.declare(name='rho', desc='air density', value=1.225, lower=1, upper=2, unit='kg/m^3')
CT = syms.declare(name='CT', desc='Thrust coefficient', value=0, lower=0, upper=10, unit='')
CP = syms.declare(name='CP', desc='Power coefficient', value=0, lower=0, upper=0, unit='') ## lower = upper??


n = rpm_prop / 60  # angular velocity, rev/s
#omega_prop = n*2*ca.pi  # angular velocity of prop, rad/s

in_to_m = 0.0254
rpm_to_rps = 2*np.pi/60  # rev/min *(2*pi rad/rev)*(1min/60 sec) -> rad/sec

# propeller thrust and torque calculated using 'proper' model
#r_prop = r_prop_in*in_to_m  # radius of prop, m
#J = V_inf/(n*2*r_prop)
#V_prop = omega_prop*r_prop  # velocity of prop tip, m/s
#q_prop = rho*V_prop**2/2  # dynamics pressure of prop tip, N/m^2
#A_prop = ca.pi*r_prop**2  # area of prop disc, m^2
#T_prop = q_prop*A_prop*CT  # thrust of prop
#Q_prop = q_prop*A_prop*r_prop*CP  # torque of prop

# propeller thrust and torque using 'terrible' model
r_prop = r_prop_in *in_to_m  # radius of prop, m
J = V_inf/(n*2*r_prop)
T_prop = rho*(n**2)*((r_prop*2)**4)*CT  # thrust of prop
Q_prop = rho*(n**2)*((r_prop*2)**5)*(CP / (2*ca.pi))  # torque of prop
## CQ = CP / 2pi CITE!!!! https://m-selig.ae.illinois.edu/props/propDB.html

#%%
def prepare(prop_name, V_inf0, thrust_required, margin, max_volts):
    prop_data = prop_db[prop_name]
    key0 = list(prop_data['dynamic'].keys())[0]
    my_prop = prop_data['dynamic'][key0]['data']
    CT_lut = ca.interpolant('CT','bspline',[my_prop.index],my_prop.CT)
    CP_lut = ca.interpolant('CP','bspline',[my_prop.index],my_prop.CP)

    eta_prop_lut = ca.interpolant('eta','bspline',[my_prop.index],my_prop.eta)
    Q_prop_lut = ca.substitute(Q_prop, CP, CP_lut(J))
    T_prop_lut = ca.substitute(T_prop, CT, CT_lut(J))

    states = ca.vertcat(rpm_prop)
    params = ca.vertcat(rho, r_prop_in, V_inf)
    
    p0 = [1.225, prop_data['D']/2, V_inf0]

    f_Q_prop = ca.Function('Q_prop', [states, params], [Q_prop_lut])
    f_T_prop = ca.Function('T_prop', [states, params], [T_prop_lut])
    f_eta_prop = ca.Function('eta_prop', [states, params], [eta_prop_lut(J)])
    f_thrustC_prop = ca.Function('CT_prop', [states, params], [CT_lut(J)])
    f_powerC_prop = ca.Function('CP_prop', [states, params], [CP_lut(J)])

    rpm = np.array([np.linspace(1, 5000, 1000)])
    #ct_J = np.around([V_inf0/(r*rpm_to_rps*prop_data['D']) for r in ct_rpm[0]], decimals=4)

    fig = plt.figure(figsize=(8,8))
    ###ax1 = plt.subplot(511)
    ###plt.plot(rpm.T, f_Q_prop(rpm, p0).T, label='prop')
    #plt.plot(motor_omega, Q_m, '--', label='motor')
    ###plt.ylabel('torque, N-m')
    ###plt.legend()
    ###plt.grid(True)

    #ax2 = plt.subplot(512)
    #plt.plot(rpm.T, f_T_prop(rpm, p0).T, label='prop')
    #plt.ylabel('thrust, N')
    #plt.grid(True)

    #ax3 = plt.subplot(513)
    #plt.plot(rpm.T, f_eta_prop(rpm, p0).T, label='prop')
    #plt.xlabel('prop angular velocity, rpm')
    #plt.ylabel('efficiency')
    #plt.grid(True)

    ax4 = plt.subplot(514)
    plt.plot(rpm.T, f_thrustC_prop(rpm, p0).T, label='prop')
    plt.xlabel('Advance Ratio not')
    plt.ylabel('CT')
    plt.grid(True)

    ax5 = plt.subplot(515)
    plt.plot(rpm.T, f_powerC_prop(rpm, p0).T, label='prop')
    plt.xlabel('Advance Ratio not')
    plt.ylabel('CP')
    plt.grid(True)

    #ax1.set_ylim([0, 0.01])
    #ax1.set_xlim([1,3000])

    #ax2.set_ylim([0, 0.5])
    #ax2.set_xlim([1,3000])

    #ax3.set_ylim([0, 1])
    #ax3.set_xlim([1,3000])

    ax4.set_ylim([0, 0.2])
    ax4.set_xlim([3000,1])

    ax5.set_ylim([0, 0.2])
    ax5.set_xlim([3000,1])
    plt.show()

    ## scipy.sparse.csc_matrix is said to be better by casadi author
    f_eta_prop_array = np.array(f_eta_prop(rpm,p0))[0]
    max_eta_prop = np.nanmax(f_eta_prop_array)
    index = np.where(f_eta_prop_array == max_eta_prop)
    omega = rpm[0][index][0]
    T = np.array(f_T_prop(omega, p0))[0]
    Q = np.array(f_Q_prop(omega, p0))[0]

    ## values at required thrust
    T_array = np.array(f_T_prop(rpm,p0))[0]
    proximity = (T_array - thrust_required)**2
    index1 = np.where(proximity == np.nanmin(proximity))
    T_required = T_array[index1]

    omega_required = rpm[0][index1][0]
    eta_required = f_eta_prop_array[index1]
    Q_required = np.array(f_Q_prop(rpm, p0))[0][index1]

    rpm_proximity = (eta_required - max_eta_prop)**2


    if T_required > thrust_required - margin:
    
        motordat = md.prepare_m(Q_required, omega_required, max_volts)


        #motor0 = list(motordat.keys())[0]
        #motor_omega = motordat[motor0]['omega'] / rpm_to_rps
        ##plot Q_m vs Q_prop??


        data = {'prop_name': prop_name, 'max_eta_prop': max_eta_prop, 'omega_max': omega, 'T_max': T,\
                'Q_max': Q, 'diameter [in]': prop_data['D'], 'T_required': T_required, 'omega_required': omega_required,\
                    'eta_required': eta_required, 'Q_required': Q_required, 'rpm_proximity': rpm_proximity}
        plot(data, motordat)
        store(data)

#%%

def plot(data, motordata):

    plt.figure()
    plt.title(data['prop_name'] + 'D [in] = ' + str(data['diameter [in]']))
    plt.ylabel('required voltage')
    base = np.arange(len(motordata.keys()))

    v = np.array([motordata[name]['voltage_required'] for name in list(motordata.keys())])
    plt.bar(base, v)
    plt.xticks(base, tuple(motordata.keys()))

    plt.figure()
    plt.ylabel('efficiency')
    plt.xlabel('max rpm')
    
    rpm_maxs = np.array([motordata[name]['rpm_max_m'] for name in list(motordata.keys())])
    eta_maxs = np.array([motordata[name]['eta_max_m'] for name in list(motordata.keys())])
    plt.scatter(rpm_maxs, eta_maxs)
    plt.scatter(data['omega_max'], data['max_eta_prop'])
    plt.annotate(data['prop_name'], (data['omega_max'], data['max_eta_prop']))
    plt.axvline(x=data['omega_required'], color='red')
   
   
    for i, txt in enumerate(motordata.keys()):
        plt.annotate(txt, (rpm_maxs[i], eta_maxs[i]))


def out(V_inf0, thrust_required, margin, max_volts):
    sb.call(['rm','propstuff/efficiency.xlsx'])
    for prop_name in prop_db.keys():
        prepare(prop_name, V_inf0, thrust_required, margin, max_volts)
    #propdata = pd.read_excel('propstuff/efficiency.xlsx', sheet_name='sheet1')

    

def store(data):
    f = 'propstuff/efficiency.xlsx'
    if not path.exists(f):
        writer = pd.ExcelWriter(f, engine='xlsxwriter')
        df = pd.DataFrame(data, index=[0])
        df.to_excel(writer, sheet_name='sheet1', index=False)
        writer.save()
    else:
        df = pd.DataFrame(data, index=[0])
        writer = pd.ExcelWriter(f, engine='openpyxl')
        writer.book = load_workbook(f)
        writer.sheets = dict((ws.title, ws) for ws in writer.book.worksheets)
        reader = pd.read_excel(f)
        df.to_excel(writer, index=False, header=False, startrow=len(reader)+1, sheet_name='sheet1')
        writer.close()

