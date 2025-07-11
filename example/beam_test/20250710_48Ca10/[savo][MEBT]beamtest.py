#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ! pip install -e ../../savo


# In[ ]:


import time
import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../')  #for savo import
sys.path.append('../../machineIO/')  #for machineIO import
from savo import savo
from savo.optim import adam
from machineIO.util import plot_2D_projection, dictClass
from machineIO.objFunc import SingleTaskObjectiveFunction
from machineIO.construct_machineIO import construct_machineIO
from machineIO import Evaluator
from epics import caget
import pickle


# In[ ]:


# PVs = np.loadtxt('LEBT_phys_pvlist.txt',dtype=str)
# [pv.replace(':DCH_',':PSC2_').replace(':DCV_',':PSC1_').replace(':SOLR_',':PSOL_') for pv in PVs if ':DC' in pv or ':SOL' in pv]


# In[ ]:


FC814 = 8
budget = 100
ninit = 30
model_train_budget = 200
lr = 1
set_manually = False


SCS = caget("ACS_DIAG:DEST:ACTIVE_ION_SOURCE")
ion = caget("FE_ISRC"+str(SCS)+":BEAM:ELMT_BOOK")
Q = caget("FE_ISRC"+str(SCS)+":BEAM:Q_BOOK")
A = caget("FE_ISRC"+str(SCS)+":BEAM:A_BOOK")
AQ = A/Q
ion = str(A)+ion+str(Q)

now0 = datetime.datetime.now()
fname = now0.strftime('%Y%m%d_%H%M')+'['+ion+'][savo][MEBT]beamtest'


machineIO = construct_machineIO(isOK_PVs = [],isOK_vals=[])
machineIO._ensure_set_timeout = 30
machineIO._fetch_data_time_span = 2
machineIO._ensure_set_timewait_after_ramp = 0.2



control_CSETs = [
'FE_LEBT:PSC2_D0773',
'FE_LEBT:PSC1_D0773',
#'FE_LEBT:PSOL_D0787',
'FE_LEBT:PSC2_D0790',
'FE_LEBT:PSC1_D0790',
#'FE_LEBT:PSOL_D0802',
'FE_LEBT:PSC2_D0805',
'FE_LEBT:PSC1_D0805',
#'FE_LEBT:PSOL_D0818',
'FE_LEBT:PSC2_D0821',
'FE_LEBT:PSC1_D0821',
'FE_LEBT:PSC2_D0840',
'FE_LEBT:PSC1_D0840',
'FE_LEBT:PSC2_D0868',
'FE_LEBT:PSC1_D0868',
'FE_LEBT:PSC2_D0880',
'FE_LEBT:PSC1_D0880',
'FE_LEBT:PSC2_D0901',
'FE_LEBT:PSC1_D0901',
'FE_LEBT:PSC2_D0929',
'FE_LEBT:PSC1_D0929',
'FE_LEBT:PSC2_D0948',
'FE_LEBT:PSC1_D0948',
'FE_LEBT:PSOL_D0951',
'FE_LEBT:PSC2_D0964',
'FE_LEBT:PSC1_D0964',
'FE_LEBT:PSOL_D0967',
'FE_LEBT:PSC2_D0979',
'FE_LEBT:PSC1_D0979',
'FE_LEBT:PSOL_D0982',
'FE_LEBT:PSC2_D0992',
'FE_LEBT:PSC1_D0992',
'FE_LEBT:PSOL_D0995',
# 'FE_MEBT:PSC2_D1062',
# 'FE_MEBT:PSC1_D1062',
]
control_CSETs = [pv+':I_CSET' for pv in control_CSETs]
control_RDs   = [pv.replace('_CSET','_RD') for pv in control_CSETs]
ndim = len(control_CSETs)

control_min = []
control_max = []
control_tols = []
for PV in control_CSETs:
    if 'PSC' in PV:
        control_min.append( -4.0)
        control_max.append( +4.0)
        control_tols.append(0.2)
    elif 'PSOL' in PV:
        v = caget(PV)
        control_min.append(0.8*v)
        control_max.append(1.2*v)
        control_tols.append(0.7)
    else:
        raise ValueError(f'control bounds for {PV} cannot be determined')

        
control_min = np.array(control_min)
control_max = np.array(control_max)
control_tols = np.array(control_tols)
control_maxstep = 2e-3*(control_max-control_min)





objective_goal= {
    'FE_MEBT:BPM_D1056:XPOS_RD': -0.3960103833349608,
    'FE_MEBT:BPM_D1056:YPOS_RD': 0.6678681961462211,
    'FE_MEBT:BPM_D1056:PHASE_RD': 78.75262742941518,
    'FE_MEBT:BPM_D1072:XPOS_RD': 0.18746895673132966,
    'FE_MEBT:BPM_D1072:YPOS_RD': 0.19623165740431964,
    'FE_MEBT:BPM_D1072:PHASE_RD': -25.102889168544564,
    'FE_MEBT:BPM_D1094:XPOS_RD': -0.18243672772460348,
    'FE_MEBT:BPM_D1094:YPOS_RD': -1.0409382935331426,
    'FE_MEBT:BPM_D1094:PHASE_RD': -17.109880065240283,
    'FE_MEBT:BCM_D1055:AVGPK_RD': {'more than': 0.99*FC814},
    'FE_MEBT:FC_D1102:PKAVG_RD' : {'more than':  0.8*FC814},
}


# In[ ]:


objective_tolerance = {
    'FE_MEBT:BPM_D1056:XPOS_RD' : 1,
    'FE_MEBT:BPM_D1056:YPOS_RD' : 1,
    'FE_MEBT:BPM_D1056:PHASE_RD': 1,
    'FE_MEBT:BPM_D1072:XPOS_RD' : 1,
    'FE_MEBT:BPM_D1072:YPOS_RD' : 1,
    'FE_MEBT:BPM_D1072:PHASE_RD': 1,
    'FE_MEBT:BPM_D1094:XPOS_RD' : 1,
    'FE_MEBT:BPM_D1094:YPOS_RD' : 1,
    'FE_MEBT:BPM_D1094:PHASE_RD': 1,
    'FE_MEBT:BCM_D1055:AVGPK_RD': 0.05*FC814,
    'FE_MEBT:FC_D1102:PKAVG_RD' : 0.05*0.8*FC814,
}


# In[ ]:


objective_weight= {
    'FE_MEBT:BPM_D1056:XPOS_RD' : 1.0,
    'FE_MEBT:BPM_D1056:YPOS_RD' : 1.0,
    'FE_MEBT:BPM_D1056:PHASE_RD': 1.0,
    'FE_MEBT:BPM_D1072:XPOS_RD' : 0.8,
    'FE_MEBT:BPM_D1072:YPOS_RD' : 0.8,
    'FE_MEBT:BPM_D1072:PHASE_RD': 0.8,
    'FE_MEBT:BPM_D1094:XPOS_RD' : 0.5,
    'FE_MEBT:BPM_D1094:YPOS_RD' : 0.5,
    'FE_MEBT:BPM_D1094:PHASE_RD': 0.5,
    'FE_MEBT:BCM_D1055:AVGPK_RD': 2,
    'FE_MEBT:FC_D1102:PKAVG_RD' : 2,
}


# In[ ]:


monitor_RDs   = list(objective_goal.keys())
objective_PVs = monitor_RDs
composite_objective_name = 'composite_obj'


# # obj_func

# In[ ]:


obj_func = SingleTaskObjectiveFunction(
    objective_PVs = monitor_RDs,
    composite_objective_name = composite_objective_name,
    custom_function = None,
    objective_goal = objective_goal, 
    objective_weight = objective_weight,
    objective_tolerance = objective_tolerance,
)


# In[ ]:


evaluator = Evaluator(
    machineIO = machineIO,
    control_CSETs = control_CSETs,
    control_RDs = control_RDs,
    control_tols = control_tols,
    monitor_RDs = monitor_RDs,
    df_manipulators = [obj_func.calculate_objectives_from_df],
    set_manually = set_manually,    
)
control_init = evaluator.read()[control_CSETs].mean().values


# In[ ]:


def plot_hist(history):
    fig, ax = plt.subplots(1,2,figsize=(8,3),dpi=96)
    xaxis = np.arange(len(history['y']))
    ax[0].plot(xaxis, history['y'])
    ax[0].set_xlabel('epoch');
    ax[0].set_ylabel('objective');
    ax[1].plot(xaxis, history['cpu_time'])
    ax[1].set_xlabel('epoch');
    ax[1].set_ylabel('cpu_time');
    fig.tight_layout()


# # ES

# In[ ]:


df,ramp = evaluator._set_and_read(control_init)
evaluator.clear_history()
sv = savo(control_CSETs, control_RDs, control_min, control_max, control_maxstep, objective_PVs, composite_objective_name, evaluator)
for i in range(budget):
    print(i)
    sv.step(lr=0,lrES=1)




plot_hist(sv.history)


# In[ ]:


pickle.dump(sv.history,open(fname+'_ES.pkl','wb'))
pickle.dump(evaluator.get_history()['mean'],open(fname+'_ES_eval.pkl','wb'))


# # ES + adamSG

# In[ ]:


df,ramp = evaluator._set_and_read(control_init)
evaluator.clear_history()
sv = savo(control_CSETs, control_RDs, control_min, control_max, control_maxstep, objective_PVs, composite_objective_name, evaluator, 
          model_train_budget = model_train_budget, optimizer=adam())
for i in range(ninit):
    print(i)
    sv.step(lr=0,lrES=1)
for i in range(budget-ninit):
    print(i)
    sv.step(lr=lr,lrES=1)


# In[ ]:


plot_hist(sv.history)


# In[ ]:


pickle.dump(sv.history,open(fname+'_ES_adamSG.pkl','wb'))
pickle.dump(evaluator.get_history()['mean'],open(fname+'_ES_adamSG_eval.pkl','wb'))


# # ES + SG

# In[ ]:


df,ramp = evaluator._set_and_read(control_init)
evaluator.clear_history()
sv = savo(control_CSETs, control_RDs, control_min, control_max, control_maxstep, objective_PVs, composite_objective_name, evaluator, 
          model_train_budget = model_train_budget)
for i in range(ninit):
    print(i)
    sv.step(lr=0,lrES=1)
for i in range(budget-ninit):
    print(i)
    sv.step(lr=lr,lrES=1)


# In[ ]:


plot_hist(sv.history)


# In[ ]:


pickle.dump(sv.history,open(fname+'_ES_SG.pkl','wb'))
pickle.dump(evaluator.get_history()['mean'],open(fname+'_ES_SG_eval.pkl','wb'))


# # ES + SG penalize_uncertain_gradient

# In[ ]:


# # ES + SG penalize_uncertain_gradient
df,ramp = evaluator._set_and_read(control_init)
evaluator.clear_history()
sv = savo(control_CSETs, control_RDs, control_min, control_max, control_maxstep, objective_PVs, composite_objective_name, evaluator, 
          model_train_budget = model_train_budget)
for i in range(ninit):
    print(i)
    sv.step(lr=0,lrES=1)
for i in range(budget-ninit):
    print(i)
    sv.step(lr=lr,lrES=1,penalize_uncertain_gradient=True)


# In[ ]:


plot_hist(sv.history)


# In[ ]:


pickle.dump(sv.history,open(fname+'_ES_UncertainSG.pkl','wb'))
pickle.dump(evaluator.get_history()['mean'],open(fname+'_ES_UncertainSG_eval.pkl','wb'))


# # SG wo ES

# In[ ]:


df,ramp = evaluator._set_and_read(control_init)
evaluator.clear_history()
sv = savo(control_CSETs, control_RDs, control_min, control_max, control_maxstep, objective_PVs, composite_objective_name, evaluator, 
          model_train_budget = model_train_budget)
for i in range(ninit):
    print(i)
    sv.step(lr=0,lrES=1)
for i in range(budget-ninit):
    print(i)
    sv.step(lr=lr,lrES=0)


# In[ ]:


plot_hist(sv.history)


pickle.dump(sv.history,open(fname+'_SG.pkl','wb'))
pickle.dump(evaluator.get_history()['mean'],open(fname+'_SG_eval.pkl','wb'))


df,ramp = evaluator._set_and_read(control_init)
evaluator.clear_history()

