#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 16:21:29 2021

@author: navid
"""


import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl

mpl.rcParams['figure.titlesize'] = 25 
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 22 
mpl.rcParams['xtick.labelsize'] = 22 
mpl.rcParams['ytick.labelsize'] = 22 
mpl.rcParams['legend.fontsize'] = 18






path = 'task_a'

d = np.loadtxt(f'{path}/game_rewards.txt')

plt.figure(figsize=(17,10))
plt.plot(d , label = 'Game reward' , alpha=1)
plt.legend()
plt.yticks([np.min(d) , -50 , 0 , np.mean(d) , np.max(d)])

plt.xlabel('Episodes')
plt.ylabel(r'$\sum_t r_t$')
plt.title(f'Task a', fontsize=22)
plt.savefig(f'{path}/rewards.png')

##############################################################################
##############################################################################
##############################################################################

path = 'task_b'

d = np.loadtxt(f'{path}/game_rewards.txt')

avg = []
maximum = []
minimum = []
for i in range(len(d)):
    if i < 50:
        avg.append(np.mean(d[0:100]))
        maximum.append(np.max(d[0:100]))
        minimum.append(np.min(d[0:100]))
    elif i > (len(d)-50):
        avg.append(np.mean(d[i-100:]))
        maximum.append(np.max(d[i-100:]))
        minimum.append(np.min(d[i-100:]))
    else:
        avg.append(np.mean(d[i-50:i+50]))
        maximum.append(np.max(d[i-50:i+50]))
        minimum.append(np.min(d[i-50:i+50]))

plt.figure(figsize=(17,10))
plt.plot(d , label = 'Game reward' , alpha=0.3)
plt.plot(avg , label = 'Moving average (window of 100 games)')
plt.plot(maximum , label = 'Max')
plt.plot(minimum , label = 'Min')
plt.legend()
plt.yticks([np.min(d) , -50 , 0 , np.mean(d[4000:]) , np.max(d)])
plt.plot(np.linspace(0,10000,500) , np.ones((500))*np.mean(d[4000:]) , 'r--')
plt.xlabel('Episodes')
plt.ylabel(r'$\sum_t r_t$')
plt.title(f'Task a', fontsize=22)
plt.xlim(0,10000)
plt.savefig(f'{path}/rewards.png')

##############################################################################
##############################################################################
##############################################################################

path = 'task_c'

d = np.loadtxt(f'{path}/gamma_07/game_rewards.txt')

avg = []
maximum = []
minimum = []
for i in range(len(d)):
    if i < 50:
        avg.append(np.mean(d[0:100]))
        maximum.append(np.max(d[0:100]))
        minimum.append(np.min(d[0:100]))
    elif i > (len(d)-50):
        avg.append(np.mean(d[i-100:]))
        maximum.append(np.max(d[i-100:]))
        minimum.append(np.min(d[i-100:]))
    else:
        avg.append(np.mean(d[i-50:i+50]))
        maximum.append(np.max(d[i-50:i+50]))
        minimum.append(np.min(d[i-50:i+50]))

plt.figure(figsize=(17,10))
plt.plot(d , label = 'Game reward' , alpha=0.3 , linewidth=0.03)
plt.plot(avg , label = 'Moving average (window of 100 games)')
plt.plot(maximum , label = 'Max')
plt.plot(minimum , label = 'Min')
plt.legend(loc=4)
plt.yticks([np.min(d) , -50  , np.mean(d[75000:]) , np.mean(maximum[75000:]) , np.max(d)])

plt.xlabel('Episodes')
plt.ylabel(r'$\sum_t r_t$')
plt.title(f'Task c', fontsize=22)
plt.savefig(f'{path}/gamma_07/rewards.png')




##############################################################################
##############################################################################
##############################################################################

path = 'task_2'

d = np.loadtxt(f'{path}/gamma_07/game_rewards.txt')

avg = []
maximum = []
minimum = []
for i in range(len(d)):
    if i < 50:
        avg.append(np.mean(d[0:100]))
        maximum.append(np.max(d[0:100]))
        minimum.append(np.min(d[0:100]))
    elif i > (len(d)-50):
        avg.append(np.mean(d[i-100:]))
        maximum.append(np.max(d[i-100:]))
        minimum.append(np.min(d[i-100:]))
    else:
        avg.append(np.mean(d[i-50:i+50]))
        maximum.append(np.max(d[i-50:i+50]))
        minimum.append(np.min(d[i-50:i+50]))

plt.figure(figsize=(17,10))
plt.plot(d , label = 'Game reward' , alpha=0.3 , linewidth=0.03)
plt.plot(avg , label = 'Moving average (window of 100 games)')
plt.plot(maximum , label = 'Max')
plt.plot(minimum , label = 'Min')
plt.legend(loc=2)
plt.yticks([np.min(d) , -50  , np.mean(d[6000:]) , np.mean(maximum[6000:]) , np.max(d)])
plt.plot(np.linspace(0,10000,500) , np.ones((500))*np.mean(d[6000:]) , 'r--')
plt.xlim(0,10000)
plt.xlabel('Episodes')
plt.ylabel(r'$\sum_t r_t$')
plt.title(r'Task 2 - $\gamma = 0.7$', fontsize=22)
plt.savefig(f'{path}/gamma_07/rewards.png')




path = 'task_2'

d = np.loadtxt(f'{path}/gamma_1/game_rewards.txt')

avg = []
maximum = []
minimum = []
for i in range(len(d)):
    if i < 50:
        avg.append(np.mean(d[0:100]))
        maximum.append(np.max(d[0:100]))
        minimum.append(np.min(d[0:100]))
    elif i > (len(d)-50):
        avg.append(np.mean(d[i-100:]))
        maximum.append(np.max(d[i-100:]))
        minimum.append(np.min(d[i-100:]))
    else:
        avg.append(np.mean(d[i-50:i+50]))
        maximum.append(np.max(d[i-50:i+50]))
        minimum.append(np.min(d[i-50:i+50]))

plt.figure(figsize=(17,10))
plt.plot(d , label = 'Game reward' , alpha=0.3 , linewidth=0.03)
plt.plot(avg , label = 'Moving average (window of 100 games)')
plt.plot(maximum , label = 'Max')
plt.plot(minimum , label = 'Min')
plt.legend(loc=2)
plt.yticks([np.min(d) , -50  , np.mean(d[6000:]) , np.mean(maximum[6000:]) , np.max(d)])
plt.plot(np.linspace(0,10000,500) , np.ones((500))*np.mean(d[6000:]) , 'r--')
plt.xlim(0,10000)
plt.xlabel('Episodes')
plt.ylabel(r'$\sum_t r_t$')
plt.title(r'Task 2 - $\gamma = 1$', fontsize=22)
plt.savefig(f'{path}/gamma_1/rewards.png')





























