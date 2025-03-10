# %%
"""
Implementation of Coffee-Mug workshop
"""

import numpy as np
import control as ct
import matplotlib.pyplot as plt
from numpy import linalg as LA

__author__ = "Dirk M. Luchtenburg"
__version__ = "Feb. 2025"
__copyright__ = "Copyright 2025, Dirk M. Luchtenburg"

# Parameters
Ta = 20 # C
#  Coffee
rho = 1e3 # kg/m^3
v = 410e-6 # m^3
m = rho * v
cc = 4200 # J/kg/K
Cc = cc * m
print(f'Cc = {Cc:.2f} J/K')
Rc = 0.014 # K/W
#  Mug
m = 0.365 # kg
cm = 800 # J/kg/K
Cm = cm * m
print(f'Cm = {Cm:.2f} J/K')
Rm = 1.2 # K/W

# (b)
Tc0 = 90 # C
Tm0 = Ta
Te = (Cc*Tc0 + Cm*Tm0) / (Cc + Cm)
print(f'Te = {Te:.2f} C')

# (c, d)
A = np.array([
    [-1/(Rc*Cc),  1/(Rc*Cc)],
    [ 1/(Rc*Cm), -1/Cm*(1/Rc + 1/Rm)]
])
b = 1
B = np.array([
    [0],
    [b/Cm]
])
C = np.identity(2)
D = np.zeros((2,1))
sys = ct.ss(A, B, C, D)
#  Intial condition response
#  note: x = [Tc - Ta, Tm - Ta]
x0 = np.array([Tc0 - Ta, Tm0 - Ta])
t = np.linspace(0, 300, 200)
t, y = ct.initial_response(sys, t, x0)

print('ys =', y.shape)

Tc = y[0,:] + Ta
Tm = y[1,:] + Ta

plt.figure()
plt.plot(t, Tc, label=r'$T_c$')
plt.plot(t, Tm, label=r'$T_m$')
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Temperature [C]')
plt.grid()
plt.title('Initial condition response')
# %%

evals, evecs = np.linalg.eig(A)
for eval in evals: 
    print(f'eval = {eval: .6f}')
print('evec0= ', evecs[:,0])
print('evec0= ', evecs[:,1])
# %%
init_vals = np.linalg.solve(evecs,x0)
print(init_vals)
# %%
