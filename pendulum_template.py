"""
Script Name: pendulum_template.py
Description: Simulation pendulum equations (linearized, nonlinear)
Author: Dirk M. Luchtenburg
Date Created: Feb. 2025
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *  # MATLAB-like control toolbox functionality
from scipy.integrate import solve_ivp
# %%
# Close all figures
plt.close('all')

# Parameters
m = 1
g = 10
l = 1

# Derived quantities
omega_n = np.sqrt(g/l)
T = 2*np.pi / omega_n

# ICs
theta0 = np.pi/8
omega0 = 0

# --- Linear model -----

# ...FILL IN THE BLANKS....

A = np.array([[0,1],
             [- omega_n, 0]])
B = np.array([[0], [1/(m * l ** 2)]])
C = np.eye(2)
D = np.array([[0],[0]])
sys = ss(A, B, C, D)
# --- Nonlinear model -----

# ...FILL IN THE BLANKS....

def pend(t, x):
    theta, omega = x   
    f = np.array([[omega],
                  [-omega_n * np.sin(theta) + t / (m * l ** 2)]]
    )
    return [omega,-omega_n * np.sin(theta) + 0 / (m * l ** 2)]


# IC responses
x0 = np.array([theta0, omega0])
t = np.linspace(0, 10*T, 1000)
# --- Linear model -----
y, t = initial(sys, t, x0)
theta_lin = y[:,0]
omega_lin = y[:,1]
# --- Nonlinear model -----
tspan = [t[0], t[-1]]
sol = solve_ivp(pend, tspan, x0, t_eval=t, rtol=1e-6, atol=1e-9)
theta = sol.y[0,:]
omega = sol.y[1,:]

plt.figure()
plt.plot(t, theta, label=r'$\theta$')
plt.plot(t, omega, label=r'$\omega$')
plt.xlabel('$t [s]$')
plt.ylabel(r'$ \theta~ [rad], \omega~ [rad/s]$')
plt.legend()
plt.title('IC response')


plt.figure()
plt.plot(t, theta, label=r'$\theta$')
plt.plot(t, theta_lin, label=r'$\theta$ (linear model)')
plt.xlabel('$t [s]$')
plt.ylabel(r'$ \theta~ [rad], \omega~ [rad/s]$')
plt.legend()
plt.title('IC response')
plt.show()
# %%