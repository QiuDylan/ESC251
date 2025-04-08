# %%
from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np

# Set up ss for quarter car
# Input: road profile, Output: displacement main body
# Approximate bus values:
m1 = 2500 # kg, main body
m2 = 320 # kg, suspension
kw = 500e3 # N/m, wheel, tire
ks = 80e3 # N/m, suspension
bs = 350e1 # Ns/mm damping suspension
bw = 15e3 # Ns/mm damping wheel, tire

A = np.array([[0,0,1,0],
             [0,0,0,1],
             [-ks/m1, ks/m1, -bs/m1, bs/m1],
             [ks/m2, (-ks-kw)/m2, bs/m2, (-bs-bw)/m2]])
B = np.array([0, 0, 0, kw/m2])
C =  np.array([1,0,0,0])
D = np.zeros((1))
sys = ss(A, B, C, D)

# Step response
t = np.linspace(0,50,1000)
y, t = step(sys,t)
#y = .01 * y 
plt.figure()
plt.plot(t, y)
plt.xlabel('Time [s]'); plt.ylabel('Response x_2 [m]'),

# Frequency response...
# use forced response, see:
# https://python-control.readthedocs.io/en/0.8.3/generated/control.forced_response.html#control.forced_response 
# %%
#Polo plot
p = poles(sys)

plt.plot(np.real(p)[0], np.imag(p)[0], 'x')
plt.plot(np.real(p)[1], np.imag(p)[1], 'x')
plt.plot(np.real(p)[2], np.imag(p)[2], 'x')
plt.plot(np.real(p)[3], np.imag(p)[3], 'x')
plt.grid(True)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('pole plot')
# %%
#BODE plot time
bode(sys)   
# %%
# Linear simulation

T = np.linspace(0,50,1000)
u = .2*np.sin(5.1*T) # road profile
yout, t, x = lsim(sys, U = u ,T = T)

plt.plot(T, yout, label = 'yout',color='r')
plt.plot(T, u, label = 'input freq', color = 'b')
plt.xlabel('Time')
plt.ylabel('Output')
plt.title('Simulation')
plt.show()
# %%
