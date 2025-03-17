# %%
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *

#Consts

R = 8.4 #ohms
Kte = 0.04 #Nm/A

J = 2.1 * 10**-5 #kgm^2
L = 1.16 * 10**-3 #H
b = 0.01 #Nm / (rad/s)
va = 12 #v
# %%

# --- State Space Model -----
A = np.array([[ (- R/L), 1 , (-Kte/ L)],
             [0 ,0, 1],
             [Kte/J , 0 , -b/J]])
B = np.array([[va/L], [0], [0]])
C = np.eye(3)
D = np.zeros((3,1))

sys = ss(A, B, C, D)

# Time vector
t = np.linspace(0, 10e-4, 1000)

# Step response
p = poles(sys)

plt.plot(np.real(p)[0], np.imag(p)[0], 'X')
plt.plot(np.real(p)[1], np.imag(p)[1], 'X')
plt.plot(np.real(p)[2], np.imag(p)[2], 'X')
plt.grid(True)
plt.axhline(0,color = 'black')
plt.axvline(0,color = 'black')
plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('pole plot')
plt.show()
# %%

y,t = step(sys, t)
y = np.squeeze(y)
fig = plt.figure()
ax = fig.add_subplot(2, 1, 2)
#ax.set_xscale('log')
#ax.set_yscale('log')
plt.plot(t, y)
plt.grid(True)
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('Step response')
plt.show()
# %%
#BODE plot time
new_A = np.array([[ (- R/L), 1 , (-Kte/ L)],
             [0 ,0, 1],
             [Kte/J , 0 , -b/J]])
new_B = np.array([[1/L], [0], [0]])
new_C = np.array([[0],[0],[1]])
new_D = np.array([[0],[0],[0]])
new_sys = ss(A, B, C, D)
mag, phase, omega = bode(new_sys)

# %%
