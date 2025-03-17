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
# %%

# --- State Space Model -----
A = np.array([[ (- R/L), 1 , (-Kte/ L)],
             [0 ,0, 1],
             [Kte/J , 0 , -b/J]])
B = np.array([[1/L], [0], [0]])
C = np.eye(3)
D = np.zeros((3,1))

sys = ss(A, B, C, D)

# Time vector
t = np.linspace(0, 400, 1000)

# Step response
p = poles(sys)

plt.plot(np.real(p)[0], np.imag(p)[0], 'x')
plt.plot(np.real(p)[1], np.imag(p)[1], 'x')
plt.plot(np.real(p)[2], np.imag(p)[2], 'x')
plt.grid(True)
plt.xlabel('real')
plt.ylabel('imaginary')
plt.title('pole plot')
plt.show()