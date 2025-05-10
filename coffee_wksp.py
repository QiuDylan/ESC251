# %%
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import *

#constants
#coffee
c = 4200 # J/kg/K
V_c = 410 # cm^3
V_c = V_c * 10 ** -6 #conv to m^3
rho = 1000 # kg/m^3
T_c0 = 90 # deg C
T_a = 20 # deg C
R_c = 0.014 #K/W
R_m = 1.20 #K/W

#mug
c_m = 800 # J/kg/K
M_m = 0.365 # kg

#derived constants

# %%
#q1
C_c = V_c * rho * c # J/k
C_m = c_m * M_m # J/k
print(f"Capacitance of coffee is: {C_c} J/K")
print(f"Capacitance of mug is: {C_m} J/k")
# %%
#q2 
E_0 = T_c0 * C_c + C_m * T_a
T_e = E_0 / (C_c + C_m) #Energy 
print("The equilibrium temp %.2f deg C" % T_e)
# %%
#q3

x0 = [T_c0, T_a] # Initial conditions

# Define the state space matrices
A = np.array([[-1/(R_c * C_c), 1/ (R_c * C_c)], 
              [1/(R_c* C_m), (-1 / (C_m)) * (1/ R_c + 1 /R_m)]])
B = np.array([[0], [1/C_m]])
C = np.eye(2)
D = np.zeros((2, 1))

sys = ss(A, B, C, D)

# Compute the step response
t = np.linspace(0, 4000, 10000)
y, t = initial(sys, t, x0)

# Plot the response
plt.figure()
plt.plot(t, y[:,0], label='T_c')
plt.plot(t, y[:,1], label='T_m')
plt.xlabel('Time [s]')
plt.ylabel('Temperature [c]')
plt.title('Temperature evolution of the coffee and mug')
plt.legend()
plt.grid(True)
plt.show()
# %%
