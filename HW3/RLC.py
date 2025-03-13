# %%
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import * # MATLAB-like control toolbox functionality

# %%
# Close all figures
plt.close('all')

# Parameters
R = 100  # 100 Ω
C = 10e-9  # 10nF
L = 0.01 # H
#Vin = 1.0   # Input voltage (step)
# --- State Space Model -----
A = np.array([[0,1],
             [-1/ (L * C) , - R / L]])
B = np.array([[0], [1/L]])
C = np.eye(2)
D = np.zeros((2,1))

sys = ss(A, B, C, D)

# Time vector
t = np.linspace(0, 10e-4, 1000)

# Step response
y, t = step(sys, t)
y = np.squeeze(y)

# Plotting
#plt.figure(figsize=(10, 6))
plt.plot(t, y)
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Step Response of RLC Filter')
#plt.legend()
#plt.ylim(0, 0.6)


plt.show()


# %%
