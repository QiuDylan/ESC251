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
Vin = 1.0   # Input voltage (step)

# Derived quantities
tau = R * C
#K = 0.5 * Vin  # Final value

# ICs
Vout0 = 0      # Initial output voltage
dVout0 = 0     # Initial rate of change
x0 = [Vout0, dVout0]
# --- State Space Model -----
A = np.array([[0,1],
             [-1 / (L * C), - R / L]])
B = np.array([[0], [1/(L*C)]])
C = np.eye(2)
D = np.array([[0],[0]])

sys = ss(A, B, C, D)

# Time vector
t = np.linspace(0, 400, 1000)

# Step response
y, t = initial(sys, t)
y = np.squeeze(y)
# Analytical solution
#Vout_analytical = K * (1 - np.exp(-t/tau))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, y)
#plt.plot(t, Vout_analytical, 'r--', label='Analytical Solution')
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Step Response of R:C Filter')
#plt.legend()
#plt.ylim(0, 0.6)

# Format plot to match homework figure

plt.show()


# %%
