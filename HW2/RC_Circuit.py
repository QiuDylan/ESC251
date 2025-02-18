# %%
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import * # MATLAB-like control toolbox functionality

# %%
# Close all figures
plt.close('all')

# Parameters
R = 100e3  # 100 kΩ
C = 1e-6  # 0.5 µF
Vin = 1.0   # Input voltage (step)

# Derived quantities
tau = R * C
K = 0.5 * Vin  # Final value

# ICs
Vout0 = 0      # Initial output voltage
dVout0 = 0     # Initial rate of change

# --- State Space Model -----
A = np.array([[-2/(R*C)]])
B = np.array([[1/(R*C)]])
C = np.array([[1]])
D = np.array([[0]])

sys = ss(A, B, C, D)

# Time vector
t = np.linspace(0, 0.35, 1000)

# Step response
[y, t] = step(sys, t)

# Analytical solution
Vout_analytical = K * (1 - np.exp(-t/tau))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, y, 'b-', label='State Space Model')
#plt.plot(t, Vout_analytical, 'r--', label='Analytical Solution')
plt.grid(True)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [V]')
plt.title('Step Response of RC Filter')
plt.legend()
plt.ylim(0, 0.6)

# Format plot to match homework figure
plt.gca().set_xticks(np.arange(0, 0.4, 0.05))
plt.gca().set_yticks(np.arange(0, 0.6, 0.1))

plt.show()
# %%