# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control.matlab import *
import control as ctrl
import matplotlib.pyplot as plt
plt.close('all')
# Parameters
R = 100
C1 = 2
C2 = 1
Q_in = 1
# Define the ODE system
def vessel_system(t, h):
    h1, h2 = h
    dh1_dt = -(h1 - h2) / (R * C1) + Q_in/C1
    dh2_dt = (h1 - h2) / (R * C2)
    return [dh1_dt, dh2_dt]

# Initial conditions
h1_0 = 125
h2_0 = 50
x0 = [h1_0, h2_0]

# Time span for the simulation
t_span = (0, 600)
t_eval = np.linspace(0, 600, 1000)

# Solve the ODE
sol = solve_ivp(vessel_system, t_span, x0)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sol.t, sol.y[0], label='h1(t)')
plt.plot(sol.t, sol.y[1], label='h2(t)')
plt.xlabel('Time')
plt.ylabel('Fluid Level')
plt.title('Fluid Levels in Communicating Vessels')
plt.legend()
plt.grid(True)
plt.show()

# Check if total fluid is conserved
initial_total = C1 * h1_0 + C2 * h2_0
final_total = C1 * sol.y[0][-1] + C2 * sol.y[1][-1]
print(f"Initial total fluid: {initial_total}")
print(f"Final total fluid: {final_total}")
# %%

# Define the state space matrices
A = np.array([[-1/(R*C1), 1/(R*C1)], 
              [1/(R*C2), -1/(R*C2)]])
B = np.array([[1/C1], [0]])
C = np.eye(2)
D = np.zeros((2, 1))

# Create the state space system
sys = ss(A, B, C, D)

# Compute the step response
t = np.linspace(0, 1000, 1000)
y, t = initial(sys, t, x0)

# Plot the response
plt.figure(figsize=(10, 6))
plt.plot(t, y[:,0], label='h1(t)')
plt.plot(t, y[:,1], label='h2(t)')
plt.xlabel('Time')
plt.ylabel('Fluid Level')
plt.title('Step Response - Fluid Levels with Constant Inflow')
plt.legend()
plt.grid(True)
plt.show()

# Step response
t = np.linspace(0, 400, 1000)
y, t = step(sys, t)
y = np.squeeze(y)

plt.figure()
plt.plot(t, y)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$h_1$, $h_2$ [m]')
plt.title('Step response')
plt.legend((r'$h_1$',r'$h_2$'))


# Step response difference
#  y = h1 - h2
C_diff = np.array([1, -1])
sys_diff = ss(A,B,C_diff,0)
# Step response
y, t = step(sys_diff, t)

plt.figure()
plt.plot(t, y)
plt.grid()
plt.xlabel('Time [s]')
plt.ylabel(r'$h_1 - h_2$ [m]')
plt.title('Step response difference')



plt.show()
# %%

# Define the time range
t = np.linspace(0, 20, 400)

# Define the analytical solution
y = 4 * np.exp(-0.5 * t) * np.sin(0.5 * t)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t, y, label='y(t) = 4 * exp(-0.5t) * sin(0.5t)')
plt.title('Solution to the Initial Value Problem')
plt.xlabel('Time (t)')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()


# %%
