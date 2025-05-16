# %% 

# Dylan Qiu, ME' 27 For Take Home Final ESC251 
# Completed 5-13-2025 For Prof. Luchtenburg's Class
import numpy as np
import matplotlib.pyplot as plt
import control
from control.matlab import * 

# %%
# Givens
m1 = 20  # kg
m2 = 50  # kg
k1 = 7410 #- 6000 # N/m
k2 = 8230 #* 2000 # N/m
b1 = 1430 #+ 1430  # Ns/m
b2 = 153  # Ns/m

# State-space 
A = np.array([
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-(k1 + k2)/m1, k2/m1, -(b1 + b2)/m1, b2/m1],
    [k2/m2, -k2/m2, b2/m2, -b2/m2]
])

B = np.array([
    [0, 0],
    [0, 0],
    [k1/m1, b1/m1],
    [0, 0]
])

C = np.array([
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

D = np.zeros((2, 2))

sys = ss(A, B, C, D)

# Calculate eigenvalues (poles)
poles = np.linalg.eigvals(A)
print("System poles:")
for i, pole in enumerate(poles):
    print(f"Pole {i+1}: {pole:.4f}")
# %%
# Sort and Plot poles
sorted_poles = sorted(poles, key=lambda x: abs(x.real))
slow_poles = sorted_poles[:2]
fast_poles = sorted_poles[2:]

print("\nSlow poles:")
for pole in slow_poles:
    print(f"{pole:.4f}")

print("\nFast poles:")
for pole in fast_poles:
    print(f"{pole:.4f}")

# Calculate natural frequency and time constant for slow poles
if slow_poles[0].imag != 0:  
    natural_freq = abs(slow_poles[0])   
    damping_ratio = -slow_poles[0].real / natural_freq
    time_constant = 1 / (damping_ratio * natural_freq)
    print(f"\nSlow poles natural frequency: {natural_freq:.4f} rad/s")
    print(f"Damping ratio: {damping_ratio:.4f}")
    print(f"Time constant: {time_constant:.4f} s")
for pole in poles:
    plt.plot(pole.real, pole.imag, 'rx', markersize=10)

# Mark slow and fast poles
for pole in slow_poles:
    plt.plot(pole.real, pole.imag,'bo', markersize=12, fillstyle='full')
for pole in fast_poles:
    plt.plot(pole.real, pole.imag,'go', markersize=12, fillstyle='full')

plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Pole Plot in Complex Plane')
plt.grid()
plt.legend(['Poles', 'Slow Poles = blue', 'Fast Poles = green'])
# %%

# Frequencies Response
frequencies = [1.57, 6.28, 25.13]  # rad/s

t = np.linspace(0, 10, 10000)
plt.figure(figsize=(15, 12))
gains = []

for i, omega in enumerate(frequencies):
    # Input signals
    u1 = 0.02 * np.sin(omega * t)  # z0(t) = 0.02*sin(omega*t)
    u2 = 0.02 * omega * np.cos(omega * t)  # z0_dot(t) 
    u = np.column_stack((u1, u2))
    y, t_out, x = lsim(sys, u, t)
    z2 = y[:,0]
    start_idx = np.where(t >= 1)[0][0]
    max_input = np.max(np.abs(u1[start_idx:]))
    max_output = np.max(np.abs(z2[start_idx:]))
    gain = max_output / max_input
    gains.append(gain)
    
    # Plot results
    plt.subplot(3, 1, i+1)
    plt.plot(t_out, u1, 'b-', label='Input z0(t)')
    plt.plot(t_out, z2, 'r-', label='Driver displacement z2(t)')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title(f'Response to input frequency ω = {frequencies[i]} rad/s, Gain = {gain:.4f}')
    
    if gain > 1:
        behavior = "amplification"
    else:
        behavior = "attenuation"
    
    plt.annotate(f'Gain = {gain:.4f} ({behavior})', 
                 xy=(0.1, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    
    plt.legend()
# %%
# Response for Frequency[1] 
gains = []
new_c = np.array([
    [0, 1, 0, 0],
    #[0, 0, 0, 0]
])
new_d = np.zeros((1,2))
t_new = np.linspace(0,10,1000)
new_sys = ss(A,B, new_c, new_d)

u1_new = 0.02 * np.sin(frequencies[1] * t_new)
u2_new = 0.02 * frequencies[1] * np.cos(frequencies[1] * t_new)

u_new = np.column_stack((u1_new, u2_new))
y_out, t_newout, x = lsim(sys, u_new, t_new)
z2_new = y_out[:,0]

start_idx = np.where(t_new >= 2)[0][0]
max_input = np.max(np.abs(u1_new[start_idx:]))
max_output = np.max(np.abs(z2_new[start_idx:]))
gain = max_output / max_input
gains.append(gain)

plt.plot(t_newout, u1_new, 'b-', label='Input z0(t)')
plt.plot(t_newout, z2_new, 'r-', label='Driver displacement z2(t)')
#plt.plot(t_newout, z2_new * m2, label = 'Force experienced by driver')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.title(f'Response to input frequency ω = {frequencies[1]} rad/s, Gain = {gain:.4f}')
# %%
 
#Driver Response
plt.plot(t_newout, z2_new * m2, label = 'Force experienced by driver')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.title(f'Force experienced by driver ω = {frequencies[1]} rad/s')
# %%
