# %% 

import numpy as np
import matplotlib.pyplot as plt
import control
from control.matlab import * 

# %%
# Given parameters
m1 = 20  # kg
m2 = 50  # kg
k1 = 7410  # N/m
k2 = 8230  # N/m
b1 = 1430  # Ns/m
b2 = 153  # Ns/m

# State-space matrices
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
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

D = np.zeros((2, 2))

# Create state-space system
sys = control.StateSpace(A, B, C, D)

# Calculate eigenvalues (poles)
poles = np.linalg.eigvals(A)
print("System poles:")
for i, pole in enumerate(poles):
    print(f"Pole {i+1}: {pole:.4f}")
# %%
# Sort poles by magnitude of real part
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
if slow_poles[0].imag != 0:  # Complex conjugate pair
    natural_freq = abs(slow_poles[0])   # |s| = sqrt(real^2 + imag^2)
    damping_ratio = -slow_poles[0].real / natural_freq
    time_constant = 1 / (damping_ratio * natural_freq)
    print(f"\nSlow poles natural frequency: {natural_freq:.4f} rad/s")
    print(f"Damping ratio: {damping_ratio:.4f}")
    print(f"Time constant: {time_constant:.4f} s")

plt.axis('equal')
max_range = max(abs(np.max(poles.real)), abs(np.min(poles.real)), 
                abs(np.max(poles.imag)), abs(np.min(poles.imag)))
plt.xlim(-max_range*1.1, 0.1*max_range)
plt.ylim(-max_range*0.6, max_range*0.6)
plt.figure(figsize=(10, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid()

# Plot all poles
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

# Adjust zoom to show all poles clearly
# %%

# Frequencies to analyze
frequencies = [1.57, 6.28, 25.13]  # rad/s

# Time vector (ensure it covers several periods of the slowest frequency)
t = np.linspace(0, 10, 10000)

# Create figure for plotting
plt.figure(figsize=(15, 12))
gains = []

# Run simulation for each frequency
for i, omega in enumerate(frequencies):
    # Input signals
    u1 = 0.02 * np.sin(omega * t)  # z0(t) = 0.02*sin(omega*t)
    u2 = 0.02 * omega * np.cos(omega * t)  # z0_dot(t) = derivative of z0(t)
    u = np.column_stack((u1, u2))
    y, t_out, x = lsim(sys, u, t)
    z2 = y
    
    #start_idx = np.where(t >= 5)[0][0]
    max_input = np.max(np.abs(u1))
    max_output = np.max(np.abs(z2))
    gain = max_output / max_input
    gains.append(gain)
    
    # Plot results
    plt.subplot(3, 1, i+1)
    plt.plot(t, u1, 'b-', label='Input z0(t)')
    plt.plot(t, z2, 'r-', label='Driver displacement z2(t)')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (m)')
    plt.title(f'Response to input frequency ω = {omega} rad/s, Gain = {gain:.4f}')
    
    # Add vertical line to indicate when steady state is reached
    plt.axvline(x=5, color='gray', linestyle='--', alpha=0.5)
    
    # Determine if amplification or attenuation occurs
    if gain > 1:
        behavior = "amplification"
    else:
        behavior = "attenuation"
    
    plt.annotate(f'Gain = {gain:.4f} ({behavior})', 
                 xy=(0.1, 0.85), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange", alpha=0.8))
    
    plt.legend()

# %%
