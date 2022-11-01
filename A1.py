import math
import numpy as np
import matplotlib.pyplot as plt


# ----- 2 -----
# ANALYTICAL SOLUTION FOR THE PROPAGATION OF
# SURFACE WATER WAVES INTO A HOMOGENOUS AQUIFER:
# /EQ.(5) FROM SAWYER/
def an_solution(x,t,A,D,omega,phi):
    return A*np.exp(-x*np.sqrt(omega/(2*D)))*np.sin((-x*np.sqrt(omega/(2*D)))+(omega*t)+phi)

# ----- 3 -----
# WE BEGIN BY DEFINING THE REQUIRED VARIABLES.
realizations = 100
x = np.array([1, 10, 100])    # x ARRAY GIVEN IN ASSIGNMENT.
t = np.array([7, 30, 180])    # t ARRAY GIVEN IN ASSIGNMENT.
Sy = 0.25                     # SPECIFIC YIELD VALUE FOUND IN SAWYER.
b = 10                        # SATURATED THICKNESS.
phi = 0
m_s2_to_m_d2 = (24*60*60)**2  # CONVERSION FACTOR FROM m/s^2 TO m/d^2

# SAMPLING PARAMETERS FROM GIVEN DISTRIBUTIONS
D = np.exp(np.random.uniform(low=1E-04, high=10, size=realizations)*b/Sy) * m_s2_to_m_d2
A = np.exp([np.random.uniform(low=-1, high=1, size=realizations)])
omega = np.array([np.random.uniform(low=(2 * math.pi)/7, high=math.pi, size=realizations)])

# CREATING A PLOT OBJECT
fig, axs = plt.subplots(nrows=3, ncols=3, constrained_layout=True, sharex=True, sharey=True)
fig.suptitle("Computed Head-Value Histograms")
fig.supxlabel("Head Changes in Meters")
fig.supylabel("Number of Observations")

# CREATING MATRIX TO STORE ANALYTIC SOLUTIONS COMPUTED FROM REALIZATIONS.
an_sol_mat = np.zeros((realizations, x.size*t.size))
sol_column_counter = 0
for idx in range(x.size):
    for jdx in range(t.size):
        an_sol_mat[:, sol_column_counter] = an_solution(x[idx], t[jdx], A, D, omega, phi)
        axs[idx, jdx].hist(an_sol_mat[:, sol_column_counter])                 # PLOTTING HISTOGRAMS.
        axs[idx, jdx].set_title("x = {}m, t = {}d".format(x[idx], t[jdx]))    # SUBPLOT TITLES.
        sol_column_counter = sol_column_counter + 1

plt.show()
