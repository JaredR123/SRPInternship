import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Input for rate constants and population size
beta = float(input("Beta = "))
gamma = float(input("Gamma = "))
population = int(input("Population size = "))

# Manual input or automatic starting I
model_method = input("Would you like to input starting SIR groups manually, or would you like the "
                     "program to automatically start with an acceptable I? (m for manual, a for automatic) ")

""" This gives a rough minimum of infected as to not go many generations without significant changes in SIR groups.
If this number is too low to start an epidemic, but an epidemic is possible, the highest possible I that will generate
an epidemic will be used. 0.001 was determined after testing with multiple graphs to ensure that no trends are missed,
while at the same time not leaving too long where the graph is just 3 straight lines."""
if model_method == "a":
    min_infected = 0.001 * population
    if min_infected >= (population * (beta / gamma - 1)) / (beta / gamma):  # this is the maximum I for an epidemic
        min_infected = (population * (beta / gamma - 1)) / (beta / gamma)
    # Starting vector for SIR
    sir_vector = [population - min_infected, min_infected, 0]

else:
    sir_vector = [float(input("S = ")), float(input("I = ")), float(input("R = "))]

if population != sir_vector[0] + sir_vector[1] + sir_vector[2]:
    print("Sum of SIR and N are not the same, graph may be inaccurate because of this")    
    
# check for possibility of epidemic
if beta / gamma < 1 or sir_vector[0] * (beta / gamma) < population:
    print("No epidemic can occur, take graphical results with a grain of salt")


# defines the ODEs for the SIR groups
def s_prime(beta, susceptible, infected):
    return -beta * susceptible * infected / population


def i_prime(beta, gamma, susceptible, infected):
    return beta * susceptible * infected / population - gamma * infected


def r_prime(gamma, infected):
    return gamma * infected


def sir_model(t, y):
    # assigning SIR based on a vector
    susceptible = y[0]
    infected = y[1]
    recovered = y[2]

    # ODEs for SIR model
    dSdt = s_prime(beta, susceptible, infected)
    dIdt = i_prime(beta, gamma, susceptible, infected)
    dRdt = r_prime(gamma, infected)

    return [dSdt, dIdt, dRdt]


# stops the simulation when the total change of SIR groups is less than 0.5% of the population
def event(t, y):
    return abs(s_prime(beta, y[0], y[1])) + abs(i_prime(beta, gamma, y[0], y[1])) \
           + abs(r_prime(gamma, y[1])) - 0.003 * population


event.direction = -1
event.terminal = True

# Evaluate system of ODEs and get the final t
solved = solve_ivp(sir_model, (0, 1000), sir_vector, events=event, dense_output=True)
final_t = solved.t[-1]

# Extract the solutions and creates 100 steps
t = np.linspace(0, final_t, 100)
solution = solved.sol(t)
s_data = solution[0]
i_data = solution[1]
r_data = solution[2]

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(t, s_data, color='tab:orange', label='Susceptible')
ax.plot(t, i_data, color='tab:red', label='Infected')
ax.plot(t, r_data, color='tab:blue', label='Recovered')
ax.legend()
plt.title('SIR trajectory')

# create labels
plt.xlabel('t')
plt.ylabel('SIR Populations')

# set the limits
ax.set_xlim([0, t[-1]])
ax.set_ylim([0, population])

# display the plot
plt.show()
