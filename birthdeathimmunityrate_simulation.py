import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import xlwt

# Read text file for input
input_file = open("input.txt", "r")
input_lines = input_file.readlines()
beta = float(input_lines[0])  # Rate of infection
gamma = float(input_lines[1])  # Rate of recovery
theta = float(input_lines[2])  # Rate of loss of immunity
birth_rate = float(input_lines[3])
death_rate = float(input_lines[4])
death_infection = float(input_lines[5])
initial_population = float(input_lines[6])
model_method = input_lines[7]

""" This gives a rough minimum of infected as to not go many generations without significant changes in SIR groups.
If this number is too low to start an epidemic, but an epidemic is possible, the highest possible I that will generate
an epidemic will be used. 0.001 was determined after testing with multiple graphs to ensure that no trends are missed,
while at the same time not leaving too long where the graph is just 3 straight lines."""
if model_method == "a":
    min_infected = 0.001 * initial_population
    if min_infected >= (initial_population * (beta / gamma - 1)) / (beta / gamma):  # Maximum I for an epidemic
        min_infected = (initial_population * (beta / gamma - 1)) / (beta / gamma)
    # Starting vector for SIR
    sirn_vector = [initial_population - min_infected, min_infected, 0, initial_population]

else:
    sirn_vector = [float(input_lines[8]), float(input_lines[9]), float(input_lines[10]), initial_population]

# Check for possibility of epidemic (FIX FIX FIX FIX FIX)
if beta / gamma < 1 or sirn_vector[0] * (beta / gamma) < initial_population:
    print("No epidemic can occur, take graphical results with a grain of salt")


# Defines the ODEs for the SIR groups
def s_prime(susceptible, infected, recovered, population):
    return -beta * susceptible * infected / population + theta * recovered + birth_rate * population - \
           death_rate * susceptible


def i_prime(susceptible, infected, population):
    return beta * susceptible * infected / population - gamma * infected - death_rate * infected \
           - death_infection * infected


def r_prime(infected, recovered):
    return gamma * infected - theta * recovered - death_rate * recovered


def n_prime(infected, population):
    return (birth_rate - death_rate) * population - death_infection * infected


def sir_model(t, y):
    # Assigning SIR based on a vector
    susceptible = y[0]
    infected = y[1]
    recovered = y[2]
    population = y[3]

    # ODEs for SIR model
    dSdt = s_prime(susceptible, infected, recovered, population)
    dIdt = i_prime(susceptible, infected, population)
    dRdt = r_prime(infected, recovered)
    dNdt = dSdt + dIdt + dRdt

    return [dSdt, dIdt, dRdt, dNdt]


# Stops the simulation when the sum of the change in percentage of SIR groups is close to 0
def event(t, y):
    # 0.001 is added to the compartments to avoid a ZeroDivisionError
    dSNdt_SN = (s_prime(y[0], y[1], y[2], y[3]) * y[3] - n_prime(y[1], y[3]) * y[0]) / ((y[0] + 0.001) * y[3])
    dINdt_IN = (i_prime(y[0], y[1], y[3]) * y[3] - n_prime(y[1], y[3]) * y[1]) / ((y[1] + 0.001) * y[3])
    dRNdt_RN = (r_prime(y[1], y[2]) * y[3] - n_prime(y[1], y[3]) * y[2]) / ((y[2] + 0.001) * y[3])
    return abs(dSNdt_SN) + abs(dINdt_IN) + abs(dRNdt_RN) - 0.01


event.direction = -1
event.terminal = True

# Evaluate system of ODEs and get the final t at equilibrium
solved = solve_ivp(sir_model, (0, 1000), sirn_vector, events=event, dense_output=True)
final_t = solved.t[-1]

# Extract the solutions and creates 100 steps
t = np.linspace(0, final_t, 100)
solution = solved.sol(t)
s_data = solution[0]
i_data = solution[1]
r_data = solution[2]
n_data = solution[3]

# Getting the final SIR values
final_S = s_data[-1]
final_I = i_data[-1]
final_R = r_data[-1]
final_N = n_data[-1]

# Plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(t, s_data, color='tab:orange', label='Susceptible')
ax.plot(t, i_data, color='tab:red', label='Infected')
ax.plot(t, r_data, color='tab:blue', label='Recovered')
ax.legend()
plt.title('SIR trajectory')

# Create labels
plt.xlabel('t')
plt.ylabel('SIR Populations')

# Set limits
ax.set_xlim([0, t[-1]])
ax.set_ylim([0, max(n_data)])

# Display the plot
plt.show()

# Initialize an excel workbook
book = xlwt.Workbook(encoding="utf-8")

# Add a sheet
sheet1 = book.add_sheet("Sheet1")

# Exporting the SIR data
sheet1.write(1, 0, "S")
sheet1.write(2, 0, "I")
sheet1.write(3, 0, "R")
sheet1.write(0, 1, "# of people")
sheet1.write(1, 1, final_S)
sheet1.write(2, 1, final_I)
sheet1.write(3, 1, final_R)
sheet1.write(0, 2, "% of people")
sheet1.write(1, 2, final_S/final_N * 100)
sheet1.write(2, 2, final_I/final_N * 100)
sheet1.write(3, 2, final_R/final_N * 100)
sheet1.write(5, 0, "Growth rate")
sheet1.write(5, 1, n_prime(final_I, final_N)/final_N)

# Saves the workbook
book.save("SIR_spreadsheet.xls")
