import os

import matplotlib.pyplot as plt
import numpy as np
import xlwt
from scipy.integrate import solve_ivp
from scipy.misc import derivative
from scipy.optimize import root_scalar

# Read text file for input
input_file = open("input.txt", "r")
input_lines = input_file.readlines()
beta = float(input_lines[0])  # Rate of infection
gamma = float(input_lines[1])  # Rate of recovery
theta = float(input_lines[2])  # Rate of loss of immunity
birth_rate = float(input_lines[3])
aging_rate = float(input_lines[4])
death_rate_old = float(input_lines[5])
death_infection_young = float(input_lines[6])
death_infection_old = float(input_lines[7])
birth_rate_step = float(input_lines[8])
aging_rate_step = float(input_lines[9])
death_rate_old_step = float(input_lines[10])
death_infection_young_step = float(input_lines[11])
death_infection_old_step = float(input_lines[12])
iteration_max = int(input_lines[13])
iteration_step = float(input_lines[14])
population_step = 0
iteration = 0
initial_population = 1
save_type = '.jpg'


# Death rate for young individuals
def death_rate_young(young_fraction):
    return 1 / (5 + (np.e ** (20 * (1 / young_fraction - 1 - 0.18)))) + 0.15


# Defines the ODEs for the SIR and age groups
def s_prime(susceptible, infected, recovered, young, old, population):
    return -beta * susceptible * infected / population + theta * recovered + birth_rate * young - \
           death_rate_young(young / population) * (young / population) * susceptible - \
           death_rate_old * (old / population) * susceptible


def i_prime(susceptible, infected, young, old, population):
    return beta * susceptible * infected / population - gamma * infected - \
           death_rate_young(young / population) * (young / population) * infected - \
           death_rate_old * (old / population) * infected - \
           death_infection_young * (young / population) * infected - \
           death_infection_old * (old / population) * infected


def r_prime(infected, recovered, young, old, population):
    return gamma * infected - theta * recovered - \
           death_rate_young(young / population) * (young / population) * recovered - \
           death_rate_old * (old / population) * recovered


def n_prime(infected, young, old, population):
    return birth_rate * young - death_rate_young(young / population) * young - \
           death_rate_old * old - death_infection_young * (young / population) * infected - \
           death_infection_old * (old / population) * infected


def y_prime(infected, young, population):
    return birth_rate * young - death_rate_young(young / population) * young - \
           death_infection_young * (young / population) * infected - \
           aging_rate * young


def o_prime(infected, young, old, population):
    return aging_rate * young - death_rate_old * old - \
           death_infection_old * (old / population) * infected


def age_ratio_prime(young_fraction):
    return young_fraction * (death_rate_young(young_fraction) - birth_rate - death_rate_old) - \
           death_rate_young(young_fraction) + birth_rate + death_rate_old - aging_rate


def sir_model(t, y):
    # Assigning SIR based on a vector
    susceptible = y[0]
    infected = y[1]
    recovered = y[2]
    population = y[3]
    young = y[4]
    old = y[5]

    # ODEs for SIR model
    dSdt = s_prime(susceptible, infected, recovered, young, old, population)
    dIdt = i_prime(susceptible, infected, young, old, population)
    dRdt = r_prime(infected, recovered, young, old, population)
    dNdt = n_prime(infected, young, old, population)
    dYdt = y_prime(infected, young, population)
    dOdt = o_prime(infected, young, old, population)

    return [dSdt, dIdt, dRdt, dNdt, dYdt, dOdt]


# Stops the simulation when the sum of the change in percentage of SIR groups is close to 0
def event(t, y):
    # 0.001 is added to the compartments to avoid a ZeroDivisionError
    dSNdt_SN = (s_prime(y[0], y[1], y[2], y[4], y[5], y[3]) * y[3] -
                n_prime(y[1], y[4], y[5], y[3]) * y[0]) / ((y[0] + 0.001) * y[3])
    dINdt_IN = (i_prime(y[0], y[1], y[4], y[5], y[3]) * y[3] -
                n_prime(y[1], y[4], y[5], y[3]) * y[1]) / ((y[1] + 0.001) * y[3])
    dRNdt_RN = (r_prime(y[1], y[2], y[4], y[5], y[3]) * y[3] -
                n_prime(y[1], y[4], y[5], y[3]) * y[2]) / ((y[2] + 0.001) * y[3])
    return abs(dSNdt_SN) + abs(dINdt_IN) + abs(dRNdt_RN) - 0.01


event.direction = -1
event.terminal = True


def the_magic(filename):
    # Solves for the young to old ratio before the pathogen
    young_fraction_list = []
    growth_rate_list = []
    for num in range(1, 11):
        root = root_scalar(age_ratio_prime, method="secant",
                           x0=num * 0.1, x1=num * 0.1 + 0.5).root
        if root not in young_fraction_list and derivative(age_ratio_prime, root) < 0:
            young_fraction_list.append(root)
            growth_rate_list.append(birth_rate - aging_rate - death_rate_young(root))

    # Optimizes the young fraction by choosing the fraction with the greatest growth rate
    young_fraction = young_fraction_list[growth_rate_list.index(max(growth_rate_list))]

    """ This gives a rough minimum of infected as to not go many generations without significant changes in SIR groups.
    If this number is too low to start an epidemic, but an epidemic is possible, the highest possible I that will
    generate an epidemic will be used. 0.001 was determined after testing with multiple graphs to ensure that no trends
    are missed, while at the same time not leaving too long where the graph is just 3 straight lines."""
    min_infected = 0.001 * initial_population
    if min_infected >= (initial_population * (beta / gamma - 1)) / (beta / gamma):  # Maximum I for an epidemic
        min_infected = (initial_population * (beta / gamma - 1)) / (beta / gamma)
    # Starting vector for SIR
    sirn_vector = [initial_population - min_infected, min_infected, 0, initial_population,
                   young_fraction * initial_population,
                   (1 - young_fraction) * initial_population]

    # Check for possibility of endemic equilibrium
    # From Mena-Lorcat, J., & Hethcote, H. (1992). Dynamic models of infectious diseases as regulators of population
    # sizes. Journal of Mathematical Biology, 30(7).
    """r = birth_rate - death_rate
    sigma = beta / (gamma + death_infection + death_rate)
    if beta / (gamma + death_infection + birth_rate) < 1 or \
            ((r * sigma) / (death_infection * (sigma - 1))) * (1 + (gamma / (theta + death_rate))) < 1:
        print("Disease or population dies out, so no endemic equilibrium; " +
              "take results with a grain of salt")"""

    print(sirn_vector)

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
    y_data = solution[4]
    o_data = solution[5]

    # Getting the final SIR and age values
    final_S = s_data[-1]
    final_I = i_data[-1]
    final_R = r_data[-1]
    final_N = n_data[-1]
    final_Y = y_data[-1]
    final_O = o_data[-1]

    # Getting the growth rates
    initial_GR = n_prime(i_data[0], y_data[0], o_data[0], n_data[0]) / n_data[0]
    final_GR = n_prime(final_I, final_Y, final_O, final_N) / final_N
    dNdt_N = []
    for num in range(0, len(n_data)):
        dNdt_N.append(n_prime(i_data[num], y_data[num], o_data[num], n_data[num]) / n_data[num])

    # Plot the data
    fig1 = plt.figure(0)
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(t, s_data, color='tab:orange', label='Susceptible')
    ax.plot(t, i_data, color='tab:red', label='Infected')
    ax.plot(t, r_data, color='tab:blue', label='Recovered')
    ax.plot(t, dNdt_N, color='tab:green', label='Growth rate')
    ax.legend()
    plt.title('SIR Trajectory')

    # Create labels
    plt.xlabel('t')
    plt.ylabel('SIR Populations')

    # Set limits
    ax.set_xlim([0, t[-1]])
    ax.set_ylim([0, max(n_data)])

    # Save figure 1
    plt.savefig("Figure1Plots/" + filename)

    # Getting the SIR percentage values
    s_percent = s_data / n_data
    i_percent = i_data / n_data
    r_percent = r_data / n_data

    # Plotting percentages
    fig2 = plt.figure(1)
    percent_sir = fig2.add_subplot(1, 1, 1)
    percent_sir.plot(t, s_percent, color='tab:orange', label='Susceptible')
    percent_sir.plot(t, i_percent, color='tab:red', label='Infected')
    percent_sir.plot(t, r_percent, color='tab:blue', label='Recovered')
    percent_sir.legend()
    plt.title('SIR Proportion Trajectory')

    # Create labels
    plt.xlabel('t')
    plt.ylabel('SIR Percentages')

    # Set limits
    percent_sir.set_xlim([0, t[-1]])
    percent_sir.set_ylim([0, 1])

    # Save figure 2
    plt.savefig("Figure2Plots/" + filename)

    # Getting the age percentage values
    y_percent = y_data / n_data
    o_percent = o_data / n_data

    # Plotting percentages
    fig3 = plt.figure(2)
    percent_age = fig3.add_subplot(1, 1, 1)
    percent_age.plot(t, y_percent, color='tab:orange', label='Young')
    percent_age.plot(t, o_percent, color='tab:red', label='Old')
    percent_age.legend()
    plt.title('Age Proportion Trajectory')

    # Create labels
    plt.xlabel('t')
    plt.ylabel('Age Percentages')

    # Set limits
    percent_age.set_xlim([0, t[-1]])
    percent_age.set_ylim([0, 1])

    # Save figure 3
    plt.savefig("Figure3Plots/" + filename)

    # Display the plot
    plt.show()

    # Initialize an excel workbook
    book = xlwt.Workbook(encoding="utf-8")

    # Add a sheet
    sheet1 = book.add_sheet("Sheet1")

    # Exporting the SIR data
    sheet1.write(1, 0, "S - young")
    sheet1.write(2, 0, "I - young")
    sheet1.write(3, 0, "R - young")
    sheet1.write(0, 1, "# of hosts")
    sheet1.write(1, 1, final_S * final_Y / final_N)
    sheet1.write(2, 1, final_I * final_Y / final_N)
    sheet1.write(3, 1, final_R * final_Y / final_N)
    sheet1.write(0, 2, "% of hosts")
    sheet1.write(1, 2, final_S * final_Y / final_N ** 2 * 100)
    sheet1.write(2, 2, final_I * final_Y / final_N ** 2 * 100)
    sheet1.write(3, 2, final_R * final_Y / final_N ** 2 * 100)
    sheet1.write(4, 0, "S - old")
    sheet1.write(5, 0, "I - old")
    sheet1.write(6, 0, "R - old")
    sheet1.write(4, 1, final_S * final_O / final_N)
    sheet1.write(5, 1, final_I * final_O / final_N)
    sheet1.write(6, 1, final_R * final_O / final_N)
    sheet1.write(4, 2, final_S * final_O / final_N ** 2 * 100)
    sheet1.write(5, 2, final_I * final_O / final_N ** 2 * 100)
    sheet1.write(6, 2, final_R * final_O / final_N ** 2 * 100)
    sheet1.write(8, 0, "Initial Growth Rate")
    sheet1.write(8, 1, initial_GR)
    sheet1.write(9, 0, "Final Growth Rate")
    sheet1.write(9, 1, final_GR)
    sheet1.write(10, 0, "Parameter Regime")
    if final_GR >= initial_GR:
        sheet1.write(10, 1, "Type 1: GR Increase")
    else:
        if (death_rate_young(y_data[0] / o_data[0])) < death_rate_young(y_data[-1] / o_data[-1]):
            sheet1.write(10, 1, "Type 2: GR Decrease Due to Collaboration Loss")
        else:
            sheet1.write(10, 1, "Type 3: Massive population loss")

    # Saves the workbook
    book.save("SIR_spreadsheet.xls")


os.makedirs("Figure1Plots", exist_ok=True)
os.makedirs("Figure2Plots", exist_ok=True)
os.makedirs("Figure3Plots", exist_ok=True)
while iteration < iteration_max:
    file_name = 'BR' + str(birth_rate) + ' AR' + str(aging_rate) + ' DRO' + str(death_rate_old) + \
                ' DIY' + str(death_infection_young) + ' DIO' + str(death_infection_old) + \
                ' P' + str(initial_population) + save_type
    # Defines current parameter iteration
    the_magic(file_name)
    birth_rate += birth_rate_step
    aging_rate += aging_rate_step
    death_rate_old += death_rate_old_step
    death_infection_young += death_infection_young_step
    death_infection_old += death_infection_old_step
    initial_population += population_step
    iteration += iteration_step
