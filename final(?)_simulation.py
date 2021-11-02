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
beta = float(input_lines[0][5:])  # Rate of infection
gamma = float(input_lines[1][5:])  # Rate of recovery
theta = float(input_lines[2][5:])  # Rate of loss of immunity
birth_rate = float(input_lines[3][5:])
aging_rate = float(input_lines[4][5:])
death_rate_old = float(input_lines[5][5:])
death_infection_young = float(input_lines[6][7:])
death_infection_old = float(input_lines[7][7:])
a = float(input_lines[8][4:])  # Collaboration
b = float(input_lines[9][4:])  # Competition
birth_rate_step = float(input_lines[10])
aging_rate_step = float(input_lines[11])
death_rate_old_step = float(input_lines[12])
death_infection_young_step = float(input_lines[13])
death_infection_old_step = float(input_lines[14])
a_step = float(input_lines[15])
b_step = float(input_lines[16])
iteration_max = int(input_lines[17])
iteration_step = float(input_lines[18])
equivalent_gr = input_lines[19]
population_step = 0
iteration = 0
initial_population = 1
young_death_list = []
old_death_list = []
save_type = '.jpg'


# Proportion of young population that has enough "resources" to give birth
def fertile_proportion(young_fraction):
    return (1 - young_fraction ** (1000 ** (0.5 - a))) * \
           young_fraction ** (1000 ** (b - 0.5))


# Defines the ODEs for the SIR and age groups (Add young/old derivatives for each compartment)
def sy_prime(y_susceptible, infected, y_recovered, young, population):
    return -beta * y_susceptible * infected / population + theta * y_recovered + \
           birth_rate * young * fertile_proportion(young / population) - aging_rate * y_susceptible


def so_prime(y_susceptible, o_susceptible, infected, o_recovered, population):
    return -beta * o_susceptible * infected / population + theta * o_recovered - \
           death_rate_old * o_susceptible + aging_rate * y_susceptible


def iy_prime(y_susceptible, infected, y_infected, young, population):
    return beta * y_susceptible * infected / population - gamma * y_infected - \
           death_infection_young * y_infected - aging_rate * y_infected


def io_prime(o_susceptible, y_infected, o_infected, population):
    return beta * o_susceptible * (y_infected + o_infected) / population - gamma * o_infected - \
           death_rate_old * o_infected - death_infection_old * o_infected + \
           aging_rate * y_infected


def ry_prime(y_infected, y_recovered, young, population):
    return gamma * y_infected - theta * y_recovered - \
           aging_rate * y_recovered


def ro_prime(o_infected, y_recovered, o_recovered):
    return gamma * o_infected - theta * o_recovered - \
           death_rate_old * o_recovered + aging_rate * y_recovered


def s_prime(y_susceptible, o_susceptible, infected, recovered, young, population):
    return -beta * (y_susceptible + o_susceptible) * infected / population + theta * recovered \
           + birth_rate * young * fertile_proportion(young / population) - \
           death_rate_old * o_susceptible


def i_prime(susceptible, y_infected, o_infected, young, population):
    return beta * susceptible * (y_infected + o_infected) / population - \
           gamma * (y_infected + o_infected) - \
           death_rate_old * o_infected - \
           death_infection_young * y_infected - \
           death_infection_old * o_infected


def r_prime(infected, y_recovered, o_recovered, young, population):
    return gamma * infected - theta * (y_recovered + o_recovered) - \
           death_rate_old * o_recovered


def n_prime(y_infected, o_infected, young, old, population):
    return birth_rate * young * fertile_proportion(young / population) - \
           death_rate_old * old - death_infection_young * y_infected - \
           death_infection_old * o_infected


def y_prime(y_infected, young, population):
    return birth_rate * young * fertile_proportion(young / population) - \
           death_infection_young * y_infected - aging_rate * young


def o_prime(o_infected, young, old):
    return aging_rate * young - death_rate_old * old - \
           death_infection_old * o_infected


def age_ratio_prime(young_fraction):
    return birth_rate * fertile_proportion(young_fraction) + death_rate_old - aging_rate - \
           young_fraction * (birth_rate * fertile_proportion(young_fraction) + death_rate_old)


def sir_model(t, y):
    # Assigning SIR based on vector
    y_susceptible = y[0]
    o_susceptible = y[1]
    y_infected = y[2]
    o_infected = y[3]
    y_recovered = y[4]
    o_recovered = y[5]
    population = y[6]
    young = y[7]
    old = y[8]

    # ODES for SIR model
    dSYdt = sy_prime(y_susceptible, y_infected + o_infected, y_recovered, young, population)
    dSOdt = so_prime(y_susceptible, o_susceptible, y_infected + o_infected, o_recovered, population)
    dIYdt = iy_prime(y_susceptible, y_infected + o_infected, y_infected, young, population)
    dIOdt = io_prime(o_susceptible, y_infected, o_infected, population)
    dRYdt = ry_prime(y_infected, y_recovered, young, population)
    dROdt = ro_prime(o_infected, y_recovered, o_recovered)
    dNdt = dSYdt + dIYdt + dRYdt + dSOdt + dIOdt + dROdt  # n_prime(y_infected, o_infected, young, old, population)
    dYdt = dSYdt + dIYdt + dRYdt  # y_prime(y_infected, young, population)
    dOdt = dSOdt + dIOdt + dROdt  # o_prime(o_infected, young, old)

    return [dSYdt, dSOdt, dIYdt, dIOdt, dRYdt, dROdt, dNdt, dYdt, dOdt]


# Stops the simulation when the sum of the change in percentage of SIR groups is close to 0
def event(t, y):
    dYNdt_YN = (y_prime(y[2], y[7], y[6]) * y[6] -
                n_prime(y[2], y[3], y[7], y[8], y[6]) * y[7]) / (y[8] * y[6])
    dONdt_ON = (o_prime(y[3], y[7], y[8]) * y[6] -
                n_prime(y[2], y[3], y[7], y[8], y[6]) * y[8]) / (y[8] * y[6])
    return abs(dYNdt_YN) + abs(dONdt_ON) - 0.001


event.direction = -1
event.terminal = True


def equivalent_GR_finder(growth_rate, sirn_vector):
    # Finds the young death rate for which the growth rate is equivalent to if there was an old death rate
    global death_infection_old
    global death_infection_young
    prev_death_infection_old = death_infection_old
    death_infection_young = 0
    death_infection_old = 0
    growth_rate_list = []
    complete = False
    while not complete:
        death_infection_young += 0.001
        solved = solve_ivp(sir_model, (0, 1000), sirn_vector, events=event)
        solution = solved.y
        final_IY = solution[2][-1]
        final_IO = solution[3][-1]
        final_N = solution[6][-1]
        final_Y = solution[7][-1]
        final_O = solution[8][-1]
        new_growth_rate = n_prime(final_IY, final_IO, final_Y, final_O, final_N) / final_N
        growth_rate_list.append(new_growth_rate)
        if new_growth_rate < growth_rate:
            complete = True
    death_infection_old = prev_death_infection_old
    new_young_death = death_infection_young
    death_infection_young = 0
    if len(growth_rate_list) == 1:
        return new_young_death
    elif growth_rate - growth_rate_list[-1] < growth_rate_list[-2] - growth_rate:
        return new_young_death
    else:
        return new_young_death - 0.001


def the_magic(filename):
    # Solves for the young to old ratio before the pathogen
    young_fraction_list = []
    growth_rate_list = []
    for num in range(1, 11):
        root = root_scalar(age_ratio_prime, method="secant",
                           x0=num * 0.1, x1=num * 0.1 + 0.5).root
        if not isinstance(root, complex) and derivative(age_ratio_prime, root, dx=1e-6) < 0:
            young_fraction_list.append(root)
            growth_rate_list.append(birth_rate * fertile_proportion(root) - aging_rate)

    # Optimizes the young fraction by choosing the fraction with the greatest growth rate
    young_fraction = young_fraction_list[growth_rate_list.index(max(growth_rate_list))]

    # This gives a rough minimum of infected as to not go many generations without significant changes in SIR groups.
    min_infected = 0.001 * initial_population
    # Starting vector for SIR
    sirn_vector = [(initial_population - min_infected) * young_fraction,
                   (initial_population - min_infected) * (1 - young_fraction),
                   min_infected * young_fraction, min_infected * (1 - young_fraction),
                   0, 0, initial_population, young_fraction * initial_population,
                   (1 - young_fraction) * initial_population]

    # Evaluate system of ODEs and get the final t at equilibrium
    solved = solve_ivp(sir_model, (0, 1000), sirn_vector, events=event, dense_output=True)
    final_t = solved.t[-1]

    # Extract the solutions and creates 150 steps
    t = np.linspace(0, final_t, 150)
    solution = solved.sol(t)
    sy_data = solution[0]
    so_data = solution[1]
    iy_data = solution[2]
    io_data = solution[3]
    ry_data = solution[4]
    ro_data = solution[5]
    s_data = [x + y for x, y in zip(sy_data, so_data)]
    i_data = [x + y for x, y in zip(iy_data, io_data)]
    r_data = [x + y for x, y in zip(ry_data, ro_data)]
    n_data = solution[6]
    y_data = solution[7]
    o_data = solution[8]

    # Getting the final SIR and age values
    final_SY = sy_data[-1]
    final_SO = so_data[-1]
    final_IY = iy_data[-1]
    final_IO = io_data[-1]
    final_RY = ry_data[-1]
    final_RO = ro_data[-1]
    final_N = n_data[-1]
    final_Y = y_data[-1]
    final_O = o_data[-1]

    # Getting the growth rate, growth rate components, and young/old derivatives
    initial_GR = birth_rate * fertile_proportion(young_fraction) - aging_rate
    final_GR = n_prime(final_IY, final_IO, final_Y, final_O, final_N) / final_N
    dNdt_N = []
    dBdt_N = []
    dDdt_N = []
    dDIdt_N = []
    dNdt = []
    for num in range(0, len(n_data)):
        dNdt_N.append(n_prime(iy_data[num], io_data[num], y_data[num],
                              o_data[num], n_data[num]) / n_data[num])
        dBdt_N.append(birth_rate * y_data[num] *
                      fertile_proportion(y_data[num] / n_data[num]) / n_data[num])
        dDdt_N.append((-death_rate_old * o_data[num]) / n_data[num])
        dDIdt_N.append((-death_infection_young * iy_data[num] -
                        death_infection_old * io_data[num]) / n_data[num])
        dNdt.append(n_prime(iy_data[num], io_data[num], y_data[num],
                            o_data[num], n_data[num]))

    # Plot the data
    fig1 = plt.figure(0)
    ax = fig1.add_subplot(1, 1, 1)
    ax.plot(t, s_data, color='tab:orange', label='Susceptible')
    ax.plot(t, i_data, color='tab:red', label='Infected')
    ax.plot(t, r_data, color='tab:blue', label='Recovered')
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
    # percent_age.set_xlim([0, t[-1]])
    # percent_age.set_ylim([0.2, 0.8])

    # Save figure 3
    plt.savefig("Figure3Plots/" + filename)

    # Plotting growth rate
    fig4 = plt.figure(3)
    growth_rate = fig4.add_subplot(1, 1, 1)
    growth_rate.plot(t, dNdt_N, color='tab:purple', label='Growth Rate')
    # growth_rate.plot(t, dBdt_N, color='tab:green', label='Birth Rate')
    # growth_rate.plot(t, dDdt_N, color='tab:orange', label='Death Rate')
    # growth_rate.plot(t, dDIdt_N, color='tab:red', label='Infection Death Rate')
    growth_rate.legend()
    plt.title('Growth Rate Trajectory')

    # Create labels
    plt.xlabel('t')
    plt.ylabel('Growth rate')

    # Save figure 4
    plt.savefig("Figure4Plots/" + filename)

    plt.show()

    if equivalent_gr == "y":
        equivalent_young_death = equivalent_GR_finder(final_GR, sirn_vector)
        print(equivalent_young_death)
        print(death_infection_old)
        young_death_list.append(equivalent_young_death)

    # Initialize an excel workbook
    book = xlwt.Workbook(encoding="utf-8")

    # Add a sheet
    sheet1 = book.add_sheet("Sheet1")

    # Exporting the SIR data
    sheet1.write(1, 0, "S - young")
    sheet1.write(2, 0, "I - young")
    sheet1.write(3, 0, "R - young")
    sheet1.write(0, 1, "# of hosts")
    sheet1.write(1, 1, final_SY)
    sheet1.write(2, 1, final_IY)
    sheet1.write(3, 1, final_RY)
    sheet1.write(0, 2, "% of hosts")
    sheet1.write(1, 2, final_SY / final_N * 100)
    sheet1.write(2, 2, final_IY / final_N * 100)
    sheet1.write(3, 2, final_RY / final_N * 100)
    sheet1.write(4, 0, "S - old")
    sheet1.write(5, 0, "I - old")
    sheet1.write(6, 0, "R - old")
    sheet1.write(4, 1, final_SO)
    sheet1.write(5, 1, final_IO)
    sheet1.write(6, 1, final_RO)
    sheet1.write(4, 2, final_SO / final_N * 100)
    sheet1.write(5, 2, final_IO / final_N * 100)
    sheet1.write(6, 2, final_RO / final_N * 100)
    sheet1.write(8, 0, "Initial Growth Rate")
    sheet1.write(8, 1, initial_GR)
    sheet1.write(9, 0, "Final Growth Rate")
    sheet1.write(9, 1, final_GR)
    sheet1.write(10, 0, "Parameter Regime")
    if final_GR >= initial_GR:
        sheet1.write(10, 1, "Type 1: GR Increase")
    else:
        if (fertile_proportion(y_data[0] / n_data[0])) < fertile_proportion(y_data[-1] / n_data[-1]):
            sheet1.write(10, 1, "Type 2: GR Decrease Due to Collaboration Loss")
        else:
            sheet1.write(10, 1, "Type 3: Massive population loss")

    # Saves the workbook
    book.save("SIR_spreadsheet.xls")


os.makedirs("Figure1Plots", exist_ok=True)
os.makedirs("Figure2Plots", exist_ok=True)
os.makedirs("Figure3Plots", exist_ok=True)
os.makedirs("Figure4Plots", exist_ok=True)
while iteration < iteration_max:
    file_name = 'BR' + str(birth_rate) + ' AR' + str(aging_rate) + ' DRO' + str(death_rate_old) + \
                ' DIY' + str(death_infection_young) + ' DIO' + str(death_infection_old) + \
                ' P' + str(initial_population) + save_type
    # Defines current parameter iteration
    the_magic(file_name)
    old_death_list.append(death_infection_old)
    birth_rate += birth_rate_step
    aging_rate += aging_rate_step
    death_rate_old += death_rate_old_step
    death_infection_young += death_infection_young_step
    death_infection_old += death_infection_old_step
    a += a_step
    initial_population += population_step
    iteration += iteration_step


# Plotting death rates
fig5 = plt.figure(4)
death_rates = fig5.add_subplot(1, 1, 1)
death_rates.plot(old_death_list, young_death_list, color='tab:red', label='Death Rates')
death_rates.legend()
plt.title('Infection Death Rates for Equivalent Growth Rates')

# Create labels
plt.xlabel('Old Infection Death Rate')
plt.ylabel('Young Infection Death Rate')

# Display the plot
plt.show()
