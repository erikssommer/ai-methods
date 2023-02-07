from scipy.optimize import fsolve
import numpy as np

def equation(R_indiff):
    return -np.exp(-100/R_indiff) - 0.5 * np.exp(-500/R_indiff) + 0.5

R_indiff = fsolve(equation, 100)[0]
print("R_indiff:", R_indiff)