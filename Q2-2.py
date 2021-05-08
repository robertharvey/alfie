# A program for finding the Poincare section of a Duffing Oscillator
# with 0.1 <= F_e <= 0.43 and gamma = 0.25
# results for the section are output to the file 'Q2-2.dat'

#####################################
# import
import numpy as np
import sys
from scipy.integrate import solve_ivp

###############################################################################
# Information to the user
print('This program, "Q2-2.py", implements Runge-Kutta 4th and 5th order'
      ' ODE solvers to return three 1D arrays "t", "x" and "v" as results of the '
      'initial value problem defined by the coupled ODEs: \n'
      'dv/dt = F_e(cos(omega*t)) -2(gamma)(v) + x - x^3\n'
      'dx/dt = v)')
print('Specifically, the data corresponding to the theta=0 plane is recorded for'
      '0.1 <= F_e <= 0.43')
print('Event results are outputted to the file "Q2-2.dat"\n')

##################################################################################
# first we define the function containing the equations of motion, which here are:
# dv/dt = F_e(cos(omega*t)) -2(gamma)(v) + x - x^3
# dx/dt = v
# with dtheta/dt = omega
def f(t,p):
    #x and v defined using the 2D array p
    x = p[0]
    v = p[1]
    # EoM built using x and v as constituent 1D arrays of the p 2D array
    dvdt = (F_e*np.cos(omega*t)) -(2.0*gamma*v) + (x) - (x**3)
    dxdt = v
    dthetadt = omega
    derivs = [dxdt,dvdt]
    # return relevant derivatives
    return derivs

################################################
# we then define an event function to capture the Poincare section.
# i.e. results only when the trajectory crosses the theta=0 plane.
def event(t,p):
 x = p[0]
 v = p[1]
 theta = omega*t
 return np.cos(theta)-0.0

################################################
# parameters for the function call

#omega : angular frequency

omega = 1.0

# F_e : magnitude of driving force

F_e = [] # F_e declared as list
F_e.append(0.1) # Initial F_e value
df = 0.001 # step in F_e
i = 0 # loop index

# gamma : oscillator damping parameter

gamma = 0.25

# tspan

tmin = 8250.0  # minimum (initial) t value
tmax = 10000.0  # maximum t value, completing range
tspan = (tmin, tmax)  # t range for function call

# initial conditions (x,v(tmin))

init = [1.0,1.0]

# tstep

dt0 = 1e-3  # initial tstep
maxdt = 1e-2  # maximum tstep

# solver tolerance

rtol = 1e-3  # relative tolerance
atol = 1e-3  # absolute tolerance

########################################################
# loop over F_e values with dat file open
# results are written each iteration to "Q2-2.dat"
with open('Q2-2a.dat', 'w') as file:
    while F_e[i] <= 0.43:
        solutions = []
        F_e.append(F_e[i] + df)
        ###########################################################################
        # call the solver, implementing the above parameters
        # and Runge-Kutta ODE solvers of orders 4 and 5
        # results at event criterion are saved
        solutions = solve_ivp(f,
                              tspan,
                              init,
                              method='RK45', # one of RK23, RK45, Radau, BDF, LSODA
                              first_step=dt0,
                              atol=atol,
                              rtol=rtol,
                              events=[event],
                              dense_output=True,
                              max_step=maxdt)

        # debugging: show the return status if non-zero
        if(solutions.status != 0):
            print ("Status of ODE solver is non-zero : something went wrong :(")
            sys.exit

        events = solutions.sol(solutions.t_events[0])


        for data in zip(solutions.t_events[0], events[0], events[1]):
            #print('{0:g} {1:g} {2:g} {3:g}'.format(data[0], data[1], data[2],F_e[i]))
            file.write('{0:g} {1:g} {2:g}\n'.format(data[0], data[1],F_e[i]))
        print(F_e[i])
        i = i + 1

############################################################
# Program complete, output to user
print('\nProgram complete, results outputted successfully.')