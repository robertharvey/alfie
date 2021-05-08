#code for the final CW assignment

#to do:
#convert lists to arrays (would require significant restructure, might not be worth it)
#currently the long period integration encounters a memory error, try getting rid of the eta loop in the function as it's not necessary here and choosing a lower eta value

#This program computes the orbits of 2 '2-body problems', using the leap-frog algorithm with a variable timestep
#if set for each PN which would contain alpha and beta values for every possibility, or a function which would return values based upon inputs

#import
import gc
import resource

import math
import numpy as np
from itertools import zip_longest
from matplotlib import pyplot as plt

#information to the user
print('This program (V_6479942.py) implements a leap-frog algorithm with a variable timestep to compute orbits of earth and mercury in two 2-body systems approximated as one-body systems.')
print('')
print('The mercury system is also used to demonstrate the relation of the relative energy error for the orbit and timestep used.  As the timestep is variable here, an accuracy parameter is used instead.')
print('')
print('Time, position and velocity values are written to the file "orbits_variable.dat"')
print('')
print('Timestep and energy values are written to the file "energy_variable.dat"')
print('')


#1) unit consideration
mpc = 3.086E+13 #mpc in metres
yr = 3600.0*365.0*24.0 #yr in seconds
M_0 = 1.98847E+30 #solar mass in kg
G_SI = 6.67E-11 #G in SI units of m^3kg^-1s^-2 for use in the physical mass unit calculation
G = 1.00  # gravitational constant in units of (AU^3)/(M_0*yr^2)
R = 1.00  # mpc
T = 1.00  # yr
M = ((4.0*(np.pi**2))*(R**3))/((G*(T**2))) #mass in code units derived from kepler's third law
M_SI = ((4.0*(np.pi**2))*(mpc**3))/((G_SI*(yr**2))) #mass unit in SI for conversion to solar masses for initial cons
M_solar = M_SI/M_0
V = R/T #velocity units in mpc/yr
#taking a wavelength of 100m and corresponding frequency of 2.99792458 MHz (arbitrarily chosen), speed of light can be found by the equation:
wavelength = 100.0/mpc
frequency = (2.99792458E+6)*yr
c = wavelength*frequency

print('Mass Units:',M_solar,' Solar Masses')
print('Velocity Units:',V,' milliparsecs per year')
print('Speed of light:',c,' milliparsecs per year')
print('')

#declare parameters and variables
#black hole masses are achieved in computational units by dividing the solar mass value by M_solar
M_1 = (1E+7)/M_solar #Black hole 1 mass
M_2 = (1E+4)/M_solar #Black hole 2 mass
M_t = M_1 + M_2 #total mass of system
nu = (M_1*M_2)/(M_t**2)
a_binary = 0.1 #binary system semi-major axis
e_binary = 0.9 #binary system eccentricity
dt = 0.0
dt_i = [] #initial timestep for mercury for each eta value for plotting



# computation of pericentre and apocentre
r_p = a_binary * (1.0 - e_binary)
r_a = a_binary * (1.0 + e_binary)
# computation of pericentre and apocentre

# computation of pericentre and apocentre velocities
v_p = ((G * M_t / a_binary) * ((1.0 + e_binary) / (1.0 - e_binary))) ** 0.5
v_a = ((G * M_t / a_binary) * ((1.0 - e_binary) / (1.0 + e_binary))) ** 0.5
# computation of pericentre and apocentre velocities

x_2 = []  # x-coordinate of BH2
y_2 = []  # y-coordinate of BH2
z_2 = []  # z-coordinate of BH2
v_x_2 = []  # x-velocity of BH2
v_y_2 = []  # y-velocity of BH2
v_z_2 = []  # z-velocity of BH2
a_x_2 = 0.0 # acceleration values of BH2
a_y_2 = 0.0
a_z_2 = 0.0
t = []  # time
dt = 0.0
dt_i = []
a = [] #semi-major axis
e = [] #eccentricity
eta = 1E-5 #accuracy parameter
p = ((4 * (np.pi ** 2) * (a_binary ** 3)) / (G * M_t)) ** 0.5 #orbital period of Binary
alpha = 0.0
alpha_0 = 0
alpha_1 = 0
alpha_2 = 0
alpha_25 = 0
beta = 0.0
beta_0 = 0
beta_1 = 0
beta_2 = 0
beta_25 = 0
i=0
#v_i = [0.00]*len(eta) #initial velocity of BH2
#v_f = [0.00]*len(eta) #final velocity of BH2
#E_i = [0.00]*len(eta) #initial energy of BH2
#E_f = [0.00]*len(eta) #final energy of BH2
#E_a = [0.00]*len(eta) #absolute energy error of BH2
#E_r = [0.00]*len(eta) #relative energy error of BH2
h_x = 0.0
h_y = 0.0
h_z = 0.0
ecc_x = [] #eccentricity arrays
ecc_y = []
ecc_z = []
ecc_abs = []
phi = []
delta_phi = 0.0


# initial conditions output
x_2.append(r_a)
y_2.append(0.0)
z_2.append(0.0)
v_x_2.append(0.0)
v_y_2.append(v_a)
v_z_2.append(0.0)

f = open("binary.ecc.init.dat", "w")
f.write("{} {} {}\n".format('X position:', float(x_2[0]), "mpc"))
f.write("{} {} {}\n".format('y position:', float(y_2[0]), "mpc"))
f.write("{} {} {}\n".format('Z position:', float(z_2[0]), "mpc"))
f.write("{} {} {}\n".format('X velocity:', float(v_x_2[0]), "mpc/yr"))
f.write("{} {} {}\n".format('Y velocity:', float(v_y_2[0]), "mpc/yr"))
f.write("{} {} {}\n".format('z velocity:', float(v_z_2[0]), "mpc/yr"))
f.close()
# initial conditions output

# info for user
print('Initial X position:', float(x_2[0]), "mpc")
print('Initial Y velocity:', float(v_y_2[0]), "mpc/yr")
print('')

#theoretical advance of pericentre per orbit
delta_phi_GR = (6.0*np.pi*G*M_1)/((c**2)*a_binary*(1.0-(e_binary**2))) #should be in rad/orbit
print('Theoretical Pericentre advance due to GR:',delta_phi_GR,' rad/orbit')

def integration(p,PN_1,PN_2,PN_25):
    gc.collect()

    # variables redeclared in the loop to clear lists for each eta
    x = []  # x-coordinate of mercury
    y = []  # y-coordinate of mercury
    z = []  # z-coordinate of mercury
    v_x = []  # x-velocity of mercury
    v_y = []  # y-velocity of mercury
    v_z = []  # z-velocity of mercury
    a_x = 0.00  # acceleration
    a_y = 0.00  # acceleration
    a_z = 0.00  # acceleration
    t = []  # time
    ecc_x = []
    ecc_y = []
    ecc_z = []
    ecc_abs = []
    phi = []
    a = []
    e = []

    # initial conditions of mercury-sun system
    i = 0  # counter for mercury-sun loop
    x.append(r_a)  # filling first array values
    y.append(0.0)
    z.append(0.0)
    v_x.append(0.0)
    v_y.append(v_a)
    v_z.append(0.0)
    t.append(0.0)
    r = (x[0] ** 2 + y[0] ** 2 + z[0] ** 2) ** 0.5  # the initial position value (=r_a_m but needed for loop)
    v = (v_x[0] ** 2 + v_y[0] ** 2 + v_z[0] ** 2) ** 0.5
    # a_x = (-G * M_t_m * x[i]) / (r ** 3) #computation of initial acceleration values
    # a_y = (-G * M_t_m * y[i]) / (r ** 3)
    # a_z = (-G * M_t_m * z[i]) / (r ** 3)

    # alpha/beta PN modification
    r_dot = ((x[i] * v_x[i]) + (y[i] * v_y[i]) + (z[i] * v_z[i])) / (r)
    alpha_0 = 1.0
    alpha_1 = (-(3.0 / 2.0) * (r_dot ** 2) * nu) + ((1.0 + (3.0 * nu)) * (v ** 2)) - (
                2.0 * (2.0 + nu) * (G * M_t / r))
    alpha_2 = ((15.0 / 8.0) * (r_dot ** 4) * nu * (1 - (3.0 * nu))) + (
                3 * (r_dot ** 2) * nu * (v ** 2) * ((2 * nu) - (3.0 / 2.0))) + (
                            nu * (v ** 4) * (3.0 - (4 * nu))) + ((G * M_t / r) * (
                (-2.0 * (r_dot ** 2) * (1.0 + (nu ** 2))) - (25.0 * r_dot * nu) - (
                    (13.0 / 2.0) * nu * (v ** 2)))) + (
                            ((G ** 2) * (M_t ** 2) / (r ** 2)) * (9.0 + (87.0 * nu / 4.0)))
    alpha_25 = -(8.0 / 5.0) * (G * M_t / r) * nu * r_dot * ((17.0 / 3.0) * (G * M_t / r) + (3.0 * (v ** 2)))
    beta_0 = 0.0
    beta_1 = -2.0 * (2.0 - nu) * r_dot
    beta_2 = (3.0 * (r_dot ** 3) * nu * ((3.0 / 2.0) + nu)) - (
                r_dot * nu * (v ** 2) * ((15.0 / 2.0) + (2.0 * nu))) + (
                            (G * M_t * r_dot / r) * (2.0 + (41.0 * nu / 2.0) + (4.0 * (nu ** 2))))
    beta_25 = (8.0 / 5.0) * (G * M_t / r) * nu * (3.0 * (G * M_t / r) + (v ** 2))

    alpha = alpha_0 + PN_1 * ((c ** -2) * alpha_1) + PN_2 * ((c ** -4) * alpha_2) + PN_25 * ((c ** -5) * alpha_25)
    beta = beta_0 + PN_1 * ((c ** -2) * beta_1) + PN_2 * ((c ** -4) * beta_2) + PN_25 * ((c ** -5) * beta_25)

    a_x = -((G * M_t) / (r ** 2)) * (
                (alpha * (x[i] / r)) + (beta * v_x[i]))  # new acceleration values for PN modifications
    a_y = -((G * M_t) / (r ** 2)) * ((alpha * (y[i] / r)) + (beta * v_y[i]))
    a_z = -((G * M_t) / (r ** 2)) * ((alpha * (z[i] / r)) + (beta * v_z[i]))

    h_x = (y[i] * v_z[i]) - (z[i] * v_y[i])
    h_y = -((x[i] * v_z[i]) - (z[i] * v_x[i]))
    h_z = (x[i] * v_y[i]) - (y[i] * v_x[i])
    h = (h_x ** 2 + h_y ** 2 + h_z ** 2) ** 0.5
    ecc_x.append((((v_y[i] * h_z) - (v_z[i] * h_y)) / (G * M_t)) - (x[i] / r))
    ecc_y.append(((-((v_x[i] * h_z) - (v_z[i] * h_x))) / (G * M_t)) - (y[i] / r))
    ecc_z.append((((v_x[i] * h_y) - (v_y[i] * h_x)) / (G * M_t)) - (z[i] / r))
    ecc_abs.append((ecc_x[i] ** 2 + ecc_y[i] ** 2 + ecc_z[i] ** 2) ** 0.5)
    phi.append(np.arctan(ecc_y[i] / ecc_x[i]))
    a.append(((2.0 / r) - ((v ** 2) / (G * M_t))) ** (-1.0))
    e.append((1 - ((h ** 2) / (G * M_t * a[i]))) ** 0.5)
    dt_i.append(eta * ((r ** 3) / (G * M_t)) ** 0.5)  # initial timestep value

    print('Accuracy Parameter for Binary orbit integration:', eta)
    print('')
    # while loop containing leap-frog algorithm for mercury-sun system, set to run until the computed mercury period is reached
    while t[i] <= p:
        dt = eta * ((r ** 3) / (G * M_t)) ** 0.5  # computation of variable timestep
        t.append(t[i] + dt)  # addition of timestep to current time to find next time

        v_x_temp = v_x[i] + (a_x * (dt / 2.0))  # calculation of temporary velocities
        v_y_temp = v_y[i] + (a_y * (dt / 2.0))
        v_z_temp = v_z[i] + (a_z * (dt / 2.0))

        x.append(x[i] + (v_x_temp * dt))  # calculation of next position value
        y.append(y[i] + (v_y_temp * dt))
        z.append(z[i] + (v_z_temp * dt))
        r = (x[i + 1] ** 2 + y[i + 1] ** 2 + z[
            i + 1] ** 2) ** 0.5  # temporary value to inform of current position for eqns

        # a_x = (-G * M_t_m * x[i+1]) / (r ** 3) #new acceleration values based on new position
        # a_y = (-G * M_t_m * y[i+1]) / (r ** 3)
        # a_z = (-G * M_t_m * z[i+1]) / (r ** 3)

        # alpha/beta PN modification
        r_dot = ((x[i] * v_x[i]) + (y[i] * v_y[i]) + (z[i] * v_z[i])) / (r)
        alpha_0 = 1.0
        alpha_1 = (-(3.0 / 2.0) * (r_dot ** 2) * nu) + ((1.0 + (3.0 * nu)) * (v ** 2)) - (
                    2.0 * (2.0 + nu) * (G * M_t / r))
        alpha_2 = ((15.0 / 8.0) * (r_dot ** 4) * nu * (1 - (3.0 * nu))) + (
                    3 * (r_dot ** 2) * nu * (v ** 2) * ((2 * nu) - (3.0 / 2.0))) + (
                        nu * (v ** 4) * (3.0 - (4 * nu))) + ((G * M_t / r) * (
                    (-2.0 * (r_dot ** 2) * (1.0 + (nu ** 2))) - (25.0 * r_dot * nu) - (
                        (13.0 / 2.0) * nu * (v ** 2)))) + (
                                ((G ** 2) * (M_t ** 2) / (r ** 2)) * (9.0 + (87.0 * nu / 4.0)))
        alpha_25 = -(8.0 / 5.0) * (G * M_t / r) * nu * r_dot * ((17.0 / 3.0) * (G * M_t / r) + (3.0 * (v ** 2)))
        beta_0 = 0.0
        beta_1 = -2.0 * (2.0 - nu) * r_dot
        beta_2 = (3.0 * (r_dot ** 3) * nu * ((3.0 / 2.0) + nu)) - (
                    r_dot * nu * (v ** 2) * ((15.0 / 2.0) + (2.0 * nu))) + (
                                (G * M_t * r_dot / r) * (2.0 + (41.0 * nu / 2.0) + (4.0 * (nu ** 2))))
        beta_25 = (8.0 / 5.0) * (G * M_t / r) * nu * (3.0 * (G * M_t / r) + (v ** 2))

        alpha = alpha_0 + (PN_1 * ((c ** -2) * alpha_1)) + (PN_2 * ((c ** -4) * alpha_2)) + (PN_25 * (
                    (c ** -5) * alpha_25))
        beta = beta_0 + (PN_1 * ((c ** -2) * beta_1)) + (PN_2 * ((c ** -4) * beta_2)) + (PN_25 * ((c ** -5) * beta_25))

        a_x = -((G * M_t) / (r ** 2)) * (
                    (alpha * (x[i] / r)) + (beta * v_x[i]))  # new acceleration values for PN modifications
        a_y = -((G * M_t) / (r ** 2)) * ((alpha * (y[i] / r)) + (beta * v_y[i]))
        a_z = -((G * M_t) / (r ** 2)) * ((alpha * (z[i] / r)) + (beta * v_z[i]))

        v_x.append(
            v_x_temp + (a_x * (dt / 2.0)))  # computation of next velocity values based on new acceleration
        v_y.append(v_y_temp + (a_y * (dt / 2.0)))
        v_z.append(v_z_temp + (a_z * (dt / 2.0)))
        v = (v_x[i + 1] ** 2 + v_y[i + 1] ** 2 + v_z[i + 1] ** 2) ** 0.5

        h_x = (y[i] * v_z[i]) - (z[i] * v_y[i])
        h_y = -((x[i] * v_z[i]) - (z[i] * v_x[i]))
        h_z = (x[i] * v_y[i]) - (y[i] * v_x[i])
        h = (h_x ** 2 + h_y ** 2 + h_z ** 2) ** 0.5
        ecc_x.append((((v_y[i] * h_z) - (v_z[i] * h_y)) / (G * M_t)) - (x[i] / r))
        ecc_y.append(((-((v_x[i] * h_z) - (v_z[i] * h_x))) / (G * M_t)) - (y[i] / r))
        ecc_z.append((((v_x[i] * h_y) - (v_y[i] * h_x)) / (G * M_t)) - (z[i] / r))
        ecc_abs.append((ecc_x[i] ** 2 + ecc_y[i] ** 2 + ecc_z[i] ** 2) ** 0.5)
        phi.append(np.arctan(ecc_y[i] / ecc_x[i]))
        a.append(((2.0 / r) - ((v ** 2) / (G * M_t))) ** (-1.0))
        e.append((1 - ((h ** 2) / (G * M_t * a[i]))) ** 0.5)

        i = i + 1  # increase counter by one
        if (i % 1000 == 0):
            print(".", end="", flush=True)
    print()
    delta_phi = phi[i] - phi[0] #radians per orbit
    #delta_phi_arcsec[count] = delta_phi[count] * 206265.0
    # print(delta_phi[count], delta_phi_arcsec[count])
    # print(ecc_abs[0], ecc_abs[i])
    #print(alpha)
    #print(beta)

    return x,y,z,v_x,v_y,v_z,t,e,delta_phi,a

print("RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
array1 = integration(p,0,0,0)
x_m = array1[0]
y_m = array1[1]
z_m = array1[2]
v_x_m = array1[3]
v_y_m = array1[4]
v_z_m = array1[5]
t_m = array1[6]
e_2 = array1[7]
delta_phi_2 = array1[8]
print("Numerical precession over one orbital period for an accuracy parameter of ",eta, ":",delta_phi_2," rad/orbit")
print("")
print("RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


#WRITING TO FILES
q = zip_longest(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) #zips together all necessary position variables for both systems, writing to the longest list and filling the rest with NONE values

#opening 'orbits_variable.dat' to write time, position and velocity values for each system by looping over lists in the q zip
with open('orbits_PN0.dat', 'w') as f:
    f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format('t_m','x_m','y_m','z_m','v_x_m','v_y_m','v_z_m','e_2'))
    for (t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) in q:
        f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2))

array1 = integration(p,1,1,0)
x_m = array1[0]
y_m = array1[1]
z_m = array1[2]
v_x_m = array1[3]
v_y_m = array1[4]
v_z_m = array1[5]
t_m = array1[6]
e_2 = array1[7]
delta_phi_2 = array1[8]
print("Numerical precession over one orbital period for an accuracy parameter of ",eta, ":",delta_phi_2," rad/orbit")
print("")
print("RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

#WRITING TO FILES
q = zip_longest(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) #zips together all necessary position variables for both systems, writing to the longest list and filling the rest with NONE values

#opening 'orbits_variable.dat' to write time, position and velocity values for each system by looping over lists in the q zip
with open('orbits_PN12.dat', 'w') as f:
    f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format('t_m','x_m','y_m','z_m','v_x_m','v_y_m','v_z_m','e_2'))
    for (t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) in q:
        f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2))




#7) Computing time necessary for 360 degree precession
p_360 = 360.0/(abs(delta_phi_2)*180.0/np.pi)
print(p_360)

array1 = integration(p_360,0,0,0)
x_m = array1[0]
y_m = array1[1]
z_m = array1[2]
v_x_m = array1[3]
v_y_m = array1[4]
v_z_m = array1[5]
t_m = array1[6]
e_2 = array1[7]
delta_phi_2 = array1[8]
print("Numerical precession over one orbital period for an accuracy parameter of ",eta, ":",delta_phi_2," rad/orbit")
print("")
print("RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


#WRITING TO FILES
q = zip_longest(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) #zips together all necessary position variables for both systems, writing to the longest list and filling the rest with NONE values

#opening 'orbits_variable.dat' to write time, position and velocity values for each system by looping over lists in the q zip
with open('orbits_PN0_360.dat', 'w') as f:
    f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format('t_m','x_m','y_m','z_m','v_x_m','v_y_m','v_z_m','e_2'))
    for (t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) in q:
        f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2))

array1 = integration(p_360,1,1,0)
x_m = array1[0]
y_m = array1[1]
z_m = array1[2]
v_x_m = array1[3]
v_y_m = array1[4]
v_z_m = array1[5]
t_m = array1[6]
e_2 = array1[7]
delta_phi_2 = array1[8]
print("Numerical precession over one orbital period for an accuracy parameter of ",eta, ":",delta_phi_2," rad/orbit")
print("")
print("RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

#WRITING TO FILES
q = zip_longest(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) #zips together all necessary position variables for both systems, writing to the longest list and filling the rest with NONE values

#opening 'orbits_variable.dat' to write time, position and velocity values for each system by looping over lists in the q zip
with open('orbits_PN12_360.dat', 'w') as f:
    f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format('t_m','x_m','y_m','z_m','v_x_m','v_y_m','v_z_m','e_2'))
    for (t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2) in q:
        f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(t_m,x_m,y_m,z_m,v_x_m,v_y_m,v_z_m,e_2))


#8)
array1 = integration(500,0,0,1)
x_m = array1[0]
y_m = array1[1]
z_m = array1[2]
v_x_m = array1[3]
v_y_m = array1[4]
v_z_m = array1[5]
t_m = array1[6]
e_2 = array1[7]
delta_phi_2 = array1[8]
a_2 = array1[9]
print("Numerical precession over one orbital period for an accuracy parameter of ",eta, ":",delta_phi_2," rad/orbit")
print("")
print("RSS:", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

#WRITING TO FILES
q = zip_longest(t_m,e_2,a_2) #zips together all necessary position variables for both systems, writing to the longest list and filling the rest with NONE values

#opening 'orbits_variable.dat' to write time, position and velocity values for each system by looping over lists in the q zip
with open('binary.ecc.orb.dat', 'w') as f:
    f.write('{0} {1} {2}\n'.format('t','e','a'))
    for (t_m,e_2,a_2) in q:
        f.write('{0} {1} {2}\n'.format(t_m,e_2,a_2))
#have fixed the in-loop eccentricities, but still need to have a look at the GR values out of the loop, must be a dimensionality thing.

