# This program computes the relative orbit for a 2 body system, using the leap-frog algorithm with a variable timestep

# import
import os
import numpy as np


# information to the user
print('This program (6479942_ecc.py) implements a leap-frog algorithm with a variable timestep to compute the '
      'relative orbit of an eccentric black hole binary system.')
print('')
print('The initial conditions are first output to the file "binary.ecc.init.dat"')
print('')
print('The coordinates and velocities, as well as the times, eccentricities and semi-major axis values'
      'are output to the files: "orbits_pn0.dat","orbits_pn12.dat","orbits_pn0_360.dat","orbits_pn12_360.dat" and '
      '"binary.ecc.orb.dat"')
print('')

# unit consideration
g = 1.00  # gravitational constant
r = 1.00  # mpc
t = 1.00  # yr
mpc = 3.086e+13  # mpc in metres
yr = 3600.0 * 365.0 * 24.0  # yr in seconds
g_si = 6.67e-11  # g in si units of m^3kg^-1s^-2 for use in the physical mass unit calculation
m_0 = 1.98847e+30  # solar mass in kg
m = (r ** 3) / (g * (t ** 2))  # mass in code units derived from kepler's third law
m_si = (4 * (np.pi ** 2) * (mpc ** 3)) / (
    (g_si * (yr ** 2)))  # code mass unit in si for conversion to solar masses for initial cons
m_solar = m_si / m_0  # the code mass unit in terms of solar masses (solar mass per code unit)
v = r / t  # velocity units in mpc/yr
# speed of light consideration:
c_si = 299792458  # speed of light in SI units m/s
c = c_si * yr / mpc  # speed of light in code units mpc/yr
# unit output for the user
print('First, the internal units are based upon G=1, [R]=1mpc, [T]=1yr, with all other units derived from these.')
print('mass units:', m, ' mpc**(3)*yr**(-2)*G**(-1), which corresponds to:', m_solar, ' solar masses')
print('velocity units:', v, ' milliparsecs per year')
print('speed of light:', c, ' milliparsecs per year')
print('')

# declare parameters and variables
m_1 = 1E+7 / m_solar  # Black hole 1 mass
m_2 = 1E+4 / m_solar  # Black hole 2 mass
m_t = m_1 + m_2  # total mass of system
nu = (m_1 * m_2) / (m_t ** 2)
a_binary = 0.1  # binary system semi-major axis
e_binary = 0.9  # binary system eccentricity
print('In this case, the masses of the black holes are:', m_1 * m_solar, ' and', m_2 * m_solar,
      'solar masses, the semi-major axis of the system is', a_binary, 'and the eccentricity is', e_binary)

print('')
eta = 1e-4  # accuracy parameter
print('Throughout the course of this program, an accuracy parameter for the variable timestep integration of', eta,
      'is used.')
print('')
p = ((4 * (np.pi ** 2) * (a_binary ** 3)) / (g * m_t)) ** 0.5  # orbital period of binary
print('The orbital period for the black hole binary is:', p, 'yrs')
print('')

# computation of pericentre and apocentre for the system
r_p = a_binary * (1.0 - e_binary)
r_a = a_binary * (1.0 + e_binary)

# computation of pericentre and apocentre velocities
v_p = ((g * m_t / a_binary) * ((1.0 + e_binary) / (1.0 - e_binary))) ** 0.5
v_a = ((g * m_t / a_binary) * ((1.0 - e_binary) / (1.0 + e_binary))) ** 0.5

# initial conditions output
x = r_a  # x-coordinate of relative orbit
y = 0.0  # y-coordinate of relative orbit
z = 0.0  # z-coordinate of relative orbit
v_x = 0.0  # x-velocity of relative orbit
v_y = v_a  # y-velocity of relative orbit
v_z = 0.0  # z-velocity of relative orbit

f = open("binary.ecc.init.dat", "w")
f.write("{} {} {}\n".format('X position:', float(x), "mpc"))
f.write("{} {} {}\n".format('y position:', float(y), "mpc"))
f.write("{} {} {}\n".format('Z position:', float(z), "mpc"))
f.write("{} {} {}\n".format('X velocity:', float(v_x), "mpc/yr"))
f.write("{} {} {}\n".format('Y velocity:', float(v_y), "mpc/yr"))
f.write("{} {} {}\n".format('z velocity:', float(v_z), "mpc/yr"))
f.close()

# output of key initial conditions for user
print('The initial conditions of the binary are:')
print('X position:', float(x), "mpc")
print('Y velocity:', float(v_y), "mpc/yr")
print('')

# theoretical advance of pericentre in radians per orbit
delta_phi_gr = (6.0 * np.pi * g * m_1) / ((c ** 2) * a_binary * (1.0 - (e_binary ** 2)))
print('theoretical pericentre advance due to gr:', delta_phi_gr, ' rad/orbit')


# function to return alpha and beta values for post-newtonian acceleration modifications
def alphabeta(r_dot, r, v, pn_1, pn_2, pn_25):
    # calculation of coefficients for PN0,1,2,2.5
    alpha_0 = 1.0
    alpha_1 = (-(3.0 / 2.0) * (r_dot ** 2) * nu) + ((1.0 + (3.0 * nu)) * (v ** 2)) - (
            2.0 * (2.0 + nu) * (g * m_t / r))
    alpha_2 = ((15.0 / 8.0) * (r_dot ** 4) * nu * (1 - (3.0 * nu))) + (
            3 * (r_dot ** 2) * nu * (v ** 2) * ((2 * nu) - (3.0 / 2.0))) + (
                      nu * (v ** 4) * (3.0 - (4 * nu))) + ((g * m_t / r) * (
            (-2.0 * (r_dot ** 2) * (1.0 + (nu ** 2))) - (25.0 * r_dot * nu) - (
            (13.0 / 2.0) * nu * (v ** 2)))) + (
                      ((g ** 2) * (m_t ** 2) / (r ** 2)) * (9.0 + (87.0 * nu / 4.0)))
    alpha_25 = -(8.0 / 5.0) * (g * m_t / r) * nu * r_dot * ((17.0 / 3.0) * (g * m_t / r) + (3.0 * (v ** 2)))
    beta_0 = 0.0
    beta_1 = -2.0 * (2.0 - nu) * r_dot
    beta_2 = (3.0 * (r_dot ** 3) * nu * ((3.0 / 2.0) + nu)) - (
            r_dot * nu * (v ** 2) * ((15.0 / 2.0) + (2.0 * nu))) + (
                     (g * m_t * r_dot / r) * (2.0 + (41.0 * nu / 2.0) + (4.0 * (nu ** 2))))
    beta_25 = (8.0 / 5.0) * (g * m_t / r) * nu * (3.0 * (g * m_t / r) + (v ** 2))

    # calculation of the alpha and beta parameters.  pn_1, pn_2 and pn_25 take values of either 1 or zero, which are
    # taken as function inputs and determine which PN modifications are utilised
    alpha = alpha_0 + pn_1 * ((c ** -2) * alpha_1) + pn_2 * ((c ** -4) * alpha_2) + pn_25 * ((c ** -5) * alpha_25)
    beta = beta_0 + pn_1 * ((c ** -2) * beta_1) + pn_2 * ((c ** -4) * beta_2) + pn_25 * ((c ** -5) * beta_25)

    return alpha, beta


def integration(p, pn_1, pn_2, pn_25, filename):
    # coordinates redefined for use in the function
    x = r_a
    y = 0.0
    z = 0.0
    v_x = 0.0
    v_y = v_a
    v_z = 0.0
    t = 0.0
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5  # the initial position value
    v = (v_x ** 2 + v_y ** 2 + v_z ** 2) ** 0.5  # the initial velocity value
    r_dot = ((x * v_x) + (y * v_y) + (z * v_z)) / r  # time derivative of r
    i = 0  # counter

    # use of alpha/beta function for PN modifications
    alpha, beta = alphabeta(r_dot, r, v, pn_1, pn_2, pn_25)

    # initial acceleration values
    a_x = -((g * m_t) / (r ** 2)) * (
            (alpha * (x / r)) + (beta * v_x))
    a_y = -((g * m_t) / (r ** 2)) * ((alpha * (y / r)) + (beta * v_y))
    a_z = -((g * m_t) / (r ** 2)) * ((alpha * (z / r)) + (beta * v_z))

    # initial angluar momentum per unit mass found from a cross product r x v
    h_x = (y * v_z) - (z * v_y)
    h_y = -((x * v_z) - (z * v_x))
    h_z = (x * v_y) - (y * v_x)
    h = (h_x ** 2 + h_y ** 2 + h_z ** 2) ** 0.5

    # eccentricity vector calculation
    ecc_x = (((v_y * h_z) - (v_z * h_y)) / (g * m_t)) - (x / r)
    ecc_y = ((-((v_x * h_z) - (v_z * h_x))) / (g * m_t)) - (y / r)
    ecc_z = (((v_x * h_y) - (v_y * h_x)) / (g * m_t)) - (z / r)
    ecc_abs = (ecc_x ** 2 + ecc_y ** 2 + ecc_z ** 2) ** 0.5

    # computing the angle phi (ecc_x / ecc_y) for pericentre advance calculations
    phi = np.arctan(ecc_y / ecc_x)
    phi_0 = phi  # storing the initial angle

    # initial semi-major axis and eccentricity
    a = ((2.0 / r) - ((v ** 2) / (g * m_t))) ** (-1.0)
    e = (1 - ((h ** 2) / (g * m_t * a))) ** 0.5

    # while loop containing leap-frog algorithm for mercury-sun system, set to run until the input period is reached
    # outputting data to file within the loop
    with open(filename, 'w') as f:
        # first output initial conditions to file
        f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(t, e, a, x, y, z, v_x, v_y, v_z))
        while t <= p:
            dt = eta * ((r ** 3) / (g * m_t)) ** 0.5  # computation of variable timestep
            t = t + dt  # addition of timestep to current time to find next time

            v_x_temp = v_x + (a_x * (dt / 2.0))  # calculation of temporary velocities
            v_y_temp = v_y + (a_y * (dt / 2.0))
            v_z_temp = v_z + (a_z * (dt / 2.0))

            x = x + (v_x_temp * dt)  # calculation of next position values
            y = y + (v_y_temp * dt)
            z = z + (v_z_temp * dt)
            r = (x ** 2 + y ** 2 + z ** 2) ** 0.5

            # alpha/beta PN modification
            r_dot = ((x * v_x) + (y * v_y) + (z * v_z)) / r
            alpha, beta = alphabeta(r_dot, r, v, pn_1, pn_2, pn_25)

            # new acceleration values for PN modifications
            a_x = -((g * m_t) / (r ** 2)) * (
                    (alpha * (x / r)) + (beta * v_x))
            a_y = -((g * m_t) / (r ** 2)) * ((alpha * (y / r)) + (beta * v_y))
            a_z = -((g * m_t) / (r ** 2)) * ((alpha * (z / r)) + (beta * v_z))

            # computation of next velocity values based on new acceleration
            v_x = v_x_temp + (a_x * (dt / 2.0))
            v_y = v_y_temp + (a_y * (dt / 2.0))
            v_z = v_z_temp + (a_z * (dt / 2.0))
            v = (v_x ** 2 + v_y ** 2 + v_z ** 2) ** 0.5

            # angular momentum per unit mass
            h_x = (y * v_z) - (z * v_y)
            h_y = -((x * v_z) - (z * v_x))
            h_z = (x * v_y) - (y * v_x)
            h = (h_x ** 2 + h_y ** 2 + h_z ** 2) ** 0.5

            # eccentricity (LRL) vector
            ecc_x = (((v_y * h_z) - (v_z * h_y)) / (g * m_t)) - (x / r)
            ecc_y = ((-((v_x * h_z) - (v_z * h_x))) / (g * m_t)) - (y / r)
            ecc_z = (((v_x * h_y) - (v_y * h_x)) / (g * m_t)) - (z / r)
            ecc_abs = (ecc_x ** 2 + ecc_y ** 2 + ecc_z ** 2) ** 0.5

            # computing the angle phi
            phi = np.arctan(ecc_y / ecc_x)

            # semi-major axis and eccentricity
            a = ((2.0 / r) - ((v ** 2) / (g * m_t))) ** (-1.0)
            e = (1 - ((h ** 2) / (g * m_t * a))) ** 0.5

            # writing to file
            f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8}\n'.format(t, e, a, x, y, z, v_x, v_y, v_z))

            i = i + 1  # adding to counter for testing

        delta_phi = phi - phi_0  # radians per orbit

        return delta_phi, i


# calling the integration function for one orbital period and without PN corrections
delta_phi_1, i_1 = integration(p, 0, 0, 0, 'orbits_pn0.dat')
print(
    'The integrating function (PN0 for one orbital period) has successfully written data to the file '
    '"orbits_pn0.dat" and completed',
    i_1, 'steps to do so.')
print("Newtonian numerical precession over one orbital period:", delta_phi_1, " rad/orbit")
print("")

# calling the integration function for one orbital period and with PN1 and PN2 corrections
delta_phi_2, i_2 = integration(p, 1, 1, 0, 'orbits_pn12.dat')
print(
    'The integrating function (PN1 and PN2 for one orbital period) has successfully written data to the file '
    '"orbits_pn12.dat" and completed',
    i_2, 'steps to do so.')
print("PN numerical precession over one orbital period:", delta_phi_2, " rad/orbit")
print("")

# computing time necessary for 360 degree precession
p_360 = (360.0 * p) / (abs(delta_phi_2) * 180.0 / np.pi)
print('The time necessary for a 360 degree precession is:', p_360, 'years')
print('')

# calling the integration function for a period corresponding to 360 degree precession and without PN corrections
delta_phi_3, i_3 = integration(p_360, 0, 0, 0, 'orbits_pn0_360.dat')
print(
    'The integrating function (PN0 for a period corresponding to 360 degree precession) has successfully written data '
    'to the file "orbits_pn0_360.dat" and completed', i_3, 'steps to do so.')
print("Newtonian numerical precession over a period corresponding to 360 degree precession:", delta_phi_3, " rad")
print("")

# calling the integration function for a period corresponding to 360 degree precession and with PN1 and PN2 corrections
delta_phi_4, i_4 = integration(p_360, 1, 1, 0, 'orbits_pn12_360.dat')
print(
    'The integrating function (PN1 and PN2 for a period corresponding to 360 degree precession) has successfully '
    'written data to the file "orbits_pn12_360.dat" and completed', i_4, 'steps to do so.')
print("PN numerical precession over a period corresponding to 360 degree precession:", delta_phi_4, " rad")
print("")

# calling the integration function for a period of 500 years and with the PN2.5 correction only
delta_phi_5, i_5 = integration(500, 0, 0, 1, 'binary.ecc.orb.dat')
print(
    'The integrating function (PN2.5 for a period of 500 years) has successfully written data to the file '
    '"binary.ecc.orb.dat" and completed',
    i_5, 'steps to do so.')
print("PN2.5 numerical precession over a period of 500 years:", delta_phi_5, " rad")
print("")

# comparing the theoretical and numerical values for pericentre advance strictly due to GR
gr_advance = delta_phi_2 - delta_phi_1  # advance due to consideration of PN effects (subtracting purely numerical)
delta_phi_err = abs(
    (gr_advance - delta_phi_gr) / delta_phi_gr) * 100.0  # relative error with regard to the theoretical value
print('The advance of the pericentre strictly due to GR effects is calculated as:', gr_advance,
      'with a relative error with respect to the theoretical value of:', delta_phi_err)
