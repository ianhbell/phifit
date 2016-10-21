"""
Data from:
A. Sakabe, D. Arai, H. Miyamoto, M. Uematsu, 2008, "Measurements of the critical parameters 
for {xNH 3 + (1-x)H 2 O} with x = (0.9098,0.7757,0.6808)", J. Chem. Thermodyn.

Collected by I. Bell
"""

x_NH3 = [0.9098, 0.7757, 0.6808]
u_x_NH3 = [0.0009, 0.0007, 0.0006]
T_K = [436.56, 478.58, 503.06]
u_T_K = [0.013, 0.013, 0.013]
p_MPa = [14.182, 17.307, 18.693]
u_p_MPa = [0.011, 0.007, 0.007]
rho_kgm3 = [262.9, 284.0, 294.6]
u_rho_kgm3 = [0.9, 0.6, 0.5]

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.plot(x_NH3, T_K, 'o-')
    plt.show()
    plt.plot(x_NH3, p_MPa, 'o-')
    plt.show()