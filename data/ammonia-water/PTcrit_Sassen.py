"""
Data from:

Sassen et al., Vapor-Liquid Equilibria for the System Ammonia + Water up to the Critical Region, J. Chem. Eng. Data, 1990

"""

x_NH3 = [0.189,0.346,0.491,0.613,0.699,0.800,1.000]
T_K = [613.5,580.1,551.1,527.3,498.1,470.0,405.4]
p_MPa = [21.52,20.83,20.13,19.49,18.13,16.57,11.34]

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.plot(x_NH3, T_K, 'o-')
    plt.show()
    plt.plot(x_NH3, p_MPa, 'o-')
    plt.show()