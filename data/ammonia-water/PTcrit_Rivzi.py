"""
Data from:
Rivzi and Heidemann, 1987, J. Chem. Eng. Data, v.32

Collected by I. Bell
"""

x_NH3 = [0.116,0.244 ,0.387 ,0.570 ,0.748 ,0.870 ,0.936,0.972,1.000]
T_K = [618.1 ,610.2,579.7,526.2,483.3,451.5,422.5,411.9,405.5]
p_MPa = [22.37,22.52,22.39,21.42,19.00,16.05,13.85,12.47,11.15]

if __name__=='__main__':
    import matplotlib.pyplot as plt
    plt.plot(x_NH3, T_K, 'o-')
    plt.show()
    plt.plot(x_NH3, p_MPa, 'o-')
    plt.show()