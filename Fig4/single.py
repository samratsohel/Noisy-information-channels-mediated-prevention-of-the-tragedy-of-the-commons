import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

# Create a single figure and axis
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)

def funcLower(b1, e):
    b2 = 1.2
    z = (1/2)*(-((((-1+b1)**2 + 4*b2*(-b1+b2)*e-2*(-1+3*b1-4*b2)*(b1-b2)*e**2+8*(b1-b2)**2*e**3-3*(b1-b2)**2*e**4)/((1-2*e)**2*(b2-b1*(-1+e)*e+b2*(-1+e)*e)**2)))**(0.5)+(-1+b1*(-1+e)**2-b2*(2+(-2+e)*e))/((-1+2*e)*(b2-b1*(-1+e)*e+b2*(-1+e)*e)))
    return z

def funcUpper(b1, e):
    b2 = 1.2
    z = (1/2)*((((-1 + b1)**2+4*b2*(-b1+b2)*e-2*(-1+3*b1-4*b2)*(b1-b2)*e**2+8*(b1-b2)**2*e**3-3*(b1-b2)**2*e**4)/((1-2*e)**2*(b2-b1*(-1+e)*e+b2*(-1+e)*e)**2))**(0.5)+(1+b1*(-1+(5-4*e)*e**2)+b2*e*(4+e*(-5+4*e)))/((-1+2*e)*(b2-b1*(-1+e)*e+b2*(-1+e)*e)))
    return z

b1 = np.linspace(1.2, 2.2, 100)
yL = funcLower(b1, 0.001)
yU = funcUpper(b1, 0.001)

x1 = np.loadtxt("cooperation_b1_n2.txt")
[d0, d1] = [np.shape(x1)[0], np.shape(x1)[1]]

xx1 = np.zeros((d1, d0))

for i in range(d0):
    for j in range(d1):
        xx1[d1 - j - 1][i] = x1[i][j]
    pass
x1 = xx1

def custom_cmap():
    colors = [np.array([203, 67, 53]) / 255, np.array([231, 76, 60]) / 255, (1, 1, 1), np.array([93, 173, 226]) / 255,
              np.array([40, 116, 166]) / 255]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=40)
    return cmap

im1 = ax1.imshow(x1, cmap=custom_cmap(), alpha=1, interpolation='bilinear', extent=[1.2, 2.2, 0, 0.9], vmax=1, vmin=0)
ax1.set_ylabel(r'$n_2$', size=30)
ax1.set_xlabel(r'$b_1$', size=30)
ax1.set_ylim(0.0, 0.9)
ax1.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax1.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
ax1.tick_params(axis='both', which='major', labelsize=25)
ax1.plot(b1, yL, color='#f4d03f', lw=4, ls='--')
ax1.plot(b1, yU, color='#58d68d', lw=4, ls='--')
ax1.set_aspect(1/0.9)  # Set aspect ratio for the axis



plt.tight_layout()

# Adjust the color bar position
cbar = plt.colorbar(im1, ax=ax1, fraction=0.032, pad=0.07)
cbar.ax.tick_params(labelsize=25)
cbar.ax.yaxis.label.set_size(20)

# Adjust the layout to reduce left margin space
plt.subplots_adjust(left=0.1, right=0.8)

# Place the \hat{\gamma} annotation on top of the color bar
plt.annotate(r'$\hat{\gamma}$', xy=(0.789, 0.88), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
             xycoords='figure fraction', size=35)

plt.savefig('stability_single.pdf')
