import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

def funcLower(b1, e):
    b2 = 1.2
    z = (1/2)*(-((((-1+b1)**2 + 4*b2*(-b1+b2)*e-2*(-1+3*b1-4*b2)*(b1-b2)*e**2+8*(b1-b2)**2*e**3-3*(b1-b2)**2*e**4)/((1-2*e)**2*(b2-b1*(-1+e)*e+b2*(-1+e)*e)**2)))**(0.5)+(-1+b1*(-1+e)**2-b2*(2+(-2+e)*e))/((-1+2*e)*(b2-b1*(-1+e)*e+b2*(-1+ e)*e)))
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
ax1.set_ylabel(r'$n_2$', size=35)
ax1.set_xlabel(r'$b_1$', size=35)
ax1.set_ylim(0.0, 0.9)
ax1.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax1.set_xticks([1.2, 1.4, 1.6, 1.8, 2.0, 2.2])
ax1.tick_params(axis='both', which='major', labelsize=30)
ax1.plot(b1, yL, color='#f4d03f', lw=4, ls='--')
ax1.plot(b1, yU, color='#58d68d', lw=4, ls='--')
ax1.set_aspect(1/0.9)  # Set aspect ratio for the first subplot

# Code for subplot 2
eP = np.linspace(-4, -1, 100)
e = 10 ** (eP)
yL = funcLower(2.0, e)
yU = funcUpper(2.0, e)

x2 = np.loadtxt("cooperation_eps_n2.txt")
[d0, d1] = [np.shape(x2)[0], np.shape(x2)[1]]

xx2 = np.zeros((d1, d0))

for i in range(d0):
    for j in range(d1):
        xx2[d1 - j - 1][i] = x2[i][j]
    pass
x2 = xx2

im2 = ax2.imshow(x2, cmap=custom_cmap(), alpha=1, interpolation='bilinear', extent=[-4, -1, 0, 0.9], vmax=1, vmin=0)
ax2.set_ylabel(r'$n_2$', size=35)
ax2.set_xlabel(r'${\rm log}_{10}\epsilon$', size=35)
ax2.set_ylim(0.0, 0.9)
ax2.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
ax2.set_xticks(np.linspace(-4, -1, 4))
ax2.tick_params(axis='both', which='major', labelsize=30)
ax2.plot(eP, yL, color='#f4d03f', lw=4, ls='--')
ax2.plot(eP, yU, color='#58d68d', lw=4, ls='--')
ax2.set_aspect(3/0.9)  # Set aspect ratio for the second subplot

# Add labels (a) and (b) inside the subplots
ax1.text(0.02, 0.98, "(a)", transform=ax1.transAxes, fontsize=35, va='top')
ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, fontsize=35, va='top')

plt.tight_layout()
plt.subplots_adjust(right=0.85)

# Use the color bar from the second subplot
cbar_ax = fig.add_axes([0.85, 0.2, 0.015, 0.7])  # Adjust the width of the color bar here
cbar = plt.colorbar(im2, cax=cbar_ax)
cbar.ax.tick_params(labelsize=25)
cbar.ax.yaxis.label.set_size(26)

plt.annotate(r'$\hat{\gamma}$', xy=(0.854, 0.9), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
             xycoords='figure fraction', size=35)

# plt.savefig('stability_combined.pdf')
b1 = 2.0
e = 10**(-3)
print(funcLower(b1, e),funcUpper(b1, e))
# plt.show()
