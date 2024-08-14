import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
plt.rcParams['text.usetex'] = True

t = np.linspace(0,0.5,101) 

coop112  = np.loadtxt('cooperation112.txt')
alpha112 = np.loadtxt('alphaIn112.txt')
coop114  = np.loadtxt('cooperation114.txt')
alpha114 = np.loadtxt('alphaIn114.txt')

coop132  = np.loadtxt('cooperation122.txt')
alpha132 = np.loadtxt('alphaIn122.txt')
coop134  = np.loadtxt('cooperation124.txt')
alpha134 = np.loadtxt('alphaIn124.txt')

coop122  = np.loadtxt('cooperation132.txt')
alpha122 = np.loadtxt('alphaIn132.txt')
coop124  = np.loadtxt('cooperation134.txt')
alpha124 = np.loadtxt('alphaIn134.txt')

coop112 = coop112-np.ones(len(t))*coop112[0]
coop114 = coop114-np.ones(len(t))*coop114[0]
coop122 = coop122-np.ones(len(t))*coop122[0]
coop124 = coop124-np.ones(len(t))*coop124[0]
coop132 = coop132-np.ones(len(t))*coop132[0]
coop134 = coop134-np.ones(len(t))*coop134[0]

alpha112 = alpha112-np.ones(len(t))*alpha112[0]
alpha114 = alpha114-np.ones(len(t))*alpha114[0]
alpha122 = alpha122-np.ones(len(t))*alpha122[0]
alpha124 = alpha124-np.ones(len(t))*alpha124[0]
alpha132 = alpha132-np.ones(len(t))*alpha132[0]
alpha134 = alpha134-np.ones(len(t))*alpha134[0]

# Create a figure and three subplots
size = 3
fig, axs = plt.subplots(1, 3, figsize=(3*size-3, size))

# Plot data on each subplot
axs[0].plot(t, coop112,color='blue',linestyle='--')
axs[0].plot(t, coop114,color='blue',linestyle='-')
axs[0].plot(t, alpha112,color='green',linestyle='--')
axs[0].plot(t, alpha114,color='green',linestyle='-')
# axs[0].set_ylabel(r'cooperation rate $\langle \gamma(t) \rangle$',size=14)

axs[1].plot(t, coop122,color='blue',linestyle='--')
axs[1].plot(t, coop124,color='blue',linestyle='-')
axs[1].plot(t, alpha122,color='green',linestyle='--')
axs[1].plot(t, alpha124,color='green',linestyle='-')


axs[2].plot(t, coop132,color='blue',linestyle='--')
axs[2].plot(t, coop134,color='blue',linestyle='-')
axs[2].plot(t, alpha132,color='green',linestyle='--')
axs[2].plot(t, alpha134,color='green',linestyle='-')


ar = 0.7#7000
dis = 0.021
for i, ax in enumerate(axs):
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_ylim(-0.25,0.85)
    ax.set_yticks([-0.2,0,0.2,0.4,0.6,0.8])
    ax.set_xticks([0,0.25,0.5])
    ax.axhline(0.0,color='k', alpha=0.3)
    ax.set_xlabel(r'noise $(n)$',size=14)
    ax.text(dis, 1-dis, f'({chr(97 + i+6)})', transform=ax.transAxes, va='top', ha='left',size=15)  # Add (a), (b), (c) text
    ax.set_aspect(ar)


# Show plot
plt.tight_layout()
plt.savefig('symmetric.pdf')