import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
plt.rcParams['text.usetex'] = True

t = np.loadtxt('t.txt')

alphaa = 1 # if you want to plot the porbability of beneficial state put 1, else it wil plot trajecories

if alphaa == 1:
    wo112 = np.loadtxt('withOutNoise112a1.txt')
    wo114 = np.loadtxt('withOutNoise114a1.txt')
    wn112 = np.loadtxt('withNoise112a1.txt')
    wn114 = np.loadtxt('withNoise114a1.txt')


    wo132 = np.loadtxt('withOutNoise122a1.txt')
    wo134 = np.loadtxt('withOutNoise124a1.txt')
    wn132 = np.loadtxt('withNoise122a1.txt')
    wn134 = np.loadtxt('withNoise124a1.txt')

    wo122 = np.loadtxt('withOutNoise132a1.txt')
    wo124 = np.loadtxt('withOutNoise134a1.txt')
    wn122 = np.loadtxt('withNoise132a1.txt')
    wn124 = np.loadtxt('withNoise134a1.txt')
    pass
if alphaa !=1:
    wo112 = np.loadtxt('withOutNoise112.txt')
    wo114 = np.loadtxt('withOutNoise114.txt')
    wn112 = np.loadtxt('withNoise112.txt')
    wn114 = np.loadtxt('withNoise114.txt')


    wo132 = np.loadtxt('withOutNoise122.txt')
    wo134 = np.loadtxt('withOutNoise124.txt')
    wn132 = np.loadtxt('withNoise122.txt')
    wn134 = np.loadtxt('withNoise124.txt')

    wo122 = np.loadtxt('withOutNoise132.txt')
    wo124 = np.loadtxt('withOutNoise134.txt')
    wn122 = np.loadtxt('withNoise132.txt')
    wn124 = np.loadtxt('withNoise134.txt')
    pass


# Create a figure and three subplots
size = 3
fig, axs = plt.subplots(1, 3, figsize=(3*size-3, size))

# Plot data on each subplot
axs[0].plot(t, wo112,color='gray',linestyle='--')
axs[0].plot(t, wo114,color='gray',linestyle='-')
axs[0].plot(t, wn112,color='red',linestyle='--')
axs[0].plot(t, wn114,color='red',linestyle='-')
axs[0].set_ylabel(r'probability of $s_1$ $\langle \alpha_{\rm in}(t) \rangle$',size=14)

if alphaa == 0:
    axs[0].set_ylabel(r'cooperation rate $\langle \gamma(t) \rangle$',size=14)

axs[1].plot(t, wo122,color='gray',linestyle='--')
axs[1].plot(t, wo124,color='gray',linestyle='-')
axs[1].plot(t, wn122,color='red',linestyle='--')
axs[1].plot(t, wn124,color='red',linestyle='-')


axs[2].plot(t, wo132,color='gray',linestyle='--')
axs[2].plot(t, wo134,color='gray',linestyle='-')
axs[2].plot(t, wn132,color='red',linestyle='--')
axs[2].plot(t, wn134,color='red',linestyle='-')


ar = 7000
dis = 0.021
for i, ax in enumerate(axs):
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_ylim(-0.04,1.14)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xticks([0,2500,5000])
    ax.set_xlabel(r'time $(t)$',size=14)
    aa = 0 
    if alphaa ==1:
        aa = 3
    ax.text(dis, 1-dis, f'({chr(97 + i+aa)})', transform=ax.transAxes, va='top', ha='left',size=15)  # Add (a), (b), (c) text
    ax.set_aspect(ar)


# Show plot
plt.tight_layout()

if alphaa == 1:
    plt.savefig('trajectoriesa1.pdf')

if alphaa != 1:
    plt.savefig('trajectories.pdf')