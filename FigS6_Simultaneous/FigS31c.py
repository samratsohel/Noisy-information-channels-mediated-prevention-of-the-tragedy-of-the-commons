import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['text.usetex'] = True

fact = 41/101

############################### functions ##########
def FunC1(z,data,minn,maxx):
    d0, d1 = np.shape(z)[0], np.shape(z)[1]
    zz = np.zeros((d1, d0))
    for i in range(d0):
        for j in range(d1):
            zz[d1 - j - 1][i] = z[i][j]
    data.append(zz)
    minn.append(0.0)
    maxx.append(1.0)
    return zz

def FunC2(x,y,data,minn,maxx):
    vmin, vmax = 100, 0.0
    d0, d1 = np.shape(x)[0], np.shape(x)[1]
    xy = np.zeros((d1, d0))
    for i in range(d0):
        for j in range(d1):
            if y[i][j] != 0:
                sam = x[i][j] / y[i][j]
                if sam > 1 or sam <= 0:
                    sam = 1
                xy[d1 - j - 1][i] = np.log10(sam)
            if np.log10(sam) > vmax:
                vmax = np.log10(sam)
            if np.log10(sam) < vmin:
                vmin = np.log10(sam)
    data.append(xy)
    minn.append(vmin)
    maxx.append(vmax)

def custom_cmap(n):
    if n in [0,1,2]:
        colors = [np.array([ 203, 67, 53 ])/255,np.array([ 231, 76, 60 ])/255,(1, 1, 1),np.array([ 93, 173, 226 ])/255,np.array([  40, 116, 166 ])/255]  # Red and blue
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=30)
    if n in [3,4,5]:
        colors = [np.array([ 255, 143, 0])/255,[1,1,1],[1,1,1],np.array([ 174, 213, 129 ])/255,np.array([104,159,56])/255]  # Red and blue
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=40)
    return cmap
############################### functions ##########


############################### data ###############
z1 = np.loadtxt("cooperation112.txt")
z3 = np.loadtxt("cooperation122.txt")
z2 = np.loadtxt("cooperation132.txt")

x1 = np.loadtxt("mInformationN112.txt")
x3 = np.loadtxt("mInformationN122.txt")
x2 = np.loadtxt("mInformationN132.txt")

y1 = np.loadtxt('capacityN112.txt')
y3 = np.loadtxt('capacityN122.txt')
y2 = np.loadtxt('capacityN132.txt')
############################### data ###############

data = []
minn = []
maxx = []

ttls = [r'${\bf q}=(10001000)$',r'${\bf q}=(10001111)$',r'${\bf q}=(10001110)$']
FunC1(z1,data,minn,maxx)
FunC1(z2,data,minn,maxx)
FunC1(z3,data,minn,maxx)
FunC2(x1,y1,data,minn,maxx)
FunC2(x2,y2,data,minn,maxx)
FunC2(x3,y3,data,minn,maxx)


# Create a figure and subplots
fig, axs = plt.subplots(2, 3, figsize=(8.2, 4.7))

# Loop through subplots and plot data
for i, ax in enumerate(axs.flat):
    im = ax.imshow(data[i], cmap=custom_cmap(i), alpha=1, interpolation='bilinear', extent=[0.0, 0.2, 0, 0.5], vmax=maxx[i], vmin=minn[i])
    ax.set_xticks([0.0, 0.1, 0.2])
    ax.set_xticklabels([r'$0.0$', r'$0.5$', r'$1.0$'])
    ax.set_xlabel(r'$\delta$', size=12)
    
    colorlabel = str(r'$\hat{E}$')
    if i in [0, 3]:
        ax.set_ylabel(r'$n$', size=18)
    if i in [0, 1, 2]:
        # ax.set_title(ttls[i])
        colorlabel = str(r'$\hat{\gamma}$')
    ax.set_aspect(fact)
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)

    cbar_x, cbar_y = 1.09, 0.8  # Place annotation at the center-top of the colorbar
    ax.annotate(colorlabel, xy=(cbar_x, cbar_y), xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', size=15, xycoords='axes fraction')

plt.subplots_adjust(wspace=0.1, hspace=0.1)
# Adjust layout
plt.tight_layout()
plt.savefig('discount_simultaneous.pdf')
