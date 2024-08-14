import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['text.usetex'] = True

def reOrgCooperation(x):
    [d0,d1] = [np.shape(x)[0],np.shape(x)[1]]
    xx = np.zeros((d1,d0))
    for i in range(d0):
        for j in range(d1-i-1):
            xx[d1-j-1][i] = x[i][j]
            pass
        pass
    x = xx
    mask = np.triu(np.ones_like(x),k=1)
    x = np.ma.array(x, mask=mask)
    return x

def reOrgEfficacy(x,y):
    [d0,d1] = [np.shape(x)[0],np.shape(x)[1]]
    xx = np.zeros((d1,d0))
    for i in range(d0):
        for j in range(d1-i-1):
            if y[i][j]!=0:
                sam = x[i][j]/y[i][j]
                if sam>1 or sam<=0: 
                    sam = 1
                xx[d1-j-1][i] = np.log10(sam)
    mask = np.triu(np.ones_like(xx),k=1)
    xx = np.ma.array(xx, mask=mask)
    return [xx, np.round(np.max(xx)+0.0,2), np.round(np.min(xx)+0.02,2)]

def reOrgAbundant(x,y):
    [d0,d1] = [np.shape(x)[0],np.shape(x)[1]]
    xx = np.zeros((d1,d0))
    yy = np.zeros((d1,d0))
    vmin,vmax = 100,0.0
    for i in range(d0):
        for j in range(d1-i-1):
            if x[i][j]==2:
                x[i][j]=0;
                pass
            if x[i][j]==4:
                x[i][j]=0;
                pass
            if x[i][j]==6:
                x[i][j]=0;
                pass
            if x[i][j]==11:
                x[i][j]=1;
                pass
            if x[i][j]==12:
                x[i][j]=1;
                pass
            if x[i][j]==14:
                x[i][j]=2;
                pass
            if x[i][j]==8:
                x[i][j]=2;
            xx[d1-j-1][i] = x[i][j]
            yy[d1-j-1][i] = y[i][j]
            if y[i][j]>vmax:
                vmax = y[i][j]
                pass
            if y[i][j]<vmin:
                vmin = y[i][j]
                pass
    l = np.unique(xx)
    n = len(l)
    mask = np.triu(np.ones_like(xx),k=1)
    xx = np.ma.array(xx, mask=mask)

    vmax = np.round(vmax+0.01,2)
    vmin = np.round(vmin-0.01,2)
    mask = np.triu(np.ones_like(yy),k=1)
    yy = np.ma.array(yy, mask=mask)
    return [xx,yy,n,l,vmin,vmax]

cooperation11 = np.loadtxt('cooperation112.txt')
cooperation13 = np.loadtxt('cooperation122.txt')
cooperation12 = np.loadtxt('cooperation132.txt')

cooperation11 = reOrgCooperation(cooperation11)
cooperation12 = reOrgCooperation(cooperation12)
cooperation13 = reOrgCooperation(cooperation13)

mInformation11 = np.loadtxt('mInformationN112.txt')
mInformation13 = np.loadtxt('mInformationN122.txt')
mInformation12 = np.loadtxt('mInformationN132.txt')

capacityN11 = np.loadtxt('capacityN112.txt')
capacityN13 = np.loadtxt('capacityN122.txt')
capacityN12 = np.loadtxt('capacityN132.txt')

efficacy11 = reOrgEfficacy(mInformation11,capacityN11)
efficacy12 = reOrgEfficacy(mInformation12,capacityN12)
efficacy13 = reOrgEfficacy(mInformation13,capacityN13)

abundant11 = np.loadtxt('abundant112.txt')
abundant13 = np.loadtxt('abundant122.txt')
abundant12 = np.loadtxt('abundant132.txt')


percent11 = np.loadtxt('percent112.txt')
percent13 = np.loadtxt('percent122.txt')
percent12 = np.loadtxt('percent132.txt')

abundant11 = reOrgAbundant(abundant11, percent11)
abundant12 = reOrgAbundant(abundant12, percent12)
abundant13 = reOrgAbundant(abundant13, percent13)

# Create custom colormap with two colors
def custom_cmap(row):
    if row == 1:
        colors = [np.array([ 203, 67, 53 ])/255,np.array([ 231, 76, 60 ])/255,(1, 1, 1),np.array([ 93, 173, 226 ])/255,np.array([  40, 116, 166 ])/255]  # Red and blue
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=30)
    if row == 2:
        colors = [np.array([ 255, 143, 0])/255,[1,1,1],np.array([ 174, 213, 129 ])/255,np.array([ 156, 204, 101])/255,np.array([ 139, 195, 74 ])/255,np.array([124, 179, 66])/255,np.array([104,159,56])/255]  # Red and blue
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=30)
    if row == 3:
        colors = [np.array([ 96, 125, 139 ])/255,np.array([255,255,255])/255,np.array([244, 143, 177])/255,np.array([244, 143, 177])/255]  # Red and blue
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=30)
    return cmap

# Generate random data for the plots
data = np.random.rand(3, 3)
data2 = np.random.rand(3, 3)
data3 = np.random.rand(3, 3)

# Create a figure and 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(15, 15))

lw = 1.5

space = 0.1
plt.subplots_adjust(wspace=5*space, hspace=space)

# Plot each data with custom color map and individual color bars
count = 0
for ax, d in zip(axs.flat, [cooperation11,cooperation12,cooperation13,efficacy11,efficacy12,efficacy13,abundant11,abundant12,abundant13]):
    if count in [0,1,2]:
        [data,vmax,vmin] = [d,1,0]
        colorlabel = str(r'$\hat{\gamma}$')
        row = 1
        dis = 0
        dataCol = data
        cbarticks = [0,0.5,1.0]

    if count in [3,4,5]:
        [data,vmax,vmin] = d
        colorlabel = str(r'$\hat{E}$')
        row = 2
        dis = 0
        dataCol = data
        cbarticks = np.round(np.linspace(vmin,vmax,4),2)

    if count in [6,7,8]:
        [data,dataCol,n,l,vmin,vmax] = d
        colorlabel = str(r'$w_{{\bf p}_{\rm m}}$')
        row = 3
        dis = 0.005
        cbarticks = np.round(np.linspace(vmin,vmax,4),2)
    count+=1

    im = ax.imshow(dataCol, cmap=custom_cmap(row), interpolation='bilinear', extent=[0,1.0,0,1.0],vmax=vmax, vmin=vmin)
    levels = 0
    # if row == 1:
    #     levels = 5
    #     contours = ax.contour(data,extent=[0,1.0,1.0,0.0],colors='black',alpha=0.5, levels=levels)
    #     pass
    # if row == 2:
    #     levels = 10
    #     contours = ax.contour(data,extent=[0,1.0,1.0,0.0],colors='black',alpha=0.5, levels=levels)
    #     pass
    if row == 3:
        from matplotlib.patches import Patch
        contours = ax.contourf(data,extent=[0,1.0,1.0,0.0], colors='none', hatches=['..', '--', '\\\\'],levels=5)
        contours = ax.contour(data,extent=[0,1.0,1.0,0.0],colors='black',alpha=1.0, levels=3)
        # legend_patches = [Patch(facecolor='none', edgecolor='black', hatch='..', label='0'),
        #                 Patch(facecolor='none', edgecolor='black', hatch='\\\\\\', label='11'),
        #           Patch(facecolor='none', edgecolor='black', hatch='----', label='14')
        #           ]
        # ax.legend(handles=legend_patches, loc='upper center', ncol=1, bbox_to_anchor=(0.7, 1.15),fontsize=16,title_fontsize='18',title=r'dec$\left({\bf p}_m\right)$',frameon=False)
        pass
    
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)
    cbar_box = cbar.ax.get_position()
    cbar_x = cbar_box.x0 + cbar_box.width / 2
    cbar_y = cbar_box.y1
    ax.annotate(colorlabel, xy=(cbar_x+dis, cbar_y+dis), xytext=(0, 5),textcoords="offset points",ha='center',va='bottom',xycoords='figure fraction', size = 28)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([0, 0.5, 1.0])  # Specify preferred x-tick positions
    ax.set_yticks([0, 0.5, 1.0])  # Specify preferred y-tick positions

    if count in [1,4,7]:
        ax.set_ylabel(r'$n_{2}$',size=30)

    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(cbarticks)
    if row==3:
        
        # cbar.set_ticklabels(cbarlabels)
        ax.set_xlabel(r'$n_{1}$',size=30)

    ax.set_xlabel(r'$n_{1}$',size=30)
    ax.set_ylabel(r'$n_{2}$',size=30)

    x = [0,1]
    y = [1,0]
    ax.plot(x,y,color='k',lw=lw,linestyle='-')
    x = [0,0.5]
    y = [0,0.5]
    ax.plot(x,y,color='k',lw=lw,linestyle='--')
    ax.spines['bottom'].set_linewidth(lw)
    ax.spines['left'].set_linewidth(lw)
    for pos in ['top', 'right']:
        ax.spines[pos].set_visible(False)

legend_patches = [Patch(facecolor='none', edgecolor='black', hatch='...', label=r'$\left({\rm ALLD};{\rm ALLD}\right)$'),
                  Patch(facecolor='none', edgecolor='black', hatch='\\\\', label=r'$\left({\rm TFT};{\rm ALLC}\right)$'),
                  Patch(facecolor='none', edgecolor='black', hatch='---', label=r'$\left({\rm ALLC};{\rm TFT}\right)$')]
# Adjust spacing between subplots

fig.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=20, frameon=False, bbox_to_anchor=(0.5, 0.04))

plt.savefig("channel0.pdf")

import fitz  # PyMuPDF

def autocrop_pdf(input_pdf_path, output_pdf_path):
    # Open the original PDF
    pdf_document = fitz.open(input_pdf_path)
    num_pages = pdf_document.page_count

    for page_number in range(num_pages):
        page = pdf_document.load_page(page_number)
        rect = page.rect

        # Get all text rectangles on the page
        text_rects = page.get_text("dict")["blocks"]
        if text_rects:
            crop_rect = None
            for block in text_rects:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            bbox = fitz.Rect(span["bbox"])
                            if crop_rect is None:
                                crop_rect = bbox
                            else:
                                crop_rect |= bbox
            if crop_rect:
                crop_rect.intersect(rect)
                page.set_cropbox(crop_rect)

    pdf_document.save(output_pdf_path)
    pdf_document.close()

# Example usage
input_pdf_path = "channel0.pdf"
output_pdf_path = "channel.pdf"
autocrop_pdf(input_pdf_path, output_pdf_path)

import os
os.remove("channel0.pdf")

