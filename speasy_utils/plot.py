import numpy as np
import matplotlib.pyplot as plt

from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime, timedelta


from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib as mpl
from speasy_utils.coord import mp_shue1997, bs_Jerab2005

import speasy as spz

def plot_features(x,figsize=(10, 10), title="Features", bins=200, xlabel="x_1", ylabel="x_2"):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    fig.suptitle("Scaled features")
    gs = GridSpec(2, 2, figure=fig)

    axes = fig.add_subplot(gs[0, :])
    axes.hist2d(x[:,0], x[:,1], bins=bins, range=None,
                         cmap="gist_ncar", 
                         density=False)
    #axes.set_title("(r_n,r_b) hist2d")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)

    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1])
    ax1.hist(x[:,0], bins=bins, density=True)
    ax1.set_xlabel(xlabel)
    ax2.hist(x[:,1], bins=bins, density=True)
    ax2.set_xlabel(ylabel)
    return fig

def plot_decision_region(x, y, clf, ax):
    x_min, x_max = x[:,0].min(), x[:,0].max()
    y_min, y_max = x[:,1].min(), x[:,1].max()
    xstep = (x_max - x_min) / 200.
    ystep = (y_max - y_min) / 200.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, xstep),
                        np.arange(y_min,y_max, ystep))
    z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    
    ax.contourf(xx, yy, z, alpha=.4)
    if y is not None:
        yx, yy = y
        ax.scatter(yx[:,0], yx[:,1], c=yy, s=20, edgecolor="k")
    
    if hasattr(clf, "cluster_centers_" ):
        c=0
        for cc in clf.cluster_centers_:
            ax.text(cc[0],cc[1],str(c))
            c+=1
    return ax

def plot_train_test_feature_densities(x_train, x_test, figsize=(10,10)):
    fig = plt.figure(constrained_layout=True, figsize=(10, 10))
    gs = GridSpec(3, 2, figure=fig)

    fig.suptitle("Train and testing features")
    ax1 = fig.add_subplot(gs[0,0])
    d1,_,_,_= ax1.hist2d(x_train[:,0], x_train[:,1], bins=200, range=None,
                         cmap="gist_ncar", 
                         density=True)
    ax1.set_title("(r_n,r_b) hist2d train")
    ax1.set_xlabel("r_n = |b| / |sw_b|")
    ax1.set_ylabel("r_b = |n| / |sw_n|")

    ax2 = fig.add_subplot(gs[0,1])
    d2,_,_,_= ax2.hist2d(x_test[:,0], x_test[:,1], bins=200, range=None,
                         cmap="gist_ncar",
                         density=True)
    ax2.set_title("(r_n,r_b) hist2d test")
    ax2.set_xlabel("log(|b| / |sw_b|)")
    ax2.set_ylabel("log(|n| / |sw_n|)")

    ax3 = fig.add_subplot(gs[1,0])
    ax3.hist(x_train[:,0], bins=300, density=True)
    ax3.set_title("r_n train")

    ax4 = fig.add_subplot(gs[1,1])
    ax4.hist(x_train[:,1], bins=300, density=True)
    ax4.set_title("r_b train")

    ax5 = fig.add_subplot(gs[2,0])
    ax5.hist(x_test[:,0], bins=300, density=True)
    ax5.set_title("r_n test")

    ax6 = fig.add_subplot(gs[2,1])
    ax6.hist(x_test[:,1], bins=300, density=True)
    ax6.set_title("r_b train")
    
    return fig
    
def plot_train_test_label_densities(x_train, x_test, y_train, y_test, figsize=(10,3)):
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    fig.suptitle("Train and test labels")
    gs = GridSpec(1, 2, figure=fig)

    ax3=fig.add_subplot(gs[0,0])
    ax3.hist(y_train, bins=3, density=True)
    ax3.set_title("train labels")

    ax4=fig.add_subplot(gs[0,1], sharey=ax3)
    ax4.hist(y_test, bins=3, density=True)
    ax4.set_title("test labels")
    ax4.set_yticks([])
    
    return fig


def plot_magnetosphere(ax):
    earth_circle = plt.Circle((0,0),1, color="black")
    ax.add_patch(earth_circle)
    theta, phi = np.linspace(0, 2*np.pi, 100), np.zeros(100)
    x_shue,y_shue,z_shue = mp_shue1997(theta, phi)
    x_jerab,y_jerab,z_jerab = bs_Jerab2005(theta, phi)
    ax.plot(x_shue,z_shue,c="black")
    ax.plot(x_jerab,z_jerab,"--",c="black")
    
def plot_clustering_result(x, y, xyz, xmin, xmax, ymin, ymax, figsize=(15,5), cmap="hot", title="clustering_result",labels=None):
    fig, axes = plt.subplots(1,3, figsize=figsize)
    plt.suptitle(title)
    axes[0].scatter(x[:,0], x[:,1], c=y, cmap=cmap, alpha=.1, marker=".")
    axes[0].set_xlabel("x_1")
    axes[0].set_ylabel("x_2")
    axes[1].hist(y,align="mid")
    if labels is not None:
        axes[1].set_xticks(set(y))
        axes[1].set_xticklabels(labels)
    axes[2].scatter(xyz[:,0], xyz[:,1], c=y, alpha=.2, cmap=cmap, marker=".")
    axes[2].set_xlabel("x (GSE)")
    axes[2].set_ylabel("y (GSE)")
    plot_magnetosphere(axes[2])
    #axes[2].plot(x_shue,z_shue,c="black")
    #axes[2].plot(x_jerab,z_jerab,"--",c="black")
    axes[2].set_xlim(xmin,xmax)
    axes[2].set_ylim(ymin,ymax)
    return fig
    
dimmap = {"x":0, "y":1, "z":2}
def plot_scatter_dims(xyz, color, ax, plane="x,y", coord="GSE", xlim=None,
                      ylim=None,
                     title=None, cmap=None):
    #dims = [dimmap[k] for k in plane.split(",")]
    #dim_lab = plane.split(",")
    plot_magnetosphere(ax=ax)
    ax.scatter(xyz[:,0], xyz[:,1], c=color, 
            cmap=cmap,alpha=.2,marker="+")
    ax.set_xlabel(f"x ({coord})")
    ax.set_ylabel(f"y ({coord})")
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    if title is not None:
        ax.set_title(f"({coord}) plane")



def plot_classification_results(xyz, y, y_pred, xmin,xmax, ymin,ymax, labels=None, scores=None, title=None):
    fig = plt.figure(constrained_layout=True, figsize=(15,15))
    if title is not None:
        fig.suptitle(title)
    gs = GridSpec(3, 3, figure=fig)        

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])

    plot_scatter_dims(xyz[y==0], "r", ax1, plane="x,y",xlim=[xmin,xmax], ylim=[ymin,ymax])
    plot_scatter_dims(xyz[y==1], "g", ax2, plane="x,z",xlim=[xmin,xmax], ylim=[ymin,ymax])
    plot_scatter_dims(xyz[y==2], "b", ax3, plane="y,z",xlim=[xmin,xmax], ylim=[ymin,ymax])

    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1])
    ax3 = fig.add_subplot(gs[1,2])

    plot_scatter_dims(xyz[y_pred==0], "r", ax1,xlim=[xmin,xmax], ylim=[ymin,ymax])
    plot_scatter_dims(xyz[y_pred==1], "g", ax2,xlim=[xmin,xmax], ylim=[ymin,ymax])
    plot_scatter_dims(xyz[y_pred==2], "b", ax3,xlim=[xmin,xmax], ylim=[ymin,ymax])

    ax1 = fig.add_subplot(gs[2,0])
    ConfusionMatrixDisplay.from_predictions(y, 
                                            y_pred,normalize="true", 
                                            ax=ax1, 
                                            display_labels=labels)
    if scores:
        ax1 = fig.add_subplot(gs[2,1:])
        
        ax1.barh(range(len(scores)), list(scores.values()), align="center")
        ax1.set_yticks(range(len(scores)),list(scores.keys()))
        ax1.set_title("Scores")
        ax1.set_xlim(0,1)
        
        

        #scores_text = ""
        #for k in scores:
        #    scores_text += f"{k}: {scores[k]}\n"
        #ax1.text(0.1, 0.6, scores_text, fontsize="xx-large")
        #ax1.axis("off")
    
    return fig
    
def timestamp(dt):
    return (dt - datetime(1970, 1,1)).total_seconds()

def plot_predicted_magnetosphere_region(start, stop, t, x, sat_b, sw_n, xyz, clf, title=None, cmap="hot", figsize=(10, 10)):
    """Plot the predicted magnetosphere regions as returned by classifier `clf`. 
    
    Arguments:
        start (datetime) : prediction start time
        stop (datetime) : prediction stop time
        t (numpy.array) : full time data
        x (numpy.array) : full feature array
        sat_b (numpy.array) : full satellite magnetic field data
        sw_n (numpy.array) : full solar wind ion density data
        xyz (numpy.array) : full satellite position data (GSE)
        title (str, optional) : figure title
        cmap (ColorMap or str, optional) : color map regions
        figsize (tuple, optional) : figure size
    
    """
    if title is None:
        title = clf.__class__.__name__
    pred_start, pred_stop = timestamp(start), timestamp(stop)
    indx = (pred_start <= t) & (t < pred_stop)
    
    T = np.array([datetime.utcfromtimestamp(tt) for tt in t[indx]])

    # prediction
    y_pred = clf.predict(x[indx])
    
    C = np.repeat(y_pred.reshape(1,y_pred.shape[0]), 3, axis=0)
    
    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = GridSpec(5,1, figure=fig)
    
    # magnetic field data plot
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_title(title)
    ax1.plot(T, sat_b[indx])
    #axes[0].set_ylim(-20, 100)
    ax1.set_ylabel("sat_b")

    # ion density data plot
    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(T, sw_n[indx])
    ax2.set_yscale("log")
    ax2.set_ylabel("sat_n")

    # 
    ax3 = fig.add_subplot(gs[2,0])
    ax3.plot(T, y_pred)
    ax3.set_xlabel("Time")
    ax3.set_ylabel("y_pred")
    
    myFmt = mdates.DateFormatter('%H:%M')
    ax3.set_xlim(T[0], T[-1])
    
    ax1.pcolorfast(ax1.get_xlim(), ax1.get_ylim(),
                   C, cmap=cmap, alpha=.2)
    ax2.pcolorfast(ax2.get_xlim(), ax2.get_ylim(),
                   C, cmap=cmap, alpha=.2)
    ax3.pcolorfast(ax3.get_xlim(), ax3.get_ylim(),
                   C, cmap=cmap, alpha=.2)


    ax3.xaxis.set_major_formatter(myFmt)
    
    ax4 = fig.add_subplot(gs[3:,0])
    plot_magnetosphere(ax4)
    ax4.scatter(xyz[indx,0], xyz[indx,1], c=y_pred, cmap=cmap, marker=".", alpha=.2)
    xmin, xmax = xyz[indx,0].min(), xyz[indx,0].max()
    xmin, xmax = xmin - (xmax-xmin)*.1, xmax + (xmax-xmin)*.1
    ymin, ymax = xyz[indx,1].min(), xyz[indx,1].max()
    ymin, ymax = ymin - (ymax-ymin)*.1, ymax + (ymax-ymin)*.1
    
    ax4.set_aspect(1, adjustable="datalim")
    ax4.set_xlim(xmin,xmax)
    ax4.set_ylim(ymin,ymax)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_xlabel("x (GSE)")
    ax4.set_ylabel("y (GSE)")
    
    return fig

def plot_scores(scores, metrics):
    """Bar plot of scores.
    
    Arguments : 
    scores (dict of dict) : keys are method functions, values are dict object with keys metric names
    metrics : list of tuples (metric_name, metric_f)
    
    
    """
    method_labels = [m.__name__ for m in scores.keys()]
    n = len(method_labels)
    print((10, n*3))
    fig = plt.figure(constrained_layout=True, figsize=(10, n*3))
    gs = GridSpec(len(metrics), 1, figure=fig)


    ax0 = None
    for i,(n,m) in enumerate(metrics):
        if ax0 is None:
            ax = fig.add_subplot(gs[i,0])
            ax0 = ax
        else:
            ax = fig.add_subplot(gs[i,0], sharex=ax0)
        ax.set_title(n)
        # score values
        ax.barh(range(len(scores)), [s[n] for s in scores.values()])
    
        ax.set_yticks(range(len(scores)), method_labels)
    
        ax.set_xlim(0,1)
    return fig

def spectro_plot(param_id, start, stop, xlabel=None, ylabel=None, 
                 zlabel=None, yscale=None,
                 channels = None, ax=None, figsize=(10,2), 
                 vmin=None, vmax=None, lognorm=True, datefmt="%H:%M",
                 cmap=None):
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)
    # get the data
    param_data = spz.get_data(param_id, start, stop)
    [n,m] = param_data.data.shape
    X = param_data.data   
    
    # channels (constant channels case)
    if channels is None:
        y = np.arange(0,m,1)
    else:
        y = channels
    
    # grid
    x1, y1 = np.meshgrid(param_data.time,y, indexing="ij")
    
    # data bounds
    if vmin is None:
        vmin = np.nanmin(X)
    if vmax is None:
        vmax = np.nanmax(X)
    
    # colormap
    if not cmap:
        cmap = mpl.cm.rainbow.copy()
        cmap.set_bad('White',0.)
    
    # normalize colormapping
    if lognorm and vmin>0.:
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm=None
    
    
    c = ax.pcolormesh(x1, y1, X, cmap=cmap, norm=norm, edgecolors="face")
    cbar = plt.colorbar(c,ax=ax, norm=norm)
    if zlabel:
        cbar.set_label(zlabel)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    x_ticks = ax.get_xticks()
    x_ticks = [datetime.utcfromtimestamp(xi) for xi in x_ticks]
    x_labels = [d.strftime(datefmt) for d in x_ticks]
    
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(x_labels)
    
    if ylabel:
        ax.set_ylabel(ylabel)
    
    ax.set_ylim(y.min(), y.max())
    
    if yscale:
        ax.set_yscale(yscale)
    
    return ax
    
