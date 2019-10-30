from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.interpolate import interp1d,interp2d
import pandas as pd

#Inputs are: Teff, logg, K, inclination, P_orb

#Compute: M1, M2, EV, DB, R1, a, tau_merge

#Define constants
G = 6.67259e-8
MSun = 1.99e33 #g
RSun = 6.96e10 #cm
c = 2.99792458e10 #cm/s

#For mass: bilinear interpolation of model grid of Althaus et al. (2013, A&A, 557, A19)
logggrid = np.array([4.,4.7,5.4,5.8,6.1,6.25,6.4,6.7,7.43])
logTeffgrid = np.array([[3.92,4.00,4.03,4.20,4.40],
                    [3.92,4.00,4.03,4.20,4.50],
                    [3.99,4.10,4.15,4.30,4.50],
                    [3.96,4.10,4.15,4.30,4.50],
                    [3.90,4.05,4.15,4.30,4.50],
                    [3.91,4.00,4.10,4.30,4.50],
                    [3.93,4.05,4.15,4.30,4.50],
                    [3.80,4.00,4.15,4.30,4.50],
                    [3.80,4.00,4.15,4.30,4.40]])

massgrid = np.array([[0.191,0.215,0.225,0.278,0.346],
                    [0.158,0.174,0.181,0.232,0.296],
                    [0.156,0.176,0.187,0.218,0.242],
                    [0.156,0.176,0.184,0.202,0.235],
                    [0.155,0.173,0.182,0.207,0.240],
                    [0.163,0.177,0.175,0.219,0.249],
                    [0.174,0.185,0.187,0.232,0.262],
                    [0.180,0.202,0.221,0.264,0.310],
                    [0.347,0.369,0.391,0.415,0.435]])

masserrgrid=np.array([[0.0019,0.0032,0.0038,0.0041,0.0267],
                     [0.0004,0.0009,0.0013,0.0056,0.0257],
                     [0.0003,0.0004,0.0016,0.0160,0.0307],
                     [0.0007,0.0020,0.0029,0.0152,0.0301],
                     [0.0002,0.0064,0.0079,0.0129,0.0295],
                     [0.0002,0.0004,0.0074,0.0119,0.0286],
                     [0.0004,0.0052,0.0056,0.0108,0.0230],
                     [0.0006,0.0018,0.0032,0.0086,0.0076],
                     [0.0020,0.0033,0.0031,0.0009,0.0050]])

def getMass(teff,logg):
    logteff = np.log10(teff)
    #interpolate first in logTeff:
    interpedmasses = np.zeros(logggrid.shape)
    interpedmasserrs = np.zeros(logggrid.shape)
    for i in range(logggrid.shape[0]):
        interpTeffMass = interp1d(logTeffgrid[i,:],massgrid[i,:],fill_value="extrapolate",bounds_error=False)
        interpTeffMasserr = interp1d(logTeffgrid[i,:],masserrgrid[i,:],fill_value="extrapolate",bounds_error=False)
        interpedmasses[i] = interpTeffMass(logteff)
        interpedmasserrs[i] = interpTeffMasserr(logteff)
    interp2 = interp1d(logggrid,interpedmasses)
    interp2err = interp1d(logggrid,interpedmasserrs)
    return (interp2(logg),interp2err(logg))


#define function to get limb and gravity darkening (representative)
def getgravdarkening(Teff):
    BG40lambda = 5000.
    beta = 0.25
    return beta * (1.43879e8/(BG40lambda*Teff))/(1.-np.exp(-1.43879e8/(BG40lambda*Teff)))

#Limb Darkening coefficients from Gianninas et al. (2013, ApJ, 766, 3)
ld = np.loadtxt('limbdarkening_g.dat')
ldlogg=ld[:,0]
ldTeff=ld[:,1]
ldlinear=ld[:,2]
ldlogguniq = np.unique(ldlogg)
ldTeffuniq = np.unique(ldTeff)
ldcoeffgrid = ldlinear.reshape((ldlogguniq.shape[0],ldTeffuniq.shape[0]))
getlimbdarkening = interp2d(ldlogguniq,ldTeffuniq,ldcoeffgrid.T)#input logg, Teff

def getM2(M1,P,K,i):
    M2sample = np.linspace(0,3,30)*MSun #Set max to 3 to avoid BH companion.
    M2interp = interp1d(((M2sample*np.sin(i))**3./(M1+M2sample)**2),M2sample,bounds_error=False,fill_value="extrapolate")
    return M2interp(P*K**3./(2.*np.pi*G))

fig, ax = plt.subplots()

plt.subplots_adjust(left=0.12, bottom=0.4,top=0.8)
phasesample = np.linspace(0,1,101)
l, = ax.plot(phasesample,phasesample, lw=2)
ax.set_ylabel("rel. flux (%)")
ax.set_xlabel("phase")
ax.set_ylim(-10,10)
ax.margins(x=0)

topax = ax.twiny()
topax.set_xticks([0,0.25,0.5,0.75,1])
topax.set_xticklabels(['farthest','toward','nearest','away','farthest'])

axcolor = 'lightgoldenrodyellow'

axteff = plt.axes([0.25, 0.05, 0.6, 0.03], facecolor=axcolor)
axlogg = plt.axes([0.25, 0.1, 0.6, 0.03], facecolor=axcolor)
axporb = plt.axes([0.25, 0.15, 0.6, 0.03], facecolor=axcolor)
axk = plt.axes([0.25, 0.2, 0.6, 0.03], facecolor=axcolor)
axinc = plt.axes([0.25, 0.25, 0.6, 0.03], facecolor=axcolor)


steff = Slider(axteff, 'Teff (K)', 8300, 30000, valinit=10000, valfmt=u'%1.0f') #valstep=50, 
slogg = Slider(axlogg, 'log(g) (cgs)', 4, 7.4, valinit=5.5, valfmt=u'%1.1f') #valstep = 0.1, 
sporb = Slider(axporb, 'P_orb (hours)', 0.25, 24, valinit=2 ) #valstep = 0.05
sk = Slider(axk, 'K (km/s)', 20, 620, valinit=200,  valfmt=u'%1.0f') #valstep = 10,
sinc = Slider(axinc, 'inc (deg)', 1, 90, valinit=45, valfmt=u'%1.0f') #valstep = 1, 

M1=0
M2=0
R1=0

def calc():
    M1,_ = getMass(steff.val,slogg.val)
    ulimb = getlimbdarkening(slogg.val,steff.val)
    tau = getgravdarkening(steff.val)
    R1 = np.sqrt(G*M1*MSun/(10.**slogg.val))/RSun
    M2 = getM2(M1*MSun,sporb.val*3600,sk.val*1e5,sinc.val*np.pi/180.)/MSun
    A = np.power((sporb.val*3600)**2.*G*(M1+M2)*MSun/(4.*np.pi**2.),1./3.)/RSun
    merge = np.power((M1+M2),1./3.)*np.power(sporb.val,8./3.)*1e-2/(M1*M2)
    DB = 100*2*sk.val*1e5/c #approx.
    
    #Calc ellipsoidal vars
    numer=3.*np.pi**2.*(15. + ulimb)*(1. + tau)*M2*MSun*(R1*RSun)**3.*np.sin(sinc.val*np.pi/180.)**2.
    denom=5.*(sporb.val*3600.)**2.*(3.-ulimb)*G*M1*(M1+M2)*MSun**2.
    EV=100.*numer/denom
    return pd.DataFrame({"M1 (MSun)":M1,"R1 (RSun)":R1,"M2 (MSun)":M2,"A (RSun)":A,
                         "merge (Gyr)":merge,"EV (%)":EV,"DB (%)":DB},
                    columns = ["M1 (MSun)","R1 (RSun)","M2 (MSun)","A (RSun)","merge (Gyr)","EV (%)","DB (%)"])

def update(val):
    df = calc()
    l.set_ydata(-df["EV (%)"][0]*np.cos(phasesample*4*np.pi) + 
                df["DB (%)"][0]*np.sin(phasesample*2*np.pi))
    
    for i in range(len(df.values[0])):
        mpl_table._cells[(1, i)]._text.set_text('%.4f' % df.values[0][i])
    
    fig.canvas.draw_idle()

df = calc()

axvals = plt.axes([0.12, 0.88, 0.78, 0.1], facecolor=axcolor)
font_size=8
bbox=[0, 0, 1, 1]
axvals.axis('off')
mpl_table = axvals.table(cellText = [['%.4f' % j for j in i] for i in df.values],
                         bbox=bbox, colLabels=df.columns)
mpl_table.auto_set_font_size(False)
mpl_table.set_fontsize(font_size)

update(1)


steff.on_changed(update)
slogg.on_changed(update)
sporb.on_changed(update)
sk.on_changed(update)
sinc.on_changed(update)

plt.show()