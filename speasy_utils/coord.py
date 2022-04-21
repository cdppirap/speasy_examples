from scipy import constants as cst
import numpy as np

from speasy_utils import printProgressBar

def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(x / r)
    phi = np.arctan2(y, z)
    phi[z==0]=np.sign(y[z==0])*np.pi/2
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.sin(theta) * np.cos(phi)
    return x, y, z

def choice_coordinate_system(R, theta, phi, **kwargs):
    coord_sys = kwargs.get('coord_sys','cartesian')
    if coord_sys == 'cartesian':
        return spherical_to_cartesian(R, theta, phi)
    elif coord_sys == 'spherical':
        return R, theta, phi
    else:
        print('Error : coord_sys parameter must be set to "cartesian" or "spherical" ')

def bs_Jerab2005(theta, phi, **kwargs):
    '''
    Jerab 2005 Bow shock model. Give positions of the box shock in plans (XY) with Z=0 and (XZ) with Y=0 as a function of the upstream solar wind.
    function's arguments :
        - Np : Proton density of the upstream conditions
        - V  : Speed of the solar wind
        - B  : Intensity of interplanetary magnetic field
        - gamma : Polytropic index ( default gamma=2.15)
        --> mean parameters :  Np=7.35, V=425.5,  B=5.49
     return : DataFrame (Pandas) with the position (X,Y,Z) in Re of the bow shock to plot (XY) and (XZ) plans.
    '''

    def make_Rav(theta, phi):
        a11 = 0.45
        a22 = 1
        a33 = 0.8
        a12 = 0.18
        a14 = 46.6
        a24 = -2.2
        a34 = -0.6
        a44 = -618

        x = np.cos(theta)
        y = np.sin(theta) * np.sin(phi)
        z = np.sin(theta) * np.cos(phi)

        a = a11 * x ** 2 +  a22 * y ** 2 + a33 * z ** 2 + a12 * x * y
        b = a14 * x + a24 * y + a34 * z
        c = a44

        delta = b ** 2 - 4 * a * c

        R = (-b + np.sqrt(delta)) / (2 * a)
        return R

    Np = kwargs.get('Np', 6.025)
    V = kwargs.get('V', 427.496)
    B = kwargs.get('B', 5.554)
    gamma = kwargs.get('gamma', 5./3)
    Ma = V * 1e3 * np.sqrt(Np * 1e6 * cst.m_p * cst.mu_0) / (B * 1e-9)

    C = 91.55
    D = 0.937 * (0.846 + 0.042 * B)
    R0 = make_Rav(0, 0)

    Rav = make_Rav(theta, phi)
    K = ((gamma - 1) * Ma ** 2 + 2) / ((gamma + 1) * (Ma ** 2 - 1))
    r = (Rav / R0) * (C / (Np * V ** 2) ** (1 / 6)) * (1 + D * K)

    return choice_coordinate_system(r, theta, phi, **kwargs)        

def mp_shue1997(theta, phi, **kwargs):
    Pd = kwargs.get("Pd", 2.056)
    Bz = kwargs.get("Bz", -0.001)

    if isinstance(Bz, float) | isinstance(Bz, int):
        if Bz >= 0:
            r0 = (11.4 + 0.13 * Bz) * Pd ** (-1 / 6.6)
        else:
            r0 = (11.4 + 0.14 * Bz) * Pd ** (-1 / 6.6)
    else:
        if isinstance(Pd, float) | isinstance(Pd, int):
            Pd = np.ones_like(Bz) * Pd
        r0 = (11.4 + 0.13 * Bz) * Pd ** (-1 / 6.6)
        r0[Bz < 0] = (11.4 + 0.14 * Bz[Bz < 0]) * Pd[Bz < 0] ** (-1 / 6.6)
    a = (0.58 - 0.010 * Bz) * (1 + 0.010 * Pd)
    r = r0 * (2. / (1 + np.cos(theta))) ** a
    return choice_coordinate_system(r, theta, phi, **kwargs)
    
def get_position_class(x, **kwargs):
    
    # spherical coordinates
    r,theta,phi = cartesian_to_spherical(x[:,0],x[:,1],x[:,2])
    # bow shock
    bs_x, bs_y, bs_z = bs_Jerab2005(theta, phi, **kwargs)
    bs_dist = np.linalg.norm(np.vstack((bs_x,bs_y,bs_z)).T, axis=1)
    # magnetopause
    mp_x, mp_y, mp_z = mp_shue1997(theta, phi, **kwargs)
    mp_dist = np.linalg.norm(np.vstack((mp_x,mp_y,mp_z)).T, axis=1)
    
    # x norm
    x_dist = np.linalg.norm(x, axis=1)
    
    ans = np.zeros(x.shape[0]).astype(int)
    ans[x_dist <= mp_dist] = 2 # magnetoshere
    ans[(mp_dist < x_dist) & (x_dist <= bs_dist)] = 1 # magnetosheath
    ans[bs_dist < x_dist] = 0 # solar wind
    
    return ans
def get_regions_dyn(x, sw_pdyn, sw_bz, sw_n, sw_V, sw_B):
    [n,m]=x.shape
    l=[]
    last_p=None
    for i in range(x.shape[0]):
        p = int(100. * float(i)/n)
        if last_p is None or last_p != p:
            printProgressBar(p, 100., prefix = "regions_dyn:", length=50)
            last_p = p

        c=get_position_class(x[i].reshape(1,m), Pd=sw_pdyn[i], Bz=sw_bz[i], Np=sw_n[i], V=sw_V[i], B=sw_B[i])
        l.append(c[0])
    return np.array(l)

