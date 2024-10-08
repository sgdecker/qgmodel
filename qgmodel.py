#    qgmodel.py: An Interactive Analytic QG Model for Synoptic Meteorology
#    Copyright 2014 Steven Decker

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from matplotlib import colormaps
from matplotlib import ticker
from scipy import integrate

Re = 6370000.
g = 9.81
R = 287.
cp = 1004.
kappa = R / cp
T0 = 250.
gamma = 0.097
lat = 40.
lat_rad = np.radians(lat)
omega = 7.292e-5
f0 = 2*omega*np.sin(lat_rad)
beta = 2*omega*np.cos(lat_rad) / Re
fmti0 = ticker.FormatStrFormatter("%.0f")
vortint = range(-28, 36, 4)

def intro():
    """Displays introduction to the program."""

    print("An Interactive Analytic QG Model for Synoptic Meteorology")
    print(u"         Copyright \u00a9 2014 Steven G. Decker")
    print()
    print("You are viewing a Python implementation of the analytic QG model")
    print("presented by Frederick Sanders in the May 1971 issue of")
    print("Monthly Weather Review.")
    print()
    print("Use the checkboxes to turn on/off various fields.  Note that vectors")
    print("may end up underneath filled contours depending on the order with")
    print("which you click things.  A restart will fix this.")
    print()
    print("A number of model parameters may be changed with the sliders:")
    print("p      - Which isobaric surface to view [hPa]")
    print("L      - Wavelength [km] (i.e., distance between T/Z maxima)")
    print(u"\u03bb/L    - Phase shift between T and \u03a6 perturbations")
    print("a      - Temperature gradient of background [K/km]")
    print("p_trop - Tropopause pressure [hPa]")
    print("T_hat  - Temperature perturbation magnitude [K]")
    print(u"\u03a6\u2080_hat - Geopotential perturbation magnitude [J/kg]")

def calc_dx(x):
    """Computes the grid spacing."""

    dx = x[1] - x[0]
    return dx

def cint2lev(arr, cint):
    """Determines appropriate contour levels for a plt.

    arr -- The array that is about to be contoured
    cint -- The requested contour interval"""
    if cint <= 0.:
        return []
    lb = arr.min()
    ub = arr.max()
    first = 0.
    while True:
        if first > lb:
            break
        first += cint
    cur = first
    res = [cur]
    while True:
        if cur > ub:
            break
        cur += cint
        res.append(cur)
    return res

def stdatmt(p):
    """The temperature [K] of the standard atmosphere profile at pressure p.

    p -- The requested pressure [mb]"""
    tarr = np.array([     216.7,       216.8,       220. ,       223.3,  \
                          226.5,       229.7,       233. ,       236.2,  \
                          239.5,       242.7,       245.9,       249.2,  \
                          252.4,       255.7,       258.9,       262.2,  \
                          265.4,       268.7,       271.9,       275.2,  \
                          278.4,       281.7,       284.9,       288.1,  \
                          291.4])
    lnp = np.array([ 5.34639299,  5.42495002,  5.50288953,  5.57972983,  \
                      5.6554672,  5.73013225,  5.80374847,  5.87639034,  \
                     5.94803499,  6.01871486,  6.08847782,  6.15736027,  \
                     6.22533041,  6.29245763,  6.35876039,  6.42422052,  \
                     6.48890093,  6.55280741,  6.61595808,  6.67835469,  \
                     6.74004643,  6.80101604,  6.86130288,  6.92091827,  \
                     6.97987127])

    return np.interp(np.log(p), lnp, tarr)

def integrand(p):
    """A function needed to compute the geopotential profile."""
    return stdatmt(p/100.) / p

def phimcalc(p):
    """The geopotential [J/kg] of the standard atmosphere profile at pressure p.

    p -- The requested pressure [mb]"""
    res, err = integrate.quad(integrand, 100*p, 100000., limit=80)
    return R * res

#def gamma(p):
#    pPa = 100. * p
#    p1 = p - 25.
#    p2 = p + 25.
#    dpPa = 100. * (p2 - p1)
#    deriv = (np.log(stdatmt(p2)) - np.log(stdatmt(p1))) / dpPa
#    return kappa - pPa * deriv
    
def sigma(p):
    """A model stability parameter at pressure p [mb]."""
    pPa = 100. * p
    return R * T0 * gamma / pPa**2

def phi1000(x, y, phi0caret, L, lamb):
    """Sanders' specification for the 1000-mb geopotential field."""
    return phi0caret * np.cos(2*np.pi*(x+lamb)/L) * np.cos(2*np.pi*y/L)

def phi3d(x, y, p, phi0caret, L, lamb, a, Tcaret, alpha):
    """The three-dimensional geopotential field."""
    lnp = np.log(p/1000.)
    return phimcalc(p) + phi1000(x, y, phi0caret, L, lamb)  \
        + R * (a*y + Tcaret * np.cos(2*np.pi*x/L) * np.cos(2*np.pi*y/L))  \
        * (lnp + .5*alpha*lnp**2)

def T(x, y, p, alpha, a, Tcaret, L):
    """The three-dimensional temperature field as specified by Sanders."""
    return stdatmt(p) - (1 - alpha * np.log(1000./p))  \
        * (a*y + Tcaret * np.cos(2*np.pi*x/L) * np.cos(2*np.pi*y/L))

def vortadvf(x, y, p, L, Tcaret, alpha, a, phi0caret, lamb):
    """Computes vorticity advection forcing analytically."""
    twopiL = 2*np.pi/L
    twopiLSI = 2*np.pi/(L*1000.)
    ln1000p = np.log(1000./p)
    one_minus_alpha_term = 1 - alpha*ln1000p
    pPa = 100. * p
    part1 = (4. * twopiLSI**3 * R*(a/1000.)*Tcaret * pPa * ln1000p  \
                 * one_minus_alpha_term * (1-.5*alpha*ln1000p)  \
                 * np.sin(twopiL*x) * np.cos(twopiL*y)) / (f0 * T0 * gamma)
    part2 = (-twopiLSI*Tcaret*beta * pPa * one_minus_alpha_term  \
                 * np.sin(twopiL*x) * np.cos(twopiL*y)) / (T0 * gamma)
    part3 = (-2. * twopiLSI**3 * (a/1000.)*phi0caret * pPa  \
                  * one_minus_alpha_term * np.sin(twopiL*(x+lamb))  \
                  * np.cos(twopiL*y)) / (f0 * T0 * gamma) 

    res = part1 + part2 + part3
    return 1e12 * res

def tempadvf(x, y, p, L, Tcaret, alpha, a, phi0caret, lamb):
    """Computes temperature advection forcing analytically."""
    twopiL = 2*np.pi/L
    twopiLSI = 2*np.pi/(1000.*L)
    ln1000p = np.log(1000./p)
    one_minus_alpha_term = 1 - alpha*ln1000p
    pPa = 100. * p
    part1 = (-2. * twopiLSI**3 * (a/1000.)*phi0caret * pPa  \
                  * one_minus_alpha_term * np.sin(twopiL*(x+lamb))  \
                  * np.cos(twopiL*y)) / (f0 * T0 * gamma)
    part2 = 2 * twopiLSI**4 * phi0caret * Tcaret * pPa * one_minus_alpha_term  \
        * np.sin(twopiL*lamb) * np.sin(2*twopiL*y) / (f0 * T0 * gamma)

    res = part1 + part2
    return 1e12 * res

def omegaf(x, y, p, L, Tcaret, alpha, a, phi0caret, lamb):
    """Computes vertical motion [ub/s] analytically."""
    twopiL = 2*np.pi/L
    twopiLSI = 2*np.pi/(1000.*L)
    ln1000p = np.log(1000./p)
    pPa = 100. * p
    denom = f0 * T0 * gamma
    a1000 = a / 1000.

    # Omega1
    K11 = twopiLSI * R * a1000 * Tcaret / denom
    K12 = Tcaret * beta / (2 * twopiLSI * T0 * gamma)

    k = f0**2 / (2 * twopiLSI**2 * R * T0 * gamma)
    qm1 = .5 * np.sqrt(1 + 4./k) - .5
    one_minus_p_qm1 = 1 - (p/1000.)**qm1

    P11 = pPa * ((2*k + 6*alpha*k*(k+1) + 6*alpha**2*k**2*(k+2))  \
                     * one_minus_p_qm1  \
                     - (2 + 6*alpha*k + 6*alpha**2*k*(k+1)) * ln1000p  \
                     + (3*alpha + 3*alpha**2*k) * ln1000p**2  \
                     - alpha**2 * ln1000p**3)
    P12 = pPa * ((alpha*k + 1) * one_minus_p_qm1 - alpha * ln1000p)

    omega1 = (K11*P11 + K12*P12) * np.sin(twopiL*x) * np.cos(twopiL*y)

    # Omega2
    K2 = 2 * twopiLSI * a1000 * phi0caret / denom

    omega2 = K2*P12 * np.sin(twopiL*(x+lamb)) * np.cos(twopiL*y)
    
    # Omega3
    K3 = twopiLSI**2 * phi0caret * Tcaret * np.sin(twopiL*lamb) / (2 * denom)

    rm1 = .5 * np.sqrt(1 + 8./k) - .5
    P3 = pPa * (alpha * ln1000p - (.25*alpha*k + 1) * (1 - (p/1000.)**rm1))

    omega3 = K3*P3 * np.sin(2*twopiL*y)

    res = 10 * (omega1 + omega2 + omega3)  # Convert Pa/s to ub/s
    return res


class Field(object):
    """The base class for any field we may wish to compute using the model.

    Classes that extend this class must define a cc method that generates the
    appropriate plot."""

    def __init__(self, visibility, name, label):
        """Initializes a Field object."""
        self.visibility = visibility
        self.created = visibility
        self.name = name
        self.label = label
        self.contset = []
        self.midlabel = "Advection/Forcing Terms (scaled SI units)"

    def set_coords(self, coords):
        """Sets up the coordinates for the field."""
        self.x = coords["x"]
        self.y = coords["y"]
        self.xx = coords["xx"]
        self.yy = coords["yy"]
        self.dim1 = self.yy.shape[0]
        self.dim2 = self.xx.shape[1]
        self.vals = np.empty((self.dim1,self.dim2))
        self.vals2 = np.empty((self.dim1,self.dim2))

    def create_contours(self, fig, ax, params, coords):
        """Creates the plot of the field."""
        self.set_coords(coords)

        if self.visibility:
            self.cc(fig, ax, params, coords)
        else:
            self.created = False

    def update_visibility(self, fig, ax, params, coords):
        """Updates the field's visibility."""
        self.visibility = not self.visibility
        if self.visibility:
            if self.created:
                self.contset.set_alpha(1.)
            else:
                self.cc(fig, ax, params, coords)
                self.created = True
        else:
            self.contset.set_alpha(0.)


class RelativeVorticity(Field):
    def __init__(self):
        Field.__init__(self, False, "zetag", r"$\zeta_g$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        self.contset = ax.contourf(self.x, self.y, self.vals, levels=vortint)
    
    def compute_vals(self, params, coords):
        dx = calc_dx(self.x)

        z = GeopotentialHeight()
        z.set_coords(coords)
        z.compute_vals(params)

        phi = 10 * g * z.vals  # Convert back to geopotential

        for j in range(1, self.dim1-1):
            self.vals[j,0] = (phi[j,-2] + phi[j-1,0] -4*phi[j,0] + phi[j+1,0]  \
                                  + phi[j,1])
            for i in range(1, self.dim2-1):
                self.vals[j,i] = (phi[j,i-1] + phi[j-1,i] -4*phi[j,i]  \
                                      + phi[j+1,i]  \
                                      + phi[j,i+1])
            self.vals[j,-1] = (phi[j,-2] + phi[j-1,-1] -4*phi[j,-1]  \
                                   + phi[j+1,-1]  \
                                   + phi[j,1])
        self.vals *= 1e5 / (f0 * (1000.*dx)**2)  # Convert result to 10^-5 s-1
        self.vals[np.ix_([0,self.dim1-1])] = np.nan

class CoriolisParameter(Field):
    def __init__(self):
        Field.__init__(self, False, "f", r"$f$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals()
        self.contset = ax.contourf(self.x, self.y, self.vals, levels=vortint)

    def compute_vals(self):
        self.vals = np.transpose(  \
            np.tile(1e5 * (f0 + 1000. * beta * self.y), (self.dim2,1)))


class AbsoluteVorticity(Field):
    def __init__(self):
        Field.__init__(self, True, "eta", r"$\zeta_g+f$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        self.contset = ax.contourf(self.x, self.y, self.vals, levels=vortint)
        cbar_ax = fig.add_axes([0.86, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(r"Vorticity ($10^{-5}$ s$^{-1}$)", labelpad=-9)

    def compute_vals(self, params, coords):
        f = CoriolisParameter()
        f.set_coords(coords)
        f.compute_vals()

        z = RelativeVorticity()
        z.set_coords(coords)
        z.compute_vals(params, coords)

        self.vals = f.vals + z.vals


class GeopotentialHeight(Field):
    def __init__(self):
        Field.__init__(self, True, "Z", r"$Z(p)$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params)
        lev = cint2lev(self.vals, 6.)
        self.contset = ax.contour(self.x, self.y, self.vals, colors="k",  \
                                      linewidths=2, levels=lev)
        self.contset.clabel(fmt=fmti0)

    def compute_vals(self, params):
        L = params["L"]
        phase = params["phase"]
        phi0caret = params["phi0caret"]
        ptrop = params["ptrop"]
        Tcaret = params["Tcaret"]
        a = params["a"]
        p = params["p"]

        lamb = L * phase
        alpha = 1. / np.log(1000./ptrop)
        phi1 = phi3d(self.xx, self.yy, p, phi0caret, L, lamb, a, Tcaret, alpha)
        self.vals = phi1 / (10*g)  # Convert from geopotential to height in dam


class GeopotentialHeight1000(Field):
    def __init__(self):
        Field.__init__(self, False, "Z1000", r"$Z_{1000}$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params)
        self.contset = ax.contour(self.x, self.y, self.vals, colors=".2",  \
                                      levels=range(-18,21,3))
        vlist = []
        for v in self.contset.levels:
            if v != 0.:
                vlist.append(v)
        self.contset.clabel(vlist, fmt=fmti0)
        for i, val in enumerate(self.contset.levels):
            if val == 0.:
                self.contset.collections[i].remove()
                break

    def compute_vals(self, params):
        L = params["L"]
        phase = params["phase"]
        phi0caret = params["phi0caret"]

        lamb = L * phase

        phi1 = phi1000(self.xx, self.yy, phi0caret, L, lamb)
        self.vals = phi1 / (10*g)  # Convert from geopotential to height in dam


class Temperature(Field):
    def __init__(self):
        Field.__init__(self, False, "T3", r"$T(p)$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params)
        self.contset = ax.contour(self.x, self.y, self.vals,  \
                                      levels=range(190,340,5))
        self.contset.clabel(fmt=fmti0)

    def compute_vals(self, params):
        L = params["L"]
        ptrop = params["ptrop"]
        a = params["a"]
        Tcaret = params["Tcaret"]
        p = params["p"]

        alpha = 1. / np.log(1000./ptrop)
        self.vals = T(self.xx, self.yy, p, alpha, a, Tcaret, L)


class Temperature1000(Temperature):
    def __init__(self):
        Field.__init__(self, False, "T", r"$T_{1000}$")

    def compute_vals(self, params):
        L = params["L"]
        ptrop = params["ptrop"]
        a = params["a"]
        Tcaret = params["Tcaret"]
        p = 1000.

        alpha = 1. / np.log(1000./ptrop)
        self.vals = T(self.xx, self.yy, p, alpha, a, Tcaret, L)
        

class Thickness(Field):
    def __init__(self):
        Field.__init__(self, False, "thick", r"$\Delta Z_{1000-p}$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        lev = cint2lev(self.vals, 6.)
        self.contset = ax.contour(self.x, self.y, self.vals, colors="k",  \
                                      linewidths=2, linestyles="--", levels=lev)
        self.contset.clabel(fmt=fmti0)

    def compute_vals(self, params, coords):
        z = GeopotentialHeight()
        z.set_coords(coords)
        z.compute_vals(params)

        z1000 = GeopotentialHeight1000()
        z1000.set_coords(coords)
        z1000.compute_vals(params)

        self.vals = z.vals - z1000.vals


class GeostrophicWind(Field):
    def __init__(self):
        Field.__init__(self, False, "Vg", r"$\mathbf{V}_g$")

    def set_coords(self, coords):
        Field.set_coords(self, coords)
        self.vals2 = np.empty((self.dim1,self.dim2))
        
    def cc(self, fig, ax, params, coords):
        self.compute_vals(params)
        self.contset = ax.quiver(self.x[::3], self.y[::3],  \
                                     self.vals[::3,::3], self.vals2[::3,::3],  \
                                     angles="xy", scale_units="xy", scale=.3,  \
                                     color="#440000")

    def compute_vals(self, params):
        L = params["L"]
        phase = params["phase"]
        phi0caret = params["phi0caret"]
        ptrop = params["ptrop"]
        Tcaret = params["Tcaret"]
        a = params["a"]
        p = params["p"]

        dx = calc_dx(self.x)

        # Geopotential
        lamb = L * phase
        alpha = 1. / np.log(1000./ptrop)
        phi = phi3d(self.xx, self.yy, p, phi0caret, L, lamb, a, Tcaret, alpha)

        # Ug
        self.vals.fill(np.nan)
        for j in range(1, self.dim1-1):
            self.vals[j,:] = phi[j-1,:] - phi[j+1,:]
        self.vals = self.vals / (f0 * 2000.*dx)

        #Vg
        self.vals2.fill(np.nan)
        for j in range(self.dim1):
            self.vals2[j,1:self.dim2-1] = phi[j,2:] - phi[j,:self.dim2-2]
        self.vals2 = self.vals2 / (f0 * 2000.*dx)

class VorticityAdvection(Field):
    def __init__(self):
        Field.__init__(self, False, "va", "Vort. Adv.")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        cmap = colormaps["winter"]
        self.contset = ax.contourf(self.x, self.y, self.vals,  \
                                       levels=range(-10,11), cmap=cmap)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(self.midlabel, labelpad=-8)

    def compute_vals(self, params, coords):
        eta = AbsoluteVorticity()
        eta.set_coords(coords)
        eta.compute_vals(params, coords)

        Vg = GeostrophicWind()
        Vg.set_coords(coords)
        Vg.compute_vals(params)

        dx = 1000. * calc_dx(self.x)
        self.vals.fill(np.nan)
        for j in range(1, self.dim1-1):
            for i in range(1, self.dim2-1):
                self.vals[j,i] = 1e4 * (Vg.vals[j,i]  \
                                            * (eta.vals[j,i-1]  \
                                                   - eta.vals[j,i+1])  \
                                            + Vg.vals2[j,i]  \
                                            * (eta.vals[j-1,i]  \
                                                   - eta.vals[j+1,i])) / dx


class DifferentialVorticityAdvection(Field):
    def __init__(self):
        Field.__init__(self, False, "dva", "Diff. VA")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        cmap = colormaps["winter"]
        self.contset = ax.contourf(self.x, self.y, self.vals,  \
                                       levels=range(-10,11), cmap=cmap)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(self.midlabel, labelpad=-8)

    def compute_vals(self, params, coords):
        local_params = params
        dp = 5000.  # 50 mb

        va1 = VorticityAdvection()
        va1.set_coords(coords)
        local_params["p"] = params["p"] - .005 * dp
        va1.compute_vals(local_params, coords)

        va2 = VorticityAdvection()
        va2.set_coords(coords)
        local_params["p"] = params["p"] + .005 * dp
        va2.compute_vals(local_params, coords)

        self.vals = 1e3 * f0 * (va1.vals - va2.vals) / (dp * sigma(params["p"]))


class DifferentialVorticityAdvectionAnalytic(Field):
    def __init__(self):
        Field.__init__(self, False, "vae", "DVA Term")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params)
        cmap = colormaps["winter"]
        self.contset = ax.contourf(self.x, self.y, self.vals,  \
                                       levels=range(-10,11), cmap=cmap)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(self.midlabel, labelpad=-8)

    def compute_vals(self, params):
        L = params["L"]
        ptrop = params["ptrop"]
        Tcaret = params["Tcaret"]
        p = params["p"]
        a = params["a"]
        phi0caret = params["phi0caret"]
        phase = params["phase"]

        lamb = L * phase
        alpha = 1. / np.log(1000./ptrop)
        self.vals = vortadvf(self.xx, self.yy, p, L, Tcaret, alpha, a,  \
                                 phi0caret, lamb)


class TemperatureAdvection(Field):
    def __init__(self):
        Field.__init__(self, False, "ta", "Temp. Adv.")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        cmap = colormaps["winter"]
        self.contset = ax.contourf(self.x, self.y, self.vals,  \
                                       levels=range(-10,11), cmap=cmap)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(self.midlabel, labelpad=-8)

    def compute_vals(self, params, coords):
        t = Temperature()
        t.set_coords(coords)
        t.compute_vals(params)

        Vg = GeostrophicWind()
        Vg.set_coords(coords)
        Vg.compute_vals(params)

        dx = 1000. * calc_dx(self.x)
        self.vals.fill(np.nan)
        for j in range(1, self.dim1-1):
            for i in range(1, self.dim2-1):
                self.vals[j,i] = 1e4 * (Vg.vals[j,i]  \
                                            * (t.vals[j,i-1]  \
                                                   - t.vals[j,i+1])  \
                                            + Vg.vals2[j,i]  \
                                            * (t.vals[j-1,i]  \
                                                   - t.vals[j+1,i])) / dx


class LapTemperatureAdvectionAnalytic(Field):
    def __init__(self):
        Field.__init__(self, False, "tae", "LTA Term")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params)
        cmap = colormaps["winter"]
        self.contset = ax.contourf(self.x, self.y, self.vals,  \
                                       levels=range(-10,11), cmap=cmap)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(self.midlabel, labelpad=-8)

    def compute_vals(self, params):
        L = params["L"]
        ptrop = params["ptrop"]
        Tcaret = params["Tcaret"]
        p = params["p"]
        a = params["a"]
        phi0caret = params["phi0caret"]
        phase = params["phase"]

        lamb = L * phase
        alpha = 1. / np.log(1000./ptrop)
        self.vals = tempadvf(self.xx, self.yy, p, L, Tcaret, alpha, a,  \
                                 phi0caret, lamb)


class TotalForcing(Field):
    def __init__(self):
        Field.__init__(self, False, "tae", "DVA + LTA")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        cmap = colormaps["winter"]
        self.contset = ax.contourf(self.x, self.y, self.vals,  \
                                       levels=range(-20,21), cmap=cmap)
        cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(self.midlabel, labelpad=-8)

    def compute_vals(self, params, coords):
        va = DifferentialVorticityAdvectionAnalytic()
        va.set_coords(coords)
        va.compute_vals(params)

        ta = LapTemperatureAdvectionAnalytic()
        ta.set_coords(coords)
        ta.compute_vals(params)

        self.vals = va.vals + ta.vals

        
class QVector(Field):
    def __init__(self):
        Field.__init__(self, False, "Q", r"$\mathbf{Q}$")

    def set_coords(self, coords):
        Field.set_coords(self, coords)
        self.vals = np.zeros((self.dim1,self.dim2))
        self.vals2 = np.zeros((self.dim1,self.dim2))
        
    def cc(self, fig, ax, params, coords):
        self.compute_vals(params, coords)
        self.contset = ax.quiver(self.x[2::3], self.y[2::3],
                                 self.vals[2::3, 2::3],
                                 self.vals2[2::3, 2::3], angles="xy", pivot="mid",
                                 scale_units="xy", scale=1, color="#660066")
    
    def compute_vals(self, params, coords):
        Vg = GeostrophicWind()
        Vg.set_coords(coords)
        Vg.compute_vals(params)

        T = Temperature()
        T.set_coords(coords)
        T.compute_vals(params)

        p = params["p"]
        pPa = 100 * p
        dx = 1000. * calc_dx(self.x)

        # Q1
        for j in range(2, self.dim1-2):
            for i in range(2, self.dim2-2):
                self.vals[j,i] = (Vg.vals[j,i+1] - Vg.vals[j,i-1])  \
                                      * (T.vals[j,i+1] - T.vals[j,i-1])  \
                                      + (Vg.vals2[j,i+1] - Vg.vals2[j,i-1])  \
                                      * (T.vals[j+1,i] - T.vals[j-1,i])
        # Q2
        for j in range(2, self.dim1-2):
            for i in range(2, self.dim2-2):
                self.vals2[j,i] = (Vg.vals[j+1,i] - Vg.vals[j-1,i])  \
                                      * (T.vals[j,i+1] - T.vals[j,i-1])  \
                                      + (Vg.vals2[j+1,i] - Vg.vals2[j-1,i])  \
                                      * (T.vals[j+1,i] - T.vals[j-1,i])

        self.vals = -1e8 * R * self.vals / (sigma(p) * pPa * 2*dx**2)
        self.vals2 = -1e8 * R * self.vals2 / (sigma(p) * pPa * 2*dx**2)
        


class QGOmega(Field):
    def __init__(self):
        Field.__init__(self, False, "qgo", r"QG $\omega$")

    def cc(self, fig, ax, params, coords):
        self.compute_vals(params)
        cmap = colormaps["BrBG_r"]
        self.contset = ax.contourf(self.x, self.y, self.vals,  \
                                       levels=range(-10,11), cmap=cmap)
        cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.6])
        cb = fig.colorbar(self.contset, cax=cbar_ax)
        cb.set_label(r"$\omega$ ($\mu$b s$^{-1}$)", labelpad=-9)

    def compute_vals(self, params):
        L = params["L"]
        ptrop = params["ptrop"]
        Tcaret = params["Tcaret"]
        p = params["p"]
        a = params["a"]
        phi0caret = params["phi0caret"]
        phase = params["phase"]

        lamb = L * phase
        alpha = 1. / np.log(1000./ptrop)
        self.vals = omegaf(self.xx, self.yy, p, L, Tcaret, alpha, a,  \
                               phi0caret, lamb)


class SandersModel(object):
    def __init__(self):
        # Initial values for the model parameters
        self.params = {"L": 3500.,  \
                           "phi0caret": 1020.,  \
                           "phase": .25,  \
                           "ptrop": 250.,  \
                           "Tcaret": 10.,  \
                           "a": .01,  \
                           "p": 500.}
        
        self.fig, self.ax = plt.subplots(figsize=(16,8))
        self.ax.set_aspect('equal')
        self.fig.subplots_adjust(bottom=0.2, left=0.1)

        self.coordinates = self._coords(self.params["L"])

        self.fields = []
        self.fields.append(RelativeVorticity())
        self.fields.append(CoriolisParameter())
        self.fields.append(AbsoluteVorticity())
        self.fields.append(GeopotentialHeight())
        self.fields.append(GeopotentialHeight1000())
        self.fields.append(Temperature())
        self.fields.append(Temperature1000())
        self.fields.append(Thickness())
        self.fields.append(GeostrophicWind())
        self.fields.append(QGOmega())
        self.fields.append(VorticityAdvection())
#        self.fields.append(DifferentialVorticityAdvection())
        self.fields.append(DifferentialVorticityAdvectionAnalytic())
        self.fields.append(TemperatureAdvection())
        self.fields.append(LapTemperatureAdvectionAnalytic())
        self.fields.append(TotalForcing())
        self.fields.append(QVector())

        self.update_plot()

    def update_plot(self):
        for f in self.fields:
            f.create_contours(self.fig, self.ax, self.params, self.coordinates)

    def on_changeL(self, Lnew):
        self.params["L"] = Lnew
        self.coordinates = self._coords(self.params["L"])
        self.ax.set_xlim(0, 2*Lnew)
        self.ax.set_ylim(0, Lnew)
        self.ax.clear()
        self.update_plot()

    def on_changephase(self, phasenew):
        self.params["phase"] = phasenew
        self.ax.clear()
        self.update_plot()
        
    def on_changea(self, anew):
        self.params["a"] = anew
        self.ax.clear()
        self.update_plot()

    def on_changeptrop(self, ptropnew):
        self.params["ptrop"] = ptropnew
        self.ax.clear()
        self.update_plot()

    def on_changeTcaret(self, Tcaretnew):
        self.params["Tcaret"] = Tcaretnew
        self.ax.clear()
        self.update_plot()

    def on_changephi0caret(self, phi0caretnew):
        self.params["phi0caret"] = phi0caretnew
        self.ax.clear()
        self.update_plot()

    def on_changep(self, pnew):
        self.params["p"] = pnew
        self.ax.clear()
        self.update_plot()

    def select_vis(self, label):
        for f in self.fields:
            if label == f.label:
                f.update_visibility(self.fig, self.ax, self.params,  \
                                        self.coordinates)
                break

        self.fig.canvas.draw()

    def _coords(self, L):
        x = np.linspace(-L, L, 101)
        y = np.linspace(-.5*L, .5*L, 51)
        xx, yy = np.meshgrid(x, y, sparse=True)
        return {"x": x, "y": y, "xx": xx, "yy": yy}

intro()

model = SandersModel()

# Length scale slider
slider_ax = plt.axes([0.1, 0.1, 0.5, 0.02])
slider1 = Slider(slider_ax, r"$L$", 2000., 4000., valinit=model.params["L"],  \
                     color='#AAAAAA')
slider1.on_changed(model.on_changeL)


# Phase shift slider
slider_ax = plt.axes([0.1, 0.07, 0.5, 0.02])
slider2 = Slider(slider_ax, r"$\lambda / L$", -.5, .5,  \
                     valinit=model.params["phase"], color='#AAAAAA')
slider2.on_changed(model.on_changephase)

# Temperature gradient slider
slider_ax = plt.axes([0.1, 0.05, 0.5, 0.01])
slider3 = Slider(slider_ax, r"$a$", 0., .02, valinit=model.params["a"],  \
                     color='#AAAAAA')
slider3.on_changed(model.on_changea)

# Tropopause pressure slider
slider_ax = plt.axes([0.1, 0.035, 0.5, 0.01])
slider4 = Slider(slider_ax, r"$p_{trop}$", 100., 450.,  \
                     valinit=model.params["ptrop"], color='#AAAAAA')
slider4.on_changed(model.on_changeptrop)

# Temperature perturbation slider
slider_ax = plt.axes([0.1, 0.02, 0.5, 0.01])
slider5 = Slider(slider_ax, r"$\hat{T}$", 0., 20.,  \
                     valinit=model.params["Tcaret"], color='#AAAAAA')
slider5.on_changed(model.on_changeTcaret)

# Geopotential perturbation slider
slider_ax = plt.axes([0.1, 0.005, 0.5, 0.01])
slider6 = Slider(slider_ax, r"$\hat{\Phi}_0$", 340., 2000.,  \
                     valinit=model.params["phi0caret"], color='#AAAAAA')
slider6.on_changed(model.on_changephi0caret)

# Pressure slider
slider_ax = plt.axes([0.1, 0.13, 0.5, 0.02])
slider7 = Slider(slider_ax, r"$p$", 100., 975.,  \
                     valinit=model.params["p"], color='#AAAAAA')
slider7.on_changed(model.on_changep)


# Visibility Buttons
check_ax = plt.axes([0.66, 0.02, 0.06, 0.15])
labels = [f.label for f in model.fields[:5]]
vis = [f.visibility for f in model.fields[:5]]
check1 = CheckButtons(check_ax, labels, vis)
check1.on_clicked(model.select_vis)

check_ax = plt.axes([0.71, 0.02, 0.1, 0.15], aspect='equal')
labels = [f.label for f in model.fields[5:10]]
vis = [f.visibility for f in model.fields[5:10]]
check2 = CheckButtons(check_ax, labels, vis)
check2.on_clicked(model.select_vis)

check_ax = plt.axes([0.79, 0.02, 0.1, 0.15], aspect='equal')
labels = [f.label for f in model.fields[10:]]
vis = [f.visibility for f in model.fields[10:]]
check3 = CheckButtons(check_ax, labels, vis)
check3.on_clicked(model.select_vis)


plt.show()
