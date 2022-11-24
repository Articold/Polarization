from . import colour_functions as cf
import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods import angular_spectrum_method, two_steps_fresnel_method

import numpy as np
from .util.backend_functions import backend as bd
from numpy.fft import fftshift,ifft2,fft2,ifftshift
from numpy.random import randn

from mpl_toolkits.mplot3d import Axes3D


class PartiallyPropagate:
    def __init__(self, wavelength, delnu, extent_x, extent_y, Nx, Ny, intensity = 0.1 * W / (m**2)):
        """
        Initializes the field, representing the cross-section profile of a plane wave

        Parameters
        ----------
        wavelength: wavelength of the plane wave
        delnu: light source band-width
        extent_x: length of the rectangular grid 
        extent_y: height of the rectangular grid 
        Nx: horizontal dimension of the grid 
        Ny: vertical dimension of the grid 
        intensity: intensity of the field
        """
        global bd
        from .util.backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.dfx = 1/extent_x
        self.fx = self.dfx*(bd.arange(Nx)-Nx//2)
        self.fx = fftshift(self.fx)
        self.FX, self.FY = bd.meshgrid(self.fx, self.fx)

        self.Nx = Nx
        self.Ny = Ny
        self.E = bd.ones((self.Ny, self.Nx)) * bd.sqrt(intensity)
        self.λ = wavelength
        self.delnu = delnu
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
    def add(self, optical_element):

        self.E = optical_element.get_E(self.E, self.xx, self.yy, self.λ)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def spatial_partially(self, z, Lcr):

        # partial spatial coherence screen parameters
        N = 50
        Lcr = Lcr
        sigma_f = 2.5*Lcr
        sigma_r = np.sqrt(4*np.pi*(sigma_f**4)/(Lcr**2))
        F = np.exp((-np.pi**2)*(sigma_f**2)*(self.FX**2 + self.FY**2))
        M = self.Nx
        #fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
        #fie=
        
        #totalfie=[[0 for j in range(512)] for i in range(512)]
        #totalfie=np.array(totalfie)
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #X = np.linspace(-0.009,0.009,512)
        #Y = np.linspace(-0.009,0.009,512)
        #X, Y = np.meshgrid(X, Y)

        
        #Z = np.array(fie())
        

        I = 0
        
        #fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
        #u_out = self.propagate_once(z, self.E * np.exp(1j*fie))
        
        for _ in range(N//2):
            fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
            #for i in range(512):
                #for j in range(512):
                    #totalfie[i,j]+=np.arctan(np.imag(fie[i,j])/np.real(fie[i,j]))
            u_out = self.propagate_once(z, self.E * np.exp(1j*np.real(fie)))
            I += np.abs(u_out) ** 2
            u_out = self.propagate_once(z, self.E * np.exp(1j*np.imag(fie)))
            I += np.abs(u_out) ** 2
        #surf = ax.plot_surface(X, Y, fie,linewidth=0, antialiased=False)
        
        #plt.show()               
        return I 
            
    def spatial_partially_nlos(self, d1, d2, Lcr, phi_org):

        # partial spatial coherence screen parameters
        N = 50
        Lcr = Lcr
        sigma_f = 2.5*Lcr
        sigma_r = np.sqrt(4*np.pi*(sigma_f**4)/(Lcr**2))
        F = np.exp((-np.pi**2)*(sigma_f**2)*(self.FX**2 + self.FY**2))
        M = self.Nx

        I = 0
        
        #fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
        #u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*fie), phi_org)
        for _ in range(N//2):
            fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
            u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*np.real(fie)), phi_org)
            I += np.abs(u_out) ** 2
            
            u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*np.imag(fie)), phi_org)
            I += np.abs(u_out) ** 2
        
        return I 


    def propagate_partially(self, z, deld):
        """
        partially coherent light source simulation,
        z: propagate distance,
        delnu: Guassian Light half-band frequency.
        
        """
        c = 3e8
        k0 = 2 * np.pi / self.λ
        nu0 = c / self.λ

        ## Gaussian lineshape parameters
        N = 51      # sample resolution
        b = self.delnu / (2 * np.sqrt(np.log(2)))
        dnu = 4 * self.delnu / N

        ## Start to simulate
        I = 0
        for n in range(1, N+1):
            # spectural density function
            nu = (n - (N + 1) / 2) * dnu + nu0
            S = 1/(np.sqrt(np.pi)*b) * np.exp(-(nu-nu0)**2 / (b**2))
            k = (2 * np.pi * nu) / c
            λ = 2 * np.pi / k
            if S*dnu > 0.04:
                # I += S*dnu*(self.propagate_singlewv(z, λ))
                u = self.propagate_singlewv(z, λ)
                u_move = self.propagate_singlewv(z+deld, λ)
                u = 0.5 * (u + u_move)
                current_I = bd.real(u * bd.conjugate(u)) 
                I += S*dnu*current_I
                # pass
            # print(n)

        return I


    def propagate_singlewv(self, z, λ):
        """compute the field in distance equal to z with the angular spectrum method"""

        # self.z = z
        E = angular_spectrum_method(self, self.E, z, λ)

        # compute Field Intensity
        # I = bd.real(E * bd.conjugate(E))  
        return E

    def propagate_once(self, z, u_in):
        """same as below, but don't record propgation distance."""
        E = angular_spectrum_method(self, u_in, z, self.λ)

        # I = bd.real(E * bd.concatenate(E))
        return E

    def diffuser_prop(self, d1, d2, u_in, phi_org):
        
        E = angular_spectrum_method(self, u_in, d1, self.λ)
        # phi_org = (np.random.rand(E.shape[0],E.shape[0]) - 0.5) * 2 * 3.1415926
        E = E * np.exp(1j * phi_org)
        E = angular_spectrum_method(self,  E, d2, self.λ)

        return E

    def diffuser_prop_singlewv(self, d1, d2, phi_org):
        
        E = angular_spectrum_method(self, self.E, d1, self.λ)
        E = E * np.exp(1j * phi_org)
        E = angular_spectrum_method(self, E, d2, self.λ)

        return E

    def propagate(self, z):
        """compute the field in distance equal to z with the angular spectrum method"""

        self.z += z
        self.E = angular_spectrum_method(self, self.E, z, self.λ)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def propagate_incoherent(self, z, deld):
        """
        We simulate the incoherent light as the sums of 380mm ~ 779mm
        coherent light
        """
        I = 0
        count = 0
        for lambda0 in range(380, 780):
            u = self.propagate_singlewv(z, lambda0 * nm)
            u_move = self.propagate_singlewv(z+deld, lambda0 * nm)
            u = 0.5 * (u + u_move)
            I += bd.real(u * bd.conjugate(u))  
            count += 1
        I = I / count
        # I = angular_spectrum_method_incoherent(self, self.E, z, self.λ)
        # lambda0 = 650 * nm
        # k = 2 * np.pi / lambda0
        # u_in = self.E + self.E * np.exp(1j * k * deld)
        # u_out = angular_spectrum_method(self, u_in, z, lambda0)
        # I_out = bd.real(u_out * bd.conjugate(u_out)) 
        return I


    def scale_propagate(self, z, scale_factor):
        """
        Compute the field in distance equal to z with the two step Fresnel propagator, rescaling the field in the new coordinates
        with extent equal to:
        new_extent_x = scale_factor * self.extent_x
        new_extent_y = scale_factor * self.extent_y

        Note that unlike within in the propagate method, Fresnel approximation is used here.
        Reference: VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.
        """
        
        self.z += z
        self.E = two_steps_fresnel_method(self, self.E, z, self.λ, scale_factor)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  


    def get_colors(self):
        """ compute RGB colors"""

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * self.I.flatten()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb


    def get_longitudinal_profile(self, start_distance, end_distance, steps):
        """
        Propagates the field at n steps equally spaced between start_distance and end_distance, and returns
        the colors and the field over the xz plane
        """

        z = bd.linspace(start_distance, end_distance, steps)

        self.E0 = self.E.copy()

        longitudinal_profile_rgb = bd.zeros((steps,self.Nx, 3))
        longitudinal_profile_E = bd.zeros((steps,self.Nx), dtype = complex)
        z0 = self.z 
        t0 = time.time()

        bar = progressbar.ProgressBar()
        for i in bar(range(steps)):
                 
            self.propagate(z[i])
            rgb = self.get_colors()
            longitudinal_profile_rgb[i,:,:]  = rgb[self.Ny//2,:,:]
            longitudinal_profile_E[i,:] = self.E[self.Ny//2,:]
            self.E = np.copy(self.E0)

        # restore intial values
        self.z = z0
        self.I = bd.real(self.E * bd.conjugate(self.E))  

        print ("Took", time.time() - t0)

        return longitudinal_profile_rgb, longitudinal_profile_E

class Partiallycorrelation:
    def __init__(self, wavelength, delnu, extent_x, extent_y, Nx, Ny, intensity = 0.1 * W / (m**2)):
        """
        Initializes the field, representing the cross-section profile of a plane wave

        Parameters
        ----------
        wavelength: wavelength of the plane wave
        delnu: light source band-width
        extent_x: length of the rectangular grid 
        extent_y: height of the rectangular grid 
        Nx: horizontal dimension of the grid 
        Ny: vertical dimension of the grid 
        intensity: intensity of the field
        """
        global bd
        from .util.backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.dfx = 1/extent_x
        self.fx = self.dfx*(bd.arange(Nx)-Nx//2)
        self.fx = fftshift(self.fx)
        self.FX, self.FY = bd.meshgrid(self.fx, self.fx)

        self.Nx = Nx
        self.Ny = Ny
        self.E = bd.ones((self.Ny, self.Nx)) * bd.sqrt(intensity)
        self.λ = wavelength
        self.delnu = delnu
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
    def add(self, optical_element):

        self.E = optical_element.get_E(self.E, self.xx, self.yy, self.λ)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def spatial_partially(self, z, Lcr):

        # partial spatial coherence screen parameters
        N = 50
        Lcr = Lcr
        sigma_f = 2.5*Lcr
        sigma_r = np.sqrt(4*np.pi*(sigma_f**4)/(Lcr**2))
        F = np.exp((-np.pi**2)*(sigma_f**2)*(self.FX**2 + self.FY**2))
        M = self.Nx
        # fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)

        I = 0
        cor=0

        #fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
        #u_out = self.propagate_once(z, self.E * np.exp(1j*fie))
        for _ in range(N//2):
            fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
            u_out = self.propagate_once(z, self.E * np.exp(1j*fie))


            cor+= np.abs(u_out[256][256]* (u_out.conjugate()))
                    
            #u_out = self.propagate_once(z, self.E * np.exp(1j*np.real(fie)))
            #I += np.abs(u_out) ** 2
            #u_out = self.propagate_once(z, self.E * np.exp(1j*np.imag(fie)))
            #I += np.abs(u_out) ** 2
        
        return cor
            
    def spatial_partially_nlos(self, d1, d2, Lcr, phi_org):

        # partial spatial coherence screen parameters
        N = 50
        Lcr = Lcr
        sigma_f = 2.5*Lcr
        sigma_r = np.sqrt(4*np.pi*(sigma_f**4)/(Lcr**2))
        F = np.exp((-np.pi**2)*(sigma_f**2)*(self.FX**2 + self.FY**2))
        M = self.Nx

        I = 0
        cor=0
        #fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
        #u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*fie), phi_org)
        for _ in range(N//2):
            fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
            u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*fie), phi_org)
            cor+= np.abs(u_out[256][256]* (u_out.conjugate()))
            #I += np.abs(u_out) ** 2
            
            #u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*np.imag(fie)), phi_org)
            #I += np.abs(u_out) ** 2
        
        return cor


    def propagate_partially(self, z, deld):
        """
        partially coherent light source simulation,
        z: propagate distance,
        delnu: Guassian Light half-band frequency.
        
        """
        c = 3e8
        k0 = 2 * np.pi / self.λ
        nu0 = c / self.λ

        ## Gaussian lineshape parameters
        N = 51      # sample resolution
        b = self.delnu / (2 * np.sqrt(np.log(2)))
        dnu = 4 * self.delnu / N

        ## Start to simulate
        I = 0
        for n in range(1, N+1):
            # spectural density function
            nu = (n - (N + 1) / 2) * dnu + nu0
            S = 1/(np.sqrt(np.pi)*b) * np.exp(-(nu-nu0)**2 / (b**2))
            k = (2 * np.pi * nu) / c
            λ = 2 * np.pi / k
            if S*dnu > 0.04:
                # I += S*dnu*(self.propagate_singlewv(z, λ))
                u = self.propagate_singlewv(z, λ)
                u_move = self.propagate_singlewv(z+deld, λ)
                u = 0.5 * (u + u_move)
                current_I = bd.real(u * bd.conjugate(u)) 
                I += S*dnu*current_I
                # pass
            # print(n)

        return I


    def propagate_singlewv(self, z, λ):
        """compute the field in distance equal to z with the angular spectrum method"""

        # self.z = z
        E = angular_spectrum_method(self, self.E, z, λ)

        # compute Field Intensity
        # I = bd.real(E * bd.conjugate(E))  
        return E

    def propagate_once(self, z, u_in):
        """same as below, but don't record propgation distance."""
        E = angular_spectrum_method(self, u_in, z, self.λ)

        # I = bd.real(E * bd.concatenate(E))
        return E

    def diffuser_prop(self, d1, d2, u_in, phi_org):
        
        E = angular_spectrum_method(self, u_in, d1, self.λ)
        # phi_org = (np.random.rand(E.shape[0],E.shape[0]) - 0.5) * 2 * 3.1415926
        E = E * np.exp(1j * phi_org)
        E = angular_spectrum_method(self,  E, d2, self.λ)

        return E

    def diffuser_prop_singlewv(self, d1, d2, phi_org):
        
        E = angular_spectrum_method(self, self.E, d1, self.λ)
        E = E * np.exp(1j * phi_org)
        E = angular_spectrum_method(self, E, d2, self.λ)

        return E

    def propagate(self, z):
        """compute the field in distance equal to z with the angular spectrum method"""

        self.z += z
        self.E = angular_spectrum_method(self, self.E, z, self.λ)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def propagate_incoherent(self, z, deld):
        """
        We simulate the incoherent light as the sums of 380mm ~ 779mm
        coherent light
        """
        I = 0
        count = 0
        for lambda0 in range(380, 780):
            u = self.propagate_singlewv(z, lambda0 * nm)
            u_move = self.propagate_singlewv(z+deld, lambda0 * nm)
            u = 0.5 * (u + u_move)
            I += bd.real(u * bd.conjugate(u))  
            count += 1
        I = I / count
        # I = angular_spectrum_method_incoherent(self, self.E, z, self.λ)
        # lambda0 = 650 * nm
        # k = 2 * np.pi / lambda0
        # u_in = self.E + self.E * np.exp(1j * k * deld)
        # u_out = angular_spectrum_method(self, u_in, z, lambda0)
        # I_out = bd.real(u_out * bd.conjugate(u_out)) 
        return I


    def scale_propagate(self, z, scale_factor):
        """
        Compute the field in distance equal to z with the two step Fresnel propagator, rescaling the field in the new coordinates
        with extent equal to:
        new_extent_x = scale_factor * self.extent_x
        new_extent_y = scale_factor * self.extent_y

        Note that unlike within in the propagate method, Fresnel approximation is used here.
        Reference: VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.
        """
        
        self.z += z
        self.E = two_steps_fresnel_method(self, self.E, z, self.λ, scale_factor)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  


    def get_colors(self):
        """ compute RGB colors"""

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * self.I.flatten()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb


    def get_longitudinal_profile(self, start_distance, end_distance, steps):
        """
        Propagates the field at n steps equally spaced between start_distance and end_distance, and returns
        the colors and the field over the xz plane
        """

        z = bd.linspace(start_distance, end_distance, steps)

        self.E0 = self.E.copy()

        longitudinal_profile_rgb = bd.zeros((steps,self.Nx, 3))
        longitudinal_profile_E = bd.zeros((steps,self.Nx), dtype = complex)
        z0 = self.z 
        t0 = time.time()

        bar = progressbar.ProgressBar()
        for i in bar(range(steps)):
                 
            self.propagate(z[i])
            rgb = self.get_colors()
            longitudinal_profile_rgb[i,:,:]  = rgb[self.Ny//2,:,:]
            longitudinal_profile_E[i,:] = self.E[self.Ny//2,:]
            self.E = np.copy(self.E0)

        # restore intial values
        self.z = z0
        self.I = bd.real(self.E * bd.conjugate(self.E))  

        print ("Took", time.time() - t0)

        return longitudinal_profile_rgb, longitudinal_profile_E

class Partiallyinterfere:
    def __init__(self, wavelength, delnu, extent_x, extent_y, Nx, Ny, intensity = 0.1 * W / (m**2)):
        """
        Initializes the field, representing the cross-section profile of a plane wave

        Parameters
        ----------
        wavelength: wavelength of the plane wave
        delnu: light source band-width
        extent_x: length of the rectangular grid 
        extent_y: height of the rectangular grid 
        Nx: horizontal dimension of the grid 
        Ny: vertical dimension of the grid 
        intensity: intensity of the field
        """
        global bd
        from .util.backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.dfx = 1/extent_x
        self.fx = self.dfx*(bd.arange(Nx)-Nx//2)
        self.fx = fftshift(self.fx)
        self.FX, self.FY = bd.meshgrid(self.fx, self.fx)

        self.Nx = Nx
        self.Ny = Ny
        self.E = bd.ones((self.Ny, self.Nx)) * bd.sqrt(intensity)
        self.λ = wavelength
        self.delnu = delnu
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
    def add(self, optical_element):

        self.E = optical_element.get_E(self.E, self.xx, self.yy, self.λ)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  
    def circ(self,r):
        #dim = [256,256]
        #centerCoord = [128,128]
        #a=r
        
        #xSize = dim[1]
        #ySize = dim[0]
        #x1 = centerCoord[1]
        #y1 = centerCoord[0]
        #[x, y] = bd.meshgrid(range(-(x1 - 0),(xSize - x1)),range(-(y1 - 0),(ySize - y1)))
        #A = (((x/a)**2 + (y/a)**2) <= 1)
        for i in range(r.shape[0]):
            for j in range(r.shape[0]):
                if r[i][j]<=1:
                   r[i][j]=1
                else:
                   r[i][j]=0
        return r
        
        
        
    def spatial_partially(self, z, Lcr):
        w=1e-3
        dels=5e-3
        f=0.25
        lf=self.λ*f
        u1=self.circ(np.sqrt((self.xx-dels/2)**2+self.yy**2)/w)+self.circ(np.sqrt((self.xx+dels/2)**2+self.yy**2)/w)
        #plt.imshow(u1)
        # partial spatial coherence screen parameters
        N = 100
        Lcr = Lcr
        sigma_f = 2.5*Lcr
        sigma_r = np.sqrt(4*np.pi*(sigma_f**4)/(Lcr**2))
        F = np.exp((-np.pi**2)*(sigma_f**2)*(self.FX**2 + self.FY**2))
        
        M = self.Nx
        # fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)

        I = 0
        cor=0

        #fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
        #u_out = self.propagate_once(z, self.E * np.exp(1j*fie))
        for _ in range(N//2):
            fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
            u_out = 1/lf*fft2(u1 * np.exp(1j*np.real(fie)))*self.dx*self.dx
            I += np.abs(u_out) ** 2
            
            u_out = 1/lf*fft2(u1 * np.exp(1j*np.real(fie)))*self.dx*self.dx
            I += np.abs(u_out) ** 2
                    
            #u_out = self.propagate_once(z, self.E * np.exp(1j*np.real(fie)))
            #I += np.abs(u_out) ** 2
            #u_out = self.propagate_once(z, self.E * np.exp(1j*np.imag(fie)))
            #I += np.abs(u_out) ** 2
        I=ifftshift(I)/N
        #plt.imshow(I)
        return I
            
    def spatial_partially_nlos(self, d1, d2, Lcr, phi_org):

        # partial spatial coherence screen parameters
        N = 50
        Lcr = Lcr
        sigma_f = 2.5*Lcr
        sigma_r = np.sqrt(4*np.pi*(sigma_f**4)/(Lcr**2))
        F = np.exp((-np.pi**2)*(sigma_f**2)*(self.FX**2 + self.FY**2))
        M = self.Nx

        I = 0
        cor=0
        #fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
        #u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*fie), phi_org)
        for _ in range(N//2):
            fie = (ifft2(F*(randn(M,M)+1j*randn(M,M)))*sigma_r/self.dfx)*(M**2) * (self.dfx**2)
            u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*fie), phi_org)
            cor+= np.abs(u_out[256][256]* (u_out.conjugate()))
            #I += np.abs(u_out) ** 2
            
            #u_out = self.diffuser_prop(d1, d2, self.E * np.exp(1j*np.imag(fie)), phi_org)
            #I += np.abs(u_out) ** 2
        
        return cor


    def propagate_partially(self, z, deld):
        """
        partially coherent light source simulation,
        z: propagate distance,
        delnu: Guassian Light half-band frequency.
        
        """
        c = 3e8
        k0 = 2 * np.pi / self.λ
        nu0 = c / self.λ

        ## Gaussian lineshape parameters
        N = 51      # sample resolution
        b = self.delnu / (2 * np.sqrt(np.log(2)))
        dnu = 4 * self.delnu / N

        ## Start to simulate
        I = 0
        for n in range(1, N+1):
            # spectural density function
            nu = (n - (N + 1) / 2) * dnu + nu0
            S = 1/(np.sqrt(np.pi)*b) * np.exp(-(nu-nu0)**2 / (b**2))
            k = (2 * np.pi * nu) / c
            λ = 2 * np.pi / k
            if S*dnu > 0.04:
                # I += S*dnu*(self.propagate_singlewv(z, λ))
                u = self.propagate_singlewv(z, λ)
                u_move = self.propagate_singlewv(z+deld, λ)
                u = 0.5 * (u + u_move)
                current_I = bd.real(u * bd.conjugate(u)) 
                I += S*dnu*current_I
                # pass
            # print(n)

        return I


    def propagate_singlewv(self, z, λ):
        """compute the field in distance equal to z with the angular spectrum method"""

        # self.z = z
        E = angular_spectrum_method(self, self.E, z, λ)

        # compute Field Intensity
        # I = bd.real(E * bd.conjugate(E))  
        return E

    def propagate_once(self, z, u_in):
        """same as below, but don't record propgation distance."""
        E = angular_spectrum_method(self, u_in, z, self.λ)

        # I = bd.real(E * bd.concatenate(E))
        return E

    def diffuser_prop(self, d1, d2, u_in, phi_org):
        
        E = angular_spectrum_method(self, u_in, d1, self.λ)
        # phi_org = (np.random.rand(E.shape[0],E.shape[0]) - 0.5) * 2 * 3.1415926
        E = E * np.exp(1j * phi_org)
        E = angular_spectrum_method(self,  E, d2, self.λ)

        return E

    def diffuser_prop_singlewv(self, d1, d2, phi_org):
        
        E = angular_spectrum_method(self, self.E, d1, self.λ)
        E = E * np.exp(1j * phi_org)
        E = angular_spectrum_method(self, E, d2, self.λ)

        return E

    def propagate(self, z):
        """compute the field in distance equal to z with the angular spectrum method"""

        self.z += z
        self.E = angular_spectrum_method(self, self.E, z, self.λ)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def propagate_incoherent(self, z, deld):
        """
        We simulate the incoherent light as the sums of 380mm ~ 779mm
        coherent light
        """
        I = 0
        count = 0
        for lambda0 in range(380, 780):
            u = self.propagate_singlewv(z, lambda0 * nm)
            u_move = self.propagate_singlewv(z+deld, lambda0 * nm)
            u = 0.5 * (u + u_move)
            I += bd.real(u * bd.conjugate(u))  
            count += 1
        I = I / count
        # I = angular_spectrum_method_incoherent(self, self.E, z, self.λ)
        # lambda0 = 650 * nm
        # k = 2 * np.pi / lambda0
        # u_in = self.E + self.E * np.exp(1j * k * deld)
        # u_out = angular_spectrum_method(self, u_in, z, lambda0)
        # I_out = bd.real(u_out * bd.conjugate(u_out)) 
        return I


    def scale_propagate(self, z, scale_factor):
        """
        Compute the field in distance equal to z with the two step Fresnel propagator, rescaling the field in the new coordinates
        with extent equal to:
        new_extent_x = scale_factor * self.extent_x
        new_extent_y = scale_factor * self.extent_y

        Note that unlike within in the propagate method, Fresnel approximation is used here.
        Reference: VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.
        """
        
        self.z += z
        self.E = two_steps_fresnel_method(self, self.E, z, self.λ, scale_factor)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  


    def get_colors(self):
        """ compute RGB colors"""

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * self.I.flatten()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb


    def get_longitudinal_profile(self, start_distance, end_distance, steps):
        """
        Propagates the field at n steps equally spaced between start_distance and end_distance, and returns
        the colors and the field over the xz plane
        """

        z = bd.linspace(start_distance, end_distance, steps)

        self.E0 = self.E.copy()

        longitudinal_profile_rgb = bd.zeros((steps,self.Nx, 3))
        longitudinal_profile_E = bd.zeros((steps,self.Nx), dtype = complex)
        z0 = self.z 
        t0 = time.time()

        bar = progressbar.ProgressBar()
        for i in bar(range(steps)):
                 
            self.propagate(z[i])
            rgb = self.get_colors()
            longitudinal_profile_rgb[i,:,:]  = rgb[self.Ny//2,:,:]
            longitudinal_profile_E[i,:] = self.E[self.Ny//2,:]
            self.E = np.copy(self.E0)

        # restore intial values
        self.z = z0
        self.I = bd.real(self.E * bd.conjugate(self.E))  

        print ("Took", time.time() - t0)

        return longitudinal_profile_rgb, longitudinal_profile_E

    from .visualization import plot_colors, plot_intensity, plot_longitudinal_profile_colors, plot_longitudinal_profile_intensity
