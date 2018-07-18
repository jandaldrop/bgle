import numpy as np

from .noise import ColoredNoiseGenerator

__all__=['noise']

class BGLEIntegrator:
    """
    The Class holding the bGLE integrator.
    The current implementation is in python, and therefore not very fast.
    """
    def __init__(self,kernel, t, m=1., dU=lambda x: 0., add_zeros=0, verbose=True):
        self.kernel=kernel
        self.t=t
        self.m=m
        self.dt=self.t[1]-self.t[0]
        self.verbose=verbose
        self.dU=dU

        if self.verbose:
            print("Found dt =", self.dt)

        self.noise_generator=ColoredNoiseGenerator(self.kernel,self.t,add_zeros=add_zeros)

    def mem_int_red(self, v):
        if len(v) < len(self.kernel):
            v=np.concatenate([np.zeros(len(self.kernel)-len(v)+1),v])
        integrand=self.kernel[1:]*v[:len(v)-len(self.kernel[1:])-1:-1]
        return (0.5*integrand[-1]+np.sum(integrand[:-1]))*self.dt

    def f_rk(self,x,v,rmi,fr,next_w,last_w,last_v,last_rmi):
        nv=v
        na=(-next_w*rmi -last_w*last_rmi -0.5*next_w*self.kernel[0]*v*self.dt-0.5*last_w*self.kernel[0]*last_v*self.dt  -self.dU(x)+fr)/self.m
        return nv, na

    def rk_step(self,x,v,rmi,fr,last_v,last_rmi):
        k1x,k1v=self.f_rk(x,v,rmi,fr,0.0,1.0,last_v,last_rmi)
        k2x,k2v=self.f_rk(x+k1x*self.dt/2,v+k1v*self.dt/2,rmi,fr,0.5,0.5,last_v,last_rmi)
        k3x,k3v=self.f_rk(x+k2x*self.dt/2,v+k2v*self.dt/2,rmi,fr,0.5,0.5,last_v,last_rmi)
        k4x,k4v=self.f_rk(x+k3x*self.dt,  v+k3v*self.dt,rmi,fr,1.0,0.0,last_v,last_rmi)
        return x + self.dt*(k1x+2.*k2x+2.*k3x+k4x)/6., v + self.dt*(k1v+2.*k2v+2.*k3v+k4v)/6.

    def integrate(self,n_steps,x0=0.,v0=0., set_noise_to_zero=False):
        if set_noise_to_zero:
            noise=np.zeros(n_steps)
        else:
            noise=self.noise_generator.generate(n_steps)
            
        x,v=x0,v0

        self.v_trj=np.zeros(n_steps)
        self.x_trj=np.zeros(n_steps)
        self.t_trj=np.arange(0.,n_steps*self.dt,self.dt)

        rmi=0.
        for ind in range(n_steps):
            last_rmi=rmi
            if n_steps>1:
                rmi=self.mem_int_red(self.v_trj[:ind])
                last_v=self.v_trj[ind-1]
            else:
                rmi=0.
                last_rmi=0.
                last_v=0.

            x,v=self.rk_step(x,v,rmi,noise[ind],last_v,last_rmi)
            self.v_trj[ind]=v
            self.x_trj[ind]=x

        return self.x_trj, self.v_trj, self.t_trj


# ---
