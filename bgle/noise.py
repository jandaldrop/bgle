import numpy as np

def ft(f,t):
    w=2.*np.pi*np.fft.fftfreq(len(f))/(t[1]-t[0])
    g=np.fft.fft(f)
    g*=(t[1]-t[0])*np.exp(-complex(0,1)*w*t[0])
    return g,w

def ift(f,w,t):
    f*=np.exp(complex(0,1)*w*t[0])
    g=np.fft.ifft(f)
    g*=len(f)*(w[1]-w[0])/(2*np.pi)
    return g

class ColoredNoiseGenerator:
    """
    A class for the generation of colored noise.
    """
    def __init__(self, kernel, t, add_zeros=0, rng=np.random.normal):
        """
        Create an instance of the ColoredNoiseGenerator class.

        Parameters
        ----------
        kernel : numpy.array
            The correlation function of the noise.
        t : numpy.array
            The time/x values of kernel.
        add_zeros : int, default=0
            Add add_zeros number of zeros to the kernel function for numeric Fourier transformation.
        """

        self.kernel=np.concatenate([kernel,np.zeros(add_zeros)])
        self.t=t
        self.rng=rng

        kernel_sym=np.concatenate((self.kernel[:0:-1],self.kernel))
        t_sym=np.concatenate((-self.t[:0:-1],self.t))

        kernel_ft,w=ft(kernel_sym,t_sym)
        sqk_ft=np.sqrt(kernel_ft)
        self.sqk=ift(sqk_ft,w,t_sym).real

    def generate(self, size):
        white_noise=self.rng(size=size)
        colored_noise=np.convolve(white_noise,self.sqk,mode='same')*np.sqrt(self.t[1]-self.t[0])
        return colored_noise
