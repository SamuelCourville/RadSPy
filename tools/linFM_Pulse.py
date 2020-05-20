######################################################################
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys


###############################################################################
###############################################################################
#                       Pulse Parameters - Modify these
###############################################################################
###############################################################################

fileString = "SHARAD3_Pulse_ideal"

## Source Pulse Parameters
f = 20e6 #(300e6-150e6)/2+150e6 #Hz
bw = 10e6 #(300e6-150e6)
pl = 85e-6 #1e-3
A = 1.0
tc = -f/(bw/pl)

# Discretization parameters
fs = 45e6 #4*f+bw; # nyquist sampling
nt = round(pl*fs)
maxt = pl/2
mint = -pl/2
t = np.linspace(mint,maxt,nt)
dt = t[2]-t[1]



###############################################################################
###############################################################################
#                       Functions - Do not modify these
###############################################################################
###############################################################################

# Create Box function
def rect(t,w,A):
    nt = len(t)
    g = np.zeros((nt))
    for i in range(0,nt):
        if t[i]>-w and t[i] < w:
            g[i] = A
        else:
            g[i] = 0
    return g

# Create Linear FM pulse
def linFM_Pulse(bw,pl,t,A,tc):
    K = bw/pl
    g = rect(t,pl/2.0,A)*np.exp(1j*(np.pi*K*(t-tc)**2))
    return(g)

def linFM_Pulse_filter(bw,pl,t,A,tc):
    K = bw/pl
    g = rect(t,pl/2.0,A)*np.exp(-1j*(np.pi*K*(t+tc)**2))
    return(g)

def main():
    nt = len(t)
    s = linFM_Pulse(bw,pl,t,A,tc)
    h = linFM_Pulse_filter(bw,pl,t,A,tc)

    H = np.fft.fftshift(np.fft.fft(h))/(nt)
    Hfreq = np.fft.fftshift(np.fft.fftfreq(nt,dt))
    
    plt.plot(Hfreq,H)

    a = np.transpose(np.asarray([t, s]))
    np.savetxt(fileString + "_sourcePulse.csv", a, delimiter=",")

    a = np.transpose(np.asarray([t, Hfreq, H]))
    np.savetxt(fileString + "_matchedFilter.csv", a, delimiter=",")
    return

main()
