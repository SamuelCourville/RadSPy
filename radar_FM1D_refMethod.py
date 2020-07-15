###############################################################################
###############################################################################
#                       1D Radar simulator
###############################################################################
###############################################################################


###############################################################################
#                       import statements and constants
###############################################################################

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import csv
import sys

c = 2.99792*10**8     # speed of light [m/s]
eps0 = 8.854188e-12   # vacuum permitivity [F/m or something like that]   
mu = 4*np.pi*10**-7
    
    
###############################################################################
#                       Global Variables
###############################################################################
H = 0
dB_noise = 0
h = np.array([0])
eps = np.array([1.0])
lossTangent = np.array([0.0])
nl=0
useConductivity = 0
windowNum = 0
fmin = 0
fmax = 0
pulseFile = ""
filterFile = ""

###############################################################################
#                       Flags
###############################################################################
timeFlag = False
plotResult = True

###############################################################################
#                       Main simulation function
###############################################################################
def freqComponent(omega,a):
    '''
        Usage:
          This function implements the math from Wait (1970) to solve for the 
	response from the reflection of a monochromatic electromagnetic wave in
	a horizontally stratified medium. This solves the reflection for one 
	frequency component of the radar pulse.

        Input Variables:
             omega -- the angular frequency component of the radar pulse.
             a -- the amplitude of the frequency component in the source pulse.

        Output Variables:
             b -- the amplitude of the reflected frequency component.

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    
    if not useConductivity:
        sigma = abs(omega)*eps*lossTangent
    else:
        sigma = lossTangent
    
    M = int(nl)
    K = np.zeros((M+1),dtype=complex)
    Z = np.zeros((M),dtype=complex)
    u = np.zeros((M+1),dtype=complex)
    gamma2 = np.zeros((M+1),dtype=complex)
    
    if omega==0:
        omega = 0.000000001

    lamb=0
    for i in range(0,M+1):
        gamma2[i] = 1j*sigma[i]*mu*omega-eps[i]*mu*omega**2
        u[i] = np.sqrt(lamb**2+gamma2[i])
        K[i] = u[i]/(sigma[i]+1j*omega*eps[i])

        
    if M>1:
        indexes = np.flipud(np.linspace(1,M-2,M-2,dtype=int))
        
        
        Z[-1] = K[M-1]*(K[M]+K[M-1]*np.tanh(u[M-1]*h[M-1]))/(K[M-1]+K[M]*np.tanh(u[M-1]*h[M-1]))
        for i in indexes:
            Z[i] = K[i]*(Z[i+1]+K[i]*np.tanh(u[i]*h[i]))/(K[i]+Z[i+1]*np.tanh(u[i]*h[i]))
    else:
        Z = np.zeros((M+1),dtype=complex)
        Z[1] = K[M-1]*(K[M]+K[M-1]*np.tanh(u[M-1]*h[M-1]))/(K[M-1]+K[M]*np.tanh(u[M-1]*h[M-1]))
    
    a = a/Z[1]
    b = a*(K[0]-Z[1])/(K[0]+Z[1])
    b = Z[1]*b
    return b


###############################################################################
#                       File loading functions
###############################################################################
def loadModel(fileName):
    '''
        Usage:
          This function loads a model file. The model file should contain all 
	of the global variables needed to run a successful simulation. See the 
	example model text file in the github repo for an example model file.

        Input Variables:
             fileName -- the name of the text file which includes the simulation
	parameters.

        Global variables updated:
           The model text file should include the following global variables:
		H - the altitude (height) of the spacecraft above the ground (m)
		dB_noise - if greater than -101 dB, the code will add random 
			gaussian noise to the synthetic radar return at the
			level denoted by dB_noise. 
		h - an array which represents the thicknesses (m) of each layer
		eps - array, the relative real dielectric constant of each layer
		lossTangent - an array of loss tangent values for the layers. 
			if useConductivity is set to 1, then this variable 
			should be set to conductivity values (S/m).
		nl - the number of layers in the model
		useConductivity - 1 or 0. See lossTangent variable description.
		windowNum - The Kaiser window degree to be applied on the output
			data. a 0 will apply no window.
		fmin - The low bound on the Kaiser frequency window (Hz). 
			use 15e6 for SHARAD.
		fmax - The high bound on the kaiser frequency window (Hz).
			use 20e6 for SHARAD.
		pulseFile - a filename and path to a csv file with the source 
			pulse. See example modelFile and corresponding pulse.
		filterFile - a filename and path to a csv file with the 
			matched filter. See example. 
		

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    global H
    global dB_noise
    global h
    global eps
    global lossTangent
    global nl
    global useConductivity
    global windowNum
    global fmin
    global fmax
    global pulseFile
    global filterFile
    with open(fileName) as f:
        junk = f.readline()
        junk = f.readline()
        H = float(f.readline().strip())                # Spacecraft Altitude (m)
        junk = f.readline()
        dB_noise = float((f.readline().strip()))       # What's the noise level in dB?
        dB_noise = dB_noise-100/dB_noise               # This is an approximation. Ask me for expln. 
        junk = f.readline()
        nl = int(float((f.readline().strip())))        # number of layers
        junk = f.readline()
        depthsstr = (f.readline().strip())
        depthsstr = depthsstr.split(' ')
        depths = [float(i) for i in depthsstr]         # Layer Thicknesses (m)
        junk = f.readline()
        epsrstr = (f.readline().strip())
        epsrstr = epsrstr.split(' ')
        epsr = [float(i) for i in epsrstr]             # real relative permitivity
        junk = f.readline()
        useConductivity = float((f.readline().strip()))# number of layers
        junk = f.readline()
        lossTangentstr = (f.readline().strip())        # loss tangent or conductivity
        lossTangentstr = lossTangentstr.split(' ')
        lossTangentPre = [float(i) for i in lossTangentstr]
        junk = f.readline()
        pulseFile = f.readline().strip()
        junk = f.readline()
        filterFile = f.readline().strip()
        junk = f.readline()
        windowNum = float((f.readline().strip()))
        junk = f.readline()
        windowFreqs = (f.readline().strip()) 
        windowFreqs = windowFreqs.split(' ')
        windowFreqs = [float(i) for i in windowFreqs]
        
        fmin = windowFreqs[0]
        fmax = windowFreqs[1]
        
        epsr = np.array(epsr)
        h = np.append(h, np.array(depths))
        eps = np.append(eps,epsr)*8.85e-12
        lossTangent = np.append(lossTangent,np.array(lossTangentPre))
        return
    
    
def loadTruePulse(fileName):
    '''
        Usage:
          This function loads a pulse file.

        Input Variables:
             fileName -- this should be a string with the name and path to a 
		file containing the desired source pulse to simulate. The CSV			file should contain two columns, time and amplitude. 

        Output:
	     The output is two arrays, E, which is a time series of the 
	Electric field amplitude from the input file, and t, which is an 
	array of the corresponding time values for the time series E.

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    with open(fileName) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        t = []
        E = []
        for row in readCSV:
            t.append(np.real(complex(row[0])))
            E.append(complex(row[1].replace('+-', '-', 1)))
    return t,E    
    
def loadMatchFilter(fileName):
    '''
        Usage:
          This function loads the matched filter

        Input Variables:
             fileName -- this should be a string with the name and path to a 
		file containing the desired matched filter to compress with 
		in the frequency domain. The CSV file should contain three 
		columns, the time series value for the corresponding pulse, 
		the frequency axis for the filter, and the amplitude spectrum 
		of the matched filter.

        Output:
	     The three colums of the input csv files as arrays.

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    with open(fileName) as csvfile:
        readCSV = csv.reader(csvfile,delimiter=',')
        t = []
        f = []
        E = []
        for row in readCSV:
            t.append(np.real(complex(row[0])))
            f.append(np.real(complex(row[1])))
            E.append(complex(row[2].replace('+-', '-', 1)))
    return t,f,E

###############################################################################
#                       Range Compression and windowing
###############################################################################
def rangeCompress(data, mfilter, f):
    '''
        Usage:
          This function range compresses the resulting reflected electromagnetic          field with the given matched filter. 

        Input Variables:
             data - the frequency spectrum of the refelcted E-field
	     mfilter - the matched filter to compress data with in the f-domain
	     f - the frequency axis values for data and mfilter.

        Output:
	     out - the range compressed simulated radar data in the time domain.

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    out = data*mfilter
    if windowNum!=0:
        i1 = np.argmin(abs(f-fmin))
        i2 = np.argmin(abs(f-fmax))
        out[0:i1] = out[0:i1]*0
        out[i1:i2] = np.kaiser(i2-i1,windowNum)*out[i1:i2]
        out[i2:-1] = out[i2:-1]*0
    out = np.fft.ifft(np.fft.ifftshift(out))*np.sqrt(len(data))
    return out

def interpFreq(t2,t1,Fnew):
    '''
        Usage:
          This function interpolates a frequency spectrum. In the case that the		user inputs a matched filter that has a different number of elements 
	than the source pulse, this function will interpolate it so that the 		number of elements match. I advise inputting pulses and matched filters		that have the same number of elements.

        Input Variables:
             t2 - the time values to interpolate to
	     t1 - the time values corresponding to the input Fnew
	     Fnew - a frequency domain transform of the time series data 
		corresponding to t1. 

        Output
	     t2 - 
             out - Fnew but interpolated to fit the time series values in t2.

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    temp = np.fft.ifft(np.fft.ifftshift(Fnew))
    return t2, np.fft.fftshift(np.fft.fft(np.interp(t2,t1,temp,left=0,right=0)))
    
def dB(x):
    '''
        Usage:
          This function converts amplitude to power in dB.

        Input Variables:
             x - an array of amplitude values.

        Output:
	     an array in dB

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    return 20*np.log10(np.abs(x))

def writeCSVs(h, t, addTime, finalData, result, modelFile):
    '''
        Usage:
          This function writes the simulation outputs into CSV files.

        Input Variables:
             h - the thicknesses of each layer.
	     t - the time values corresponding to the output data.
	     addTime - To be implemented in a future version. ignore.
	     finalData - The output radar reflection data in power (dB).
		The array is normalized so that the most powerful reflection 
		has a value of 0 dB.
	     result - The output radar reflection data in amplitude (V/m).
	     modelFile - a string with the name and path to the output csv file
		to be created. 

        Output:
	     The depth and real permittivity of each layer for plotting purpose
	     

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    outData_fname = modelFile + "_output_data.csv"
    with open(outData_fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Time (s)','power (dB)','amplitude (real)', 'amplitude (imag)'])
        for ii in np.arange(len(t)):
            _t = t[ii] + addTime
            _f = np.real(finalData[ii])
            _ra = np.real(result[ii])
            _ia = np.imag(result[ii])
            writer.writerow([_t, _f, _ra, _ia])

	#
	# Model Output
	#
    outModel_file = modelFile + "_model.csv"
    layers = np.append(np.cumsum(h),[0])
    z = np.zeros(int(nl*2-1))
    epsz = np.zeros(int(nl*2-1))
    for i in range(0,nl):
        if i == (nl-1):
            z[i*2] = layers[i]
            epsz[i*2] = np.real(eps[i+1]/8.85e-12)
        else:
            z[i*2] = layers[i]
            z[i*2+1] = layers[i+1]
            epsz[i*2] = np.real(eps[i+1]/8.85e-12)
            epsz[i*2+1] = np.real(eps[i+1]/8.85e-12)
    
    with open(outModel_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Depth (m)', 'eps'])
        for i in range(0,len(z)):
            writer.writerow([z[i], epsz[i]])
    return z, epsz
    
###############################################################################
#                       Main script
###############################################################################
def main(args):
    '''
        Usage:
          This function runs the radar simulation

        Input Variables:
             args - the model file with all simulation parameters input as a 
		name and path input

        Output
	     none

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    
    if len(args) != 2:
        print("Please provide 1 model input file name (txt)")
        quit()
    else: 
        modelFile = args[1]
    
    
    loadModel(modelFile)
    modelFile = modelFile[0:-4]
    
    t,source = loadTruePulse(pulseFile)

    n=len(t)
    d = t[1]-t[0]

    F = np.fft.fftshift(np.fft.fft(source))/np.sqrt(n)
    Ffreq = np.fft.fftshift(np.fft.fftfreq(n,d))


    Fnew = np.zeros((n),dtype=complex)
    for i in range(0,n):
        Fnew[i] = freqComponent(abs(2*np.pi*Ffreq[i]),F[i])

    

    if dB_noise>-100:
        Prx = np.max(Fnew)**2
        Pnoise = Prx*10**(dB_noise/10)*np.sqrt(n)
        noise = np.sqrt(Pnoise)*(2*np.random.rand((n))-1 + 1j*(2*np.random.rand((n))-1))
        Fnew = Fnew+noise
    
    
    
    Aillum = (H**2)*np.pi
    sigma0 = (fmax-fmin)*(np.max(t)-np.min(t))
    lam = c/((fmin+fmax)/2)
    Prx = sigma0*Aillum*(lam**2)/(64*(H**4)*np.pi**3)
    Fnew = Fnew*np.sqrt(Prx)
    
    
    if filterFile != "none":
        tmf,f,matchFilter = loadMatchFilter(filterFile)
        if (f!= Ffreq).any():
            t, Fnew = interpFreq(tmf,t,Fnew)
        result = rangeCompress(Fnew,matchFilter,np.array(f))
    else:
        result = np.fft.ifft(np.fft.ifftshift(Fnew))*np.sqrt(n)
    result = np.fft.ifftshift(result)
    

    
    addTime=0
    if timeFlag:
        addTime = 2*H/c
        
    finalData = dB(result)-np.max(dB(result))
    #
    # Write output CSVs
    #
    z, epsz = writeCSVs(h, t, addTime, finalData, result, modelFile)
   
    if plotResult:
        plotDataModel(addTime, np.array(t), result, z, epsz, modelFile)
    return 
        
    
###############################################################################
#                       plot function
###############################################################################

def plotDataModel(addTime, finalT, sout, z, epsz, modelFile):
    '''
        Usage:
          This function plots the layer model and output data side by side

        Input Variables:
             addTime - adds this amount of time to all elements in the time axis
	     finalT - the time axis corresponding to the output data.
	     sout - the reflected signal in power (dB)
	     z - layer depths
	     epsz - layer epsilon values
	     modelFile - the name of the file to save the output figure as. 

        Output:
		none

        Written by: Sam Courville
    
        Last Edited: 04/03/2020

    '''
    dB_limit = -50
    endT = 0.5
    
    for i in range(1,nl):
      endT = endT+2*(h[i]/(c/np.sqrt((eps[i]/eps0)))*1e6)

    fig=plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')

    axFont = 28
    titFont = 32

    plt.subplot(1,2,1)
    plt.plot(dB(sout)-np.max(dB(sout)),(finalT+addTime)*10**6,linewidth=3)
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([0+addTime*10**6,endT+addTime*10**6])
    axes.set_xlim([dB_limit,0])
    axes.set_ylabel('Time ($\mu$s)',fontsize=axFont)
    axes.set_xlabel('dB',fontsize=axFont)
    axes.xaxis.set_label_position('top') 
    axes.xaxis.tick_top()    
    axes.invert_yaxis()

    plt.subplot(1,2,2)
    for i in range(0,2*nl-1):
        if i==0:
            plt.plot([1.0, epsz[i]],[z[i], z[i]], color='r', linewidth=4.0)
            plt.plot([epsz[i],epsz[i+1]],[z[i],z[i+1]], color='r', linewidth=4.0)
        elif i==2*nl-2:
            plt.plot([epsz[i-1], epsz[i]],[z[i], z[i]], color='r', linewidth=4.0)
            plt.plot([epsz[i],epsz[i]],[z[i],z[i]+15], color='r', linewidth=4.0)
        else:
            plt.plot([epsz[i-1], epsz[i]],[z[i], z[i]], color='r', linewidth=4.0)
            plt.plot([epsz[i],epsz[i+1]],[z[i],z[i+1]], color='r', linewidth=4.0)
    axes = plt.gca()
    axes.set_ylabel('Depth (m)',fontsize=axFont)
    axes.set_xlabel('$\epsilon_r$',fontsize=axFont)
    axes.set_ylim([-0.5,np.max(z)+15])
    axes.set_xlim([0,np.max(epsz)+1])
    axes.invert_yaxis()
    axes.xaxis.set_label_position('top') 
    axes.xaxis.tick_top()
    fig.tight_layout()
    plt.savefig(modelFile + '_output_figure.png')
    plt.close('All')
    return 

main(sys.argv)
