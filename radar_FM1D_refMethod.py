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
          This function ... ???

        Input Variables:
             omega -- ???
             a -- ???

        Output Variables:
             b -- ??

        Written by: Sam Courville
    
        Last Edited: ???

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
          This function ... ???

        Input Variables:
             fileName -- ???

        Global variables updated:
           Explain global variables

        Written by: Sam Courville
    
        Last Edited: ???

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
        H = float(f.readline().strip())                 # Spacecraft Altitude (m)
        junk = f.readline()
        dB_noise = float((f.readline().strip()))          # What's the noise level in dB?
        dB_noise = dB_noise-100/dB_noise                # This is an approximation. Ask me for expln. 
        junk = f.readline()
        nl = float((f.readline().strip()))                # number of layers
        junk = f.readline()
        depthsstr = (f.readline().strip())
        depthsstr = depthsstr.split(' ')
        depths = [float(i) for i in depthsstr]         # Layer Thicknesses (m)
        junk = f.readline()
        epsrstr = (f.readline().strip())
        epsrstr = epsrstr.split(' ')
        epsr = [float(i) for i in epsrstr]             # real relative permitivity
        junk = f.readline()
        useConductivity = float((f.readline().strip()))   # number of layers
        junk = f.readline()
        lossTangentstr = (f.readline().strip())           # loss tangent or conductivity
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
          This function ... ???

        Input Variables:
             fileName -- ???

        Output???

        Written by: Sam Courville
    
        Last Edited: ???

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
          This function ... ???

        Input Variables:
             fileName -- ???

        Output???

        Written by: Sam Courville
    
        Last Edited: ???

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
          This function ... ???

        Input Variables:
             fileName -- ???

        Output???

        Written by: Sam Courville
    
        Last Edited: ???

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
          This function ... ???

        Input Variables:
             fileName -- ???

        Output???

        Written by: Sam Courville
    
        Last Edited: ???

    '''
    temp = np.fft.ifft(np.fft.ifftshift(Fnew))
    return t2, np.fft.fftshift(np.fft.fft(np.interp(t2,t1,temp,left=0,right=0)))
    
def dB(x):
    '''
        Usage:
          This function ... ???

        Input Variables:
             fileName -- ???

        Output???

        Written by: Sam Courville
    
        Last Edited: ???

    '''
    return 20*np.log10(x)

def writeCSVs(h, t, addTime, finalData, result, modelFile):
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
  nz = 200
  z = np.linspace(0,np.max(layers)+10,nz)
  currInd = -1
  epsz = np.zeros((nz))
  with open(outModel_file, 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Depth (m)', 'eps'])
    for i in range(0,nz):
      if z[i] < layers[currInd]:
          epsz[i] = eps[currInd]/eps0
      elif currInd<nl:
          currInd = currInd+1
          epsz[i] = eps[currInd]/eps0
      else:
          epsz[i] = eps[currInd]/eps0
      writer.writerow([z[i], epsz[i]])
  return z, epsz
    
###############################################################################
#                       Main script
###############################################################################
def main(args):
    '''
        Usage:
          This function ... ???

        Input Variables:
             fileName -- ???

        Output???

        Written by: Sam Courville
    
        Last Edited: ???

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
          This function ... ???

        Input Variables:
             fileName -- ???

        Output???

        Written by: Sam Courville
    
        Last Edited: ???

    '''
    dB_limit = -50
    endT = 6

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
    plt.plot(epsz,z,'r',linewidth=4.0)
    axes = plt.gca()
    axes.set_ylabel('Depth (m)',fontsize=axFont)
    axes.set_xlabel('$\epsilon_r$',fontsize=axFont)
    axes.set_ylim([-0,np.max(z)])
    axes.set_xlim([0,np.max(epsz)+1])
    axes.invert_yaxis()
    axes.xaxis.set_label_position('top') 
    axes.xaxis.tick_top()
    plt.savefig(modelFile + '_output_figure.png')
    plt.close('All')
    return 

main(sys.argv)
