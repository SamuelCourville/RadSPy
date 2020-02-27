###############################################################################
###############################################################################
#                       1D Radar simulator
###############################################################################
###############################################################################

###############################################################################
#                       import statements and constants
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import csv
import sys

###############################################################################
#                       Simulator class definition
###############################################################################

class EMRM_simulator:
	# Constants
	c = 2.99792*10**8     # speed of light [m/s]
	eps0 = 8.854188e-12   # vacuum permitivity [F/m or something like that]
	mu = 4*np.pi*10**-7
	

	#######################################################################
	#               Initialization functions
	#######################################################################
	def __init__(self):
		# Global Variables
		self.H = 0
		self.dB_noise = -100
		self.h = np.array([0])
		self.eps = np.array([1.0])
		self.lossTangent = np.array([0.0])
		self.nl=0
		self.useConductivity = 0
		self.windowNum = 0
		self.fmin = 0
		self.fmax = 0
		self.pulse = np.array([0.0])
		self.t = np.array([0.0])
		self.matchFilter = np.array([0.0])
		self.f = np.array([0.0])
		self.tmf = np.array([0.0])
		self.plotTime = 10 
		
		# Flags
		self.timeFlag = False
		self.plotResult = False    

	def setModel(self,H,thicknesses,eps,lossOrCond,useConductivity):
		self.useConductivity=useConductivity
		self.H = H
		self.lossTangent = np.append(self.lossTangent,lossOrCond)
		self.eps = np.append(self.eps,eps)*self.eps0
		self.h = np.append(self.h,thicknesses)
		self.nl = len(self.eps)-1

	def setPulse(self,pulse,time):
		self.pulse = pulse
		self.t = time

	def setMatchFilter(self,filt,f,t):
		self.matchFilter = filt
		self.f = f
		self.tmf = t

	def setWindowParam(self,windowNum,fmin,fmax):
		self.windowNum=windowNum
		self.fmin = fmin
		self.fmax = fmax


	#######################################################################
	#               Main simulation functions
	#######################################################################
	
	def __freqComponent__(self,omega,a):
		if not self.useConductivity:
			sigma = abs(omega)*self.eps*self.lossTangent
		else:
			sigma = self.lossTangent
    
		M = int(self.nl)
		K = np.zeros((M+1),dtype=complex)
		Z = np.zeros((M),dtype=complex)
		u = np.zeros((M+1),dtype=complex)
		gamma2 = np.zeros((M+1),dtype=complex)

		if omega==0:
			omega = 0.000000001
		lamb=0
		for i in range(0,M+1):
			gamma2[i] = 1j*sigma[i]*self.mu*omega-self.eps[i]*self.mu*omega**2
			u[i] = np.sqrt(lamb**2+gamma2[i])
			K[i] = u[i]/(sigma[i]+1j*omega*self.eps[i])
		if M>1:
			indexes = np.flipud(np.linspace(1,M-2,M-2,dtype=int))
			Z[-1] = K[M-1]*(K[M]+K[M-1]*np.tanh(u[M-1]*self.h[M-1]))/(K[M-1]+K[M]*np.tanh(u[M-1]*self.h[M-1]))
			for i in indexes:
				Z[i] = K[i]*(Z[i+1]+K[i]*np.tanh(u[i]*self.h[i]))/(K[i]+Z[i+1]*np.tanh(u[i]*self.h[i]))
		else:
			Z = np.zeros((M+1),dtype=complex)
			Z[1] = K[M-1]*(K[M]+K[M-1]*np.tanh(u[M-1]*self.h[M-1]))/(K[M-1]+K[M]*np.tanh(u[M-1]*self.h[M-1]))
    
		a = a/Z[1]
		b = a*(K[0]-Z[1])/(K[0]+Z[1])
		b = Z[1]*b
		return b


	def runSim(self):    
		if len(self.eps) < 2:
			print("Please provide 1 model input file name (txt)")
			quit()

		n=len(self.t)
		d = self.t[1]-self.t[0]
		F = np.fft.fftshift(np.fft.fft(self.pulse))/np.sqrt(n)
		Ffreq = np.fft.fftshift(np.fft.fftfreq(n,d))
		Fnew = np.zeros((n),dtype=complex)
		for i in range(0,n):
			Fnew[i] = self.__freqComponent__(abs(2*np.pi*Ffreq[i]),F[i])

		if self.dB_noise>-100:
			Prx = np.max(Fnew)**2
			Pnoise = Prx*10**(self.dB_noise/10)*np.sqrt(n)
			noise = np.sqrt(Pnoise)*(2*np.random.rand((n))-1 + 1j*(2*np.random.rand((n))-1))
			Fnew = Fnew+noise
    
		Aillum = (self.H**2)*np.pi
		sigma0 = (self.fmax-self.fmin)*(np.max(self.t)-np.min(self.t))
		lam = self.c/((self.fmin+self.fmax)/2)
		Prx = sigma0*Aillum*(lam**2)/(64*(self.H**4)*np.pi**3)
		Fnew = Fnew*np.sqrt(Prx)
    
		if len(self.matchFilter) > 1:
			if (self.f!= Ffreq).any():
				self.t, Fnew = __interpFreq__(self.tmf,self.t,Fnew)
			result = self.__rangeCompress__(Fnew,self.matchFilter,np.array(self.f))
		else:
			result = np.fft.ifft(np.fft.ifftshift(Fnew))*np.sqrt(n)
		result = np.fft.ifftshift(result)
    
		addTime=0
		if self.timeFlag:
			addTime = 2*self.H/c
        
		finalData = self.__dB__(result)-np.max(self.__dB__(result))
		a = np.transpose(np.asarray([np.array(self.t)+addTime, finalData, result]))
		#np.savetxt(modelFile + "_output_data.csv", a, delimiter=",")

		if self.plotResult:
			self.plotDataModel(addTime,np.array(self.t),result)
    
		return a
        
    
	#######################################################################
	#               File loading functions
	#######################################################################
	def loadModel(fileName):
    		global pulseFile
    		global filterFile
    		with open(fileName) as f:
        		junk = f.readline()
        		junk = f.readline()
        		H = float(f.readline().strip())# Spacecraft Altitude (m)
        		junk = f.readline()
        		dB_noise = float((f.readline().strip()))# What's the noise level in dB?
        		dB_noise = dB_noise-100/dB_noise# This is an approximation. Ask me for expln. 
        		junk = f.readline()
        		nl = float((f.readline().strip())) # number of layers
        		junk = f.readline()
        		depthsstr = (f.readline().strip())
        		depthsstr = depthsstr.split(' ')
        		depths = [float(i) for i in depthsstr] # Layer Thicknesses (m)
        		junk = f.readline()
        		epsrstr = (f.readline().strip())
        		epsrstr = epsrstr.split(' ')
        		epsr = [float(i) for i in epsrstr] # real relative permitivity
        		junk = f.readline()
        		useConductivity = float((f.readline().strip())) # number of layers
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
    
    
	def loadPulse(self,fileName):
		with open(fileName) as csvfile:
			readCSV = csv.reader(csvfile,delimiter=',')
			t = []
			E = []
			for row in readCSV:
				t.append(np.real(complex(row[0])))
				E.append(complex(row[1].replace('+-', '-', 1)))
		self.pulse = E
		self.t = t
		return t,E    
    
	def loadMatchFilter(self,fileName):
		with open(fileName) as csvfile:
			readCSV = csv.reader(csvfile,delimiter=',')
			t = []
			f = []
			E = []
			for row in readCSV:
				t.append(np.real(complex(row[0])))
				f.append(np.real(complex(row[1])))
				E.append(complex(row[2].replace('+-', '-', 1)))
		self.matchFilter = E
		self.f = f
		self.tmf = t
		return t,f,E

	########################################################################	                       Range Compression and windowing
	#######################################################################
	def __rangeCompress__(self,data, mfilter, f):
    		out = data*mfilter
    		if self.windowNum!=0:
        		i1 = np.argmin(abs(f-self.fmin))
        		i2 = np.argmin(abs(f-self.fmax))
        		out[0:i1] = out[0:i1]*0
        		out[i1:i2] = np.kaiser(i2-i1,self.windowNum)*out[i1:i2]
        		out[i2:-1] = out[i2:-1]*0
    		out = np.fft.ifft(np.fft.ifftshift(out))*np.sqrt(len(data))
    		return out

	def __interpFreq__(t2,t1,Fnew):
    		temp = np.fft.ifft(np.fft.ifftshift(Fnew))
    		return t2, np.fft.fftshift(np.fft.fft(np.interp(t2,t1,temp,left=0,right=0)))
    
	def __dB__(self,x):
    		return 20*np.log10(x)
    		#return 20*np.log10(np.abs(x))
    
	#######################################################################
	#                plot function
	#######################################################################
	def plotDataModel(self,addTime,finalT,sout):
    		dB_limit = -50
    		#endT = 0.10e3
    		endT = self.plotTime 

    		fig=plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')

    		axFont = 28
    		titFont = 32

    		plt.subplot(1,2,1)
    		plt.plot(self.__dB__(sout)-np.max(self.__dB__(sout)),(finalT+addTime)*10**6,linewidth=3)
    		#plt.plot(self.__dB__(sout)-np.max(self.__dB__(sout)),(finalT+addTime)*10**3,linewidth=3)
    		plt.grid()
    		axes = plt.gca()
    		axes.set_ylim([0+addTime*10**6,endT+addTime*10**6])
    		axes.set_xlim([dB_limit,0])
    		axes.set_ylabel('Time ($\mu$s)',fontsize=axFont)
    		#axes.set_ylabel('Time (ms)',fontsize=axFont)
    		axes.set_xlabel('dB',fontsize=axFont)
    		axes.xaxis.set_label_position('top') 
    		axes.xaxis.tick_top()    
    		#axes.set_title('Radar return',fontsize=titFont)
    		axes.invert_yaxis()


    		layers = np.append(np.cumsum(self.h),[0])
    		nz = 200
    		z = np.linspace(0,np.max(layers)+10,nz)
    		currInd = -1
    		epsz = np.zeros((nz))
    		for i in range(0,nz):
        		if z[i] < layers[currInd]:
            			epsz[i] = self.eps[currInd]/self.eps0
        		elif currInd<self.nl:
            			currInd = currInd+1
            			epsz[i] = self.eps[currInd]/self.eps0
        		else:
            			epsz[i] = self.eps[currInd]/self.eps0
        
    
    		plt.subplot(1,2,2)
    		plt.plot(epsz,z,'r',linewidth=4.0)
    		axes = plt.gca()
    		axes.set_ylabel('Depth (m)',fontsize=axFont)
    		axes.set_xlabel('$\epsilon_r$',fontsize=axFont)
    		#axes.set_xlabel('$v_p$ (km/s)',fontsize=axFont)
    		axes.set_ylim([-0,np.max(z)])
    		axes.set_xlim([0,np.max(epsz)+1])
    		#axes.set_title('Permitivity Model',fontsize=titFont)
    		axes.invert_yaxis()
    		axes.xaxis.set_label_position('top') 
    		axes.xaxis.tick_top()
    		plt.show()
    		#plt.savefig(modelFile + '_output_figure.png')

    		b = np.transpose(np.asarray([ z,epsz]))
    		#np.savetxt(modelFile + "_model.csv", b, delimiter=",")
    
