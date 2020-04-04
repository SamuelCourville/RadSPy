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
		'''
			class initialization function
		'''
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
		'''
		Usage:
			This function sets the layered model to simulate.

		Input Variables:
			H -- The altitude of the spacecraft in meters above the
				surface
			thicknesses -- The thickness of each layer in meters.
				(array)
			eps -- the real relative dielectric constant of each 
				layer. (array)
			lossOrCond -- The loss tangent or conductivity value of
				each layer. (array)
			useConductivity -- Enter 1 if you input conductivity 
				values for 'lossOrCond,' or 0 if you input
				loss tangent values.

		Output Variables:
			none

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''
		self.useConductivity=useConductivity
		self.H = H
		self.lossTangent = np.append(self.lossTangent,lossOrCond)
		self.eps = np.append(self.eps,eps)*self.eps0
		self.h = np.append(self.h,thicknesses)
		self.nl = len(self.eps)-1

	def setPulse(self,pulse,time):
		'''
		Usage:
			This function sets the source pulse used for simulation.

		Input Variables:
			pulse -- an array containing a time series of the source
				 pulse to be sued for simulation.
			t -- an array with values corresponding to the time axis
				of the pulse.

		Output Variables:
			none

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''
		self.pulse = pulse
		self.t = time

	def setMatchFilter(self,filt,f,t):
		'''
		Usage:
			This function sets the matched filter used for range
			compression.

		Input Variables:
			filt -- an array containing the matched filter in the
				frequency domain.
			f -- the frequency axis corresponding to the filter. 
			t -- the time axis corresponding to the fft of the 
				filter.

		Output Variables:
			none

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''
		self.matchFilter = filt
		self.f = f
		self.tmf = t

	def setWindowParam(self,windowNum,fmin,fmax):
		'''
		Usage:
			This function sets the simulators windowing parameters

		Input Variables:
			windowNum -- the Kaiser window degree. 6 corresponds to 
				a Hann wondow. 0 will result in no windowing.
			fmin -- the low frequency end of the window. 
			fmax -- the high frequency end of the window.

		Output Variables:
			none

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''
		self.windowNum=windowNum
		self.fmin = fmin
		self.fmax = fmax


	#######################################################################
	#               Main simulation functions
	#######################################################################
	
	def __freqComponent__(self,omega,a):
		'''
		Usage:
			This function implements the math from Wait (1970) to 
			solve for the response from the reflection of a 
			monochromatic electromagnetic wave in a horizontally 
			stratified medium. This solves the reflection for one 
			frequency component of the radar pulse.

		Input Variables:
			omega -- the angular frequency component of the radar 
				pulse.
			a -- the amplitude of the frequency component in the 
				source pulse.

		Output Variables:
			b -- the amplitude of the reflected frequency component.

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''
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
		'''
		Usage:
			This function runs the radar simulation

		Input Variables:
			none - but the simulation parameters must be set for the
			simulator class object via the load or set methods 
			provided.
			
		Output
			a - a 2D array with two columns. The second column is 
			the time series of the reflected radar signal in power 
			(dB), and the first is the time axis values associated 
			with the signal.

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''

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
				self.t, Fnew = self.__interpFreq__(self.tmf,self.t,Fnew)
			result = self.__rangeCompress__(Fnew,self.matchFilter,np.array(self.f))
		else:
			result = np.fft.ifft(np.fft.ifftshift(Fnew))*np.sqrt(n)
		result = np.fft.ifftshift(result)
    
		addTime=0
		if self.timeFlag:
			addTime = 2*self.H/c
        
		finalData = self.__dB__(result)-np.max(self.__dB__(result))
		a = np.transpose(np.asarray([np.array(self.t)+addTime, finalData, result]))

		if self.plotResult:
			self.plotDataModel(addTime,np.array(self.t),result)
    
		return a
        
    
	#######################################################################
	#               File loading functions
	#######################################################################
	def loadModel(fileName):
		'''
		Usage:
		This function loads a model file. The model file should contain 
		all of the global variables needed to run a successful 
		simulation. See the example model text file in the github repo 
		for an example model file.

		Input Variables:
			fileName -- the name of the text file which includes the
				simulation parameters.

        	Global variables updated:
		The model text file should include the following global vars:
			H - the altitude (height) of the spacecraft above the 
				ground (m)
			dB_noise - if greater than -101 dB, the code will add 
				random gaussian noise to the synthetic radar 
				return at the level denoted by dB_noise.
			h - an array which represents the thicknesses (m) of 
				each layer
			eps - array, the relative real dielectric constant of 
				each layer
			lossTangent - an array of loss tangent values for the 
				layers. if useConductivity is set to 1, then 
				this variable should be set to conductivity 
				values (S/m).
			nl - the number of layers in the model
			useConductivity - 1 or 0. See lossTangent variable 
				description.
			windowNum - The Kaiser window degree to be applied on 
				the output data. a 0 will apply no window.
			fmin - The low bound on the Kaiser frequency window(Hz).
				 use 15e6 for SHARAD.
			fmax - The high bound on the kaiser frequency window(Hz)
			 	Use 20e6 for SHARAD.
			pulseFile - a filename and path to a csv file with the 
				source pulse. See example modelFile and 
				corresponding pulse.
			filterFile - a filename and path to a csv file with the
				 matched filter. See example.

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''	
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
		'''
		Usage:
		This function loads a pulse file.

		Input Variables:
			fileName -- this should be a string with the name and 
				path to a file containing the desired source 
				pulse to simulate. The CSV file should contain 
				two columns, time and amplitude.

		Output:
			The output is two arrays, E, which is a time series of 
			the Electric field amplitude from the input file, and t,
			which is an array of the corresponding time values for 
			the time series E.

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
		self.pulse = E
		self.t = t
		return t,E    
    
	def loadMatchFilter(self,fileName):
		'''
		Usage:
			This function loads the matched filter

		Input Variables:
			fileName -- this should be a string with the name and 
				path to a file containing the desired matched 
				filter to compress with in the frequency domain.
				The CSV file should contain three columns, the 
				time series value for the corresponding pulse, 
				the frequency axis for the filter, and the 
				amplitude spectrum of the matched filter.

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
		self.matchFilter = E
		self.f = f
		self.tmf = t
		return t,f,E

	########################################################################	                       Range Compression and windowing
	#######################################################################
	def __rangeCompress__(self,data, mfilter, f):
		'''
		Usage:
		This function range compresses the resulting reflected 
		electromagnetic field with the given matched filter. 	
		
		Input Variables:
			data - the frequency spectrum of the refelcted E-field
			mfilter - the matched filter to compress data within the
				f-domain
			f - the frequency axis values for data and mfilter.

		Output:
			out - the range compressed simulated radar data in the 
				time domain.
	
		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''
		out = data*mfilter
    		if self.windowNum!=0:
        		i1 = np.argmin(abs(f-self.fmin))
        		i2 = np.argmin(abs(f-self.fmax))
        		out[0:i1] = out[0:i1]*0
        		out[i1:i2] = np.kaiser(i2-i1,self.windowNum)*out[i1:i2]
        		out[i2:-1] = out[i2:-1]*0
    		out = np.fft.ifft(np.fft.ifftshift(out))*np.sqrt(len(data))
    		return out

	def __interpFreq__(self,t2,t1,Fnew):
		'''
		Usage:
		This function interpolates a frequency spectrum. In the case 
		that the user inputs a matched filter that has a different 
		number of elements	
	
		Input Variables:
			t2 - the time values to interpolate to
			t1 - the time values corresponding to the input Fnew
			Fnew - a frequency domain transform of the time series 
				data corresponding to t1.
		Output:
			t2 -
			out - Fnew but interpolated to fit the time series 
				values in t2

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''
		temp = np.fft.ifft(np.fft.ifftshift(Fnew))
    		return t2, np.fft.fftshift(np.fft.fft(np.interp(t2,t1,temp,left=0,right=0)))
    
	def __dB__(self,x):
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
		return 20*np.log10(x)
    
	#######################################################################
	#                plot function
	#######################################################################
	def plotDataModel(self,addTime,finalT,sout):
		'''
		Usage:
		This function plots the layer model and output data side by side
		
		Input Variables:
			addTime - adds this amount of time to all elements in 
				the time axis
			finalT - the time axis corresponding to the output data.
			sout - the reflected signal in power (dB)

		Output:
			none

		Written by: Sam Courville
		Last Edited: 04/03/2020
		'''

		dB_limit = -50
    		
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
    		axes.set_xlabel('dB',fontsize=axFont)
    		axes.xaxis.set_label_position('top') 
    		axes.xaxis.tick_top()    
    		axes.invert_yaxis()


		layers = np.append(np.cumsum(self.h),[0])
		z = np.zeros(int(self.nl*2-1))
		epsz = np.zeros(int(self.nl*2-1))
		for i in range(0,self.nl):
			if i == (self.nl-1):
				z[i*2] = layers[i]
				epsz[i*2] = np.real(self.eps[i+1]/self.eps0)
			else:
				z[i*2] = layers[i]
				z[i*2+1] = layers[i+1]
				epsz[i*2] = np.real(self.eps[i+1]/self.eps0)
				epsz[i*2+1] = np.real(self.eps[i+1]/self.eps0)


    		plt.subplot(1,2,2)
		for i in range(0,2*self.nl-1):
			if i==0:
				plt.plot([1.0, epsz[i]],[z[i], z[i]], color='r', linewidth=4.0)
				plt.plot([epsz[i],epsz[i+1]],[z[i],z[i+1]], color='r', linewidth=4.0)
			elif i==2*self.nl-2:
				plt.plot([epsz[i-1], epsz[i]],[z[i], z[i]], color='r', linewidth=4.0)
				plt.plot([epsz[i],epsz[i]],[z[i],z[i]+15], color='r', linewidth=4.0)
			else:
				plt.plot([epsz[i-1], epsz[i]],[z[i], z[i]], color='r', linewidth=4.0)
				plt.plot([epsz[i],epsz[i+1]],[z[i],z[i+1]], color='r', linewidth=4.0)
    		axes = plt.gca()
    		axes.set_ylabel('Depth (m)',fontsize=axFont)
    		axes.set_xlabel('$\epsilon_r$',fontsize=axFont)
    		axes.set_ylim([-0,np.max(z)+15])
    		axes.set_xlim([0,np.max(epsz)+1])
    		axes.invert_yaxis()
    		axes.xaxis.set_label_position('top') 
    		axes.xaxis.tick_top()
    		plt.show()

    		b = np.transpose(np.asarray([ z,epsz]))
    
