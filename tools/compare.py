import numpy as np
import csv
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import sys

endT = 15 


if len(sys.argv) != 3:
	print("Please provide model input file name (no extension) and data input file name (no extension).")
	quit()
else:
	modelFile = sys.argv[1]
	dataFile = sys.argv[2]




def dB(x):
	return 10*np.log10(x)

def FFT(data):
	D = np.fft.fft(data)/len(data)
	return D

def IFFT(data):
	D = np.fft.ifft(data)*len(data)
	return D

def rangeCompress(data,N,beta=6):
	chirp = np.zeros(N,np.float)
	fhi = 25.00e6
	flo = 15.00e6
	plen = 85.05e-6
	delay_res = 135.0e-6/N
	nsamp = int(plen/delay_res)
	fslope = (flo-fhi)/plen
	
	x_term = np.arange(nsamp)
	ctime = np.arange(nsamp) * delay_res
	arg = 2.0*np.pi*ctime*(fhi+fslope*ctime/2.0)
	temp = np.sin(arg)
	chirp[0:len(temp)] = temp
	dechirp = (np.conj(FFT(chirp)))
	plt.plot(data)
	plt.show()
	
	window = np.zeros(N)
	window[0:int(N/2)] = np.kaiser(N/2,beta)
	range_record = IFFT(window*FFT(data)*dechirp)
	#range_record = window*FFT(data)
	return range_record

def compare(d1,d2,t1,t2,d,m):
	d3 = np.interp(t1,t2,d2)
	l2Norm = np.linalg.norm(d3-d1)
	fig = plt.figure(figsize=(8,9),dpi=80,facecolor='w',edgecolor='k')
	print(l2Norm)

	p1 = dB(np.abs(d1)/np.max(np.abs(d1)))
	p2 = d2 #dB(np.abs(d2)/np.max(np.abs(d2)))

	f = plt.figure(figsize=(7,10))
	ax1 = f.add_subplot(1,2,1)
	ax1.plot(p1,t1,'r',linewidth=3)
	ax1.plot(p2,t2*10**6,'k',linewidth=3)
	ax1.set_ylim([-endT/16,endT])
	ax1.set_xlim([np.min(p1),np.max(p1)])
	ax1.set_ylabel('$t$ ($\mu$s)',fontsize=16)
	ax1.set_xlabel('Normalized Power (dB)',fontsize=16)
	ax1.set_title('Data',fontsize=20)
	ax1.invert_yaxis()
	ax1.legend(['True','Synthetic'])


	layerPlot=0
	if layerPlot==1:
		layers = np.array([1.15, 0.85, 0.75, 0.76, 0.87, 0.80, 0.80, 1.35, 0.85, 0.95, 1.45, 1.1, 0.55, 0.65, 0.6, 0.6, 0.65, 0.70, 1.1, 0.95, 1.1, 1.39, 0.8, 1.15, 1.5, 1.5])
		counter = -1
		last = 0
		toPlot = np.zeros(len(d))
		for i in range(0,len(d)):
			if np.real(m[i])-3.1 < 0.001:
				toPlot[i] = 0
				last = 0
			else:
				if last == 0:
					counter = counter+1
				toPlot[i] = layers[counter]
				last = 1
		ax2 = f.add_subplot(1,2,2)
                ax2.step(toPlot,d,'k',linewidth=3)
                ax2.set_ylim([-75,np.max(d)+20])
                ax2.set_xlim([0,1.5])
                ax2.set_ylabel('$z$ (m)',fontsize=16)
                ax2.set_xlabel('dust thickness $(m)$',fontsize=16)
                ax2.set_title('Model',fontsize=20)
                ax2.invert_yaxis()
	elif layerPlot==2:
		layers = np.array([4.6,4.2,4.0,4.0,4.3,4.3,4.2,5.1,4.4,4.5,5.3,4.5,4.2,4.0,3.9,4.1,3.6,3.8,4.4,4.6,4.5,4.8,4.3,4.4,5.5,5.5])
		counter = -1
		last = 0
		toPlot = np.zeros(len(d))
		for i in range(0,len(d)):
			if np.real(m[i])-3.1 < 0.001:
				toPlot[i] = 0
				last = 0
			else:
				if last == 0:
					counter = counter+1
				toPlot[i] = layers[counter]
				last = 1
		ax2 = f.add_subplot(1,2,2)
                ax2.step(toPlot,d,'k',linewidth=3)
                ax2.set_ylim([-75,np.max(d)+20])
                ax2.set_xlim([3,6])
                ax2.set_ylabel('$z$ (m)',fontsize=16)
                ax2.set_xlabel('Layer permitivity $(\epsilon_r)$',fontsize=16)
                ax2.set_title('Model',fontsize=20)
                ax2.invert_yaxis()
		
	else:
		ax2 = f.add_subplot(1,2,2)
		ax2.step(m,d,'k',linewidth=3)
		ax2.set_ylim([-80,np.max(d)])
		ax2.set_xlim([0,np.max(m)+1])
		ax2.set_ylabel('$z$ (m)',fontsize=16)
		ax2.set_xlabel('$\epsilon_r$',fontsize=16)
		ax2.set_title('Model',fontsize=20)
		ax2.invert_yaxis()
	
	bbox=plt.gca().get_position()
	offset=-.06
	plt.gca().set_position([bbox.x0-offset, bbox.y0, bbox.x1-bbox.x0, bbox.y1 - bbox.y0])


	#f.show()
	plt.savefig(modelFile+'_w_data.png')

	return l2Norm 


def main():
	
	with open(dataFile+'.csv') as csvfile:
		readCSV = csv.reader(csvfile,delimiter=',')	
		timed = []
		Ed = []
		for row in readCSV:
			timed.append(complex(row[0]))
			Ed.append(complex(row[1]))

	with open(modelFile+'_output_data.csv') as csvfile:
		readCSV = csv.reader(csvfile,delimiter=',')	
		timem = []
		Em = []
		for row in readCSV:
			timem.append(complex(row[0]))
			if "+-" in row[1]:
				stupid = row[1].replace("+-","-")
				Em.append(complex(stupid))
			else:
				Em.append(complex(row[1]))
				

	with open(modelFile+'_model.csv') as csvfile:
		readCSV = csv.reader(csvfile,delimiter=',')	
		d = []
		m = []
		for row in readCSV:
			d.append(complex(row[0]))
			m.append(complex(row[1]))

	tm = np.real(timem)
	td=np.real(timed)
	Rd = np.array(Ed)#rangeCompress(Ed,len(Ed))
	Rm = np.array(Em)#rangeCompress(Em,len(Em))
	compare(Rd,Rm,td,tm,d,m)

	return

main()
