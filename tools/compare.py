import numpy as np
import csv
import matplotlib.pyplot as plt
import sys

endT =15 


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

	plt.subplot(1,2,1)
	plt.plot(p1,t1)
	plt.plot(p2,t2*10**6)
	axes=plt.gca()
	axes.set_ylim([-endT/16,endT])
	axes.set_xlim([np.min(p1),np.max(p1)])
	axes.set_ylabel('Time ($\mu$s)',fontsize=16)
	axes.set_xlabel('dB',fontsize=16)
	axes.set_title('Radar return',fontsize=20)
	axes.invert_yaxis()

	plt.subplot(1,2,2)
	plt.plot(m,d)
	axes=plt.gca()
	axes.set_ylim([0,np.max(d)])
	axes.set_xlim([0,np.max(m)+1])
	axes.set_ylabel('depth (m)',fontsize=16)
	axes.set_xlabel('$\epsilon_r$',fontsize=16)
	axes.set_title('Model',fontsize=20)
	axes.invert_yaxis()
	
	#plt.show()
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
