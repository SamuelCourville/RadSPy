#!/usr/bin/python3.4
#
# Import Libraries
#
import os, sys, glob, struct, csv

import sys
if sys.version_info[0] < 3:
	from HTMLParser import HTMLParser
	import urllib as url
else:
	from html.parser import HTMLParser
	import urllib.request as url

import numpy as np
import matplotlib.pyplot as plt


def downloadFPBdata(obsNum):
  basehtml = 'http://pds-geosciences.wustl.edu'
  rgramhtml = '/mro/mro-m-sharad-5-radargram-v1/mrosh_2001/data/rgram'
  geomhtml = '/mro/mro-m-sharad-5-radargram-v1/mrosh_2001/data/geom'
  gp = '/s_' + str(int(obsNum / 10000000)).zfill(4) +'xx'
  p = '/s_' + str(obsNum)[:-3].zfill(8) + '_rgram.img'
  geomp = '/s_' + str(obsNum)[:-3].zfill(8) + '_geom.tab'
  rgramURL = basehtml + rgramhtml + gp + p
  geomURL = basehtml + geomhtml + gp + geomp
  rgramOut = 'data/'+p[1:]
  geomOut = 'data/'+geomp[1:]
  print('Downloading FPB From PDS')
  url.urlretrieve(rgramURL, rgramOut)
  print('Downloading FPB GEOM From PDS')
  url.urlretrieve(geomURL, geomOut)
  return rgramOut, geomOut


def haversine(lon, lat, lons, lats, mars=True):
  #
  # Find the distance between a point and a list of coordinates
  #
  # Convert coords and coords_list to radians
  #
  if mars:
    R = 3389. #average mars radius
  else:
    R = 6371. # Earth
  lon = np.radians(lon)
  lat = np.radians(lat)
  lons = np.radians(lons)
  lats = np.radians(lats)
  #
  # Calculate distance
  #
  a = np.power(np.sin((lats - lat)/2),2) + np.cos(lats) * np.cos(lat)\
      * np.power(np.sin((lons - lon)/2),2)
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt((1-a)))
  d = R * c
  return d

def main():
  '''
  The Python program extraxcts individual pulses from decoded SHARAD data
  The default output is a CSV file with time (float) and range compressed amplitude data (complex)
  
  usage: python PulseExtraction.py {Observation Number} {latitude} {longitude} 
  The observation number needs to be the full observation number (i.e., 0577001000)
  rcTrig is optional. The default is True (i.e., return range compressed data)
			Use 'n' or if you wish to return the non-range compressed data, 
  
  This version only works for SHARAD CO-SHARPS systems.

  Written by: Matthew R Perry
  Last Updated: 18 June 2019
 
  '''
  #
  # Default Paths and variables
  #
  loc = False
  idx = True
  usage = """ There are two ways to extract an FPB Frame. The first uses the actual frame number, which can be determined through programs such as JMARS. The second is to provide this script with geographical coordinates.\n\n
To extract frame using the frame number:\n
python PulseExtraction.py {Observation Number} {frame number)\n\n
To extract fram using coordinates:\n
python PulseExtraction.py {Observation Number} {frame number)\n\n """
  sr = 0.0375
  fpbPath = '/data/{g,f}/WUSHARPS/Archive/FPB_PROD/'
  #
  # Check input arguments 
  #
  if len(sys.argv) == 3:
    print('Extracting frame based on given frame information')
    idx = True
  elif len(sys.argv) == 4:
    print('Extracting frame based on coordinates')
    loc = True
  elif sys.argv[1] == '-h' or sys.argv[1] == '--help':
    print(usage)
    exit()
  else:
    print(usage) 
    exit()

  ObsNum = int(sys.argv[1])
  #######################################################################
  #
  # FPB Data; will need to down this from the PDS since not all observations
  # have FPBs in CO-SHARPS
  #
  #######################################################################
  #
  # Check to see if file already exists
  #
  fpbFiles = glob.glob('data/s_*' + str(ObsNum)[:-3] + '*rgram*')
  
  if not fpbFiles:
    fpbFile, geomFile = downloadFPBdata(ObsNum)
  else:
    fpbFile = fpbFiles[-1]
    geomFile = glob.glob('data/s_*' + str(ObsNum)[:-3] + '*geom.tab')[-1]
  #
  # Okay, now get FPB location information
  #
  Lats = []
  Lons = []
  with open(geomFile) as F:
    for line in F:
      temp = line.strip().split(',')
      Lats.append(float(temp[2]))
      Lons.append(float(temp[3]))
  if loc:
    #############################################################
    #
    # Get latitude and longitude
    #
    #############################################################
    lat = float(sys.argv[2])
    lon = float(sys.argv[3])
    #
    # Now determine the frame number to extract from FPB
    #
    dist = haversine(lon, lat, Lons, Lats) 
    fpbframe = np.where(dist == dist.min())[0]
  elif idx:
    fpbframe = int(sys.argv[2])
    lat = Lats[fpbframe]
    lon = Lons[fpbframe]
  #
  # Print summary before processing
  #
  outFile = 'FPB_' + str(ObsNum) + '_' + str(lat) + '_' + str(lon) + '.csv'
  print('----- Summary ------')
  print('Observation File:\t{}'.format(fpbFile))
  print('Orbit File: \t{}'.format(geomFile))
  print('Latitude: \t{}'.format(lat))
  print('Longitude: \t{}'.format(lon))
  print('Frame: \t{}'.format(fpbframe))
  print('--------------------')
  #
  # Grab the FPB Frame out 
  #
  fpbData = np.fromfile(fpbFile, '<f').reshape([3600, -1])
  if fpbframe > np.shape(fpbData)[1]:
    print('Error: Frame number determined larger than the number of frames in the data ({})'.format(str(fpbframe), str(np.shape(fpbData)[1])))
    print('Exiting...')
    exit()
  fpbframe = fpbData[:, fpbframe]
  pwr = fpbframe
  idx_mx = np.where(pwr == pwr.max())[0]
  time = np.arange(0, len(pwr)) * sr - idx_mx * sr
  #
  # Format Data for easy CSV writing
  #
  #with open('data/'+outFile, 'w', newline='') as f:
  with open('data/'+outFile, 'w') as f:
    writer = csv.writer(f)
    for ii in range(len(time)):
      temp = fpbframe[ii]
      row = [time[ii], temp]
      writer.writerow(row)
  return time, fpbframe

if __name__ == '__main__':
  main()
