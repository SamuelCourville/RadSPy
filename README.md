# RadSPy
A Radar Sounding Simulator for Python. The program simulates 1D electromagnetic (radar) waves normally incident upon horizontally layered media.

## Quick use guide:

## Option A). Import radar simulator python class into code. 

Simply open the jupyter notebook file demo.ipnyb and follow instructions.


## Option B). Run radar simulator python script through terminal 

1). Open the example_model.txt file. This file shows an example subsurface
layered model to propagate your radar waves through. To run this example model,
you also need a radar source pulse. With the example model file, this repo also contains two csv
files that contain the Mars reconnaissance orbiter's SHARAD instrument's radar
pulse and it's corresponding matched filter for range compression. The example
model file references these files and uses them as the source pulse to forward model. 

2). To run the example model, run the following command: python
radar_FM1D_refMethod.py example_model.txt

3). The script will output the data as a csv file, and also a figure with the
reflected power that would be recorded back at the source. 

4). Create your own model file with the same format as the example file and
model any set of 1D layers you can think of!




