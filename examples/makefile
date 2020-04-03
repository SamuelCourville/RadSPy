download_data:
	Python ../tools/FPBPulseExtraction.py 00719502000 $(trace)

run_model:
	Python ../radar_FM1D_refMethod.py models/$(trace)/model.txt
run_model2:
	Python ../radar_FM1D_refMethod.py models/$(trace)/model2.txt

compare:
	Python ../tools/compare.py models/$(trace)/model data/FPB_719502000_$(trace)
compare2:
	Python ../tools/compare.py models/$(trace)/model2 data/FPB_804402001_$(trace)
