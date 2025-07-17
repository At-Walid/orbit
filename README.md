This repository contains the code and materials for the paper entitled “Surrogate Fitness Functions for DNN Testing without Ground Truth.”

## Repository contents

This repository contains the files necessary to run the fitness variants. We implemented 8 variants, 4 using pix2pixHD and 4 using CycleGAN. To run each of them we follow the following instructions.
install the required packages using the following command:
```javascript
pip install ./ORBIT/requirements.txt
```


To run one of the variants, here are the steps:

		For ORBIT variants:
			 change line 1 in main.py to: import VARIANT_NAME
			 VARIANT_NAME depends on the ORBIT variant, here is the list of variants and their respective files:
				ORBIT_flip_pix2pixHD: flip_mars
				ORBIT_flip_CycleGAN: flip_mars_cyclegan
				ORBIT_noise_pix2pixHD: noise_mars
				ORBIT_noise_CycleGAN: noise_mars_cyclegan
				ORBIT_SA_pix2pixHD: surprise_mars
				ORBIT_SA_CycleGAN: surprise_mars_cyclegan
				ORBIT_MCD_pix2pixHD: uncertainty_mars
				ORBIT_MCD_CycleGAN: uncertainty_mars_cyclegan
		For DESIGNATE:
			change line 1 in main.py to: import search2_feat_mars
		For Random:
			change line 1 in main.py to: import search_random_mars

After choosing the variant, run the following command:

```javascript
python main.py
```

The results will be saved to ./results/

## Reproducing RQ1 results
To reproduce our accuracy and diversity results, follow the instructions in the provided Jupyter notebooks as outlined below:
	- test_mars_simulator.ipynb: test the generated inputs using the DeeplabV3 DNN with the the U-test and the effect size
	- compute_diversity.ipynb: Calculate the diversity of each set with the U-test and the effect size
## Reproducing RQ2 results
To retrain DEEPLABV3 DNN using the results of one of the approaches, use the code in retrain_APPROACH.py.

