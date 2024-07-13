# LEAP-solution
My solution for the Kaggle competition of [LEAP (ClimSim)](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).  

# The steps to reproduce my solution  
## 1. Creating the datasets
### For the low-resolution dataset:
1.1. Download the data: see [here](https://www.kaggle.com/code/shlomoron/leap-download-data-1) a notebook that download 1/32 of the data. This need to be repeated 32 times by changing the index from 0 to any integer up to 31 in this line:  
'files = [all_files[i] for i in range(0, len(all_files), 32)]'  
You can acsess all the notebooks by changing the index of the notebook in the link (the number at the end of the link).  
1.2. Encode the downloaded data to TFRecords. See notebook [here](https://www.kaggle.com/code/shlomoron/leap-data-to-tfrecs-1-s). This notebook also need to be repeated 32 times. You can acsess all 32 notebooks by changing the index in the link.  
1.3. Create kaggle datasets of TFRecords by combining the output of the notebooks from 1.2. Each dataset is created from 4 notebooks, for a total of 8 dataset. See dataset creation notebook [here](https://www.kaggle.com/code/shlomoron/leap-tfrec-combined-1-s/notebook).  
