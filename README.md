# LEAP-solution
My solution for the Kaggle competition of [LEAP (ClimSim)](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).  

# The steps to reproduce my solution  
## 1. Creating the datasets
### For the low-resolution dataset:
1.1. Download the data: see [here](https://www.kaggle.com/code/shlomoron/leap-download-data-1) a notebook that download 1/32 of the data. This need to be repeated 32 times by changing the index from 0 to any integer up to 31 in this line:  
'files = [all_files[i] for i in range(0, len(all_files), 32)]'  
For example, for index=5 this line will read:  
'files = [all_files[i] for i in range(5, len(all_files), 32)]'  
You can acsess all the notebooks by changing the index of the notebook in the link (the number at the end of the link).  
1.2. Encode the downloaded data to TFRecords. See notebook [here](https://www.kaggle.com/code/shlomoron/leap-data-to-tfrecs-1-s). This notebook also need to be repeated 32 times. You can acsess all 32 notebooks by changing the index in the link.  
1.3. Create kaggle datasets of TFRecords by combining the output of the notebooks from 1.2. Each dataset is created from 4 notebooks, for a total of 8 dataset. See dataset creation notebook [here]([https://www.kaggle.com/code/shlomoron/leap-tfrec-combined-1-s/notebook](https://www.kaggle.com/code/shlomoron/leap-tfrec-combined-1-s-public).  This time I only give a copy (not the original notebook) since this notebook print username and Kaggle key, the latter should not be made public. In any case the dataset creation notebook cannot by directly linked to the dataset (if I want I can delete the dataset and create a new one with the same name from another, different notebook) so if you want to make sure I did not hide any leak in the dataset, you will have to check it manually or recreate it with this notebook (you will need to provide your own 'kaggle.json' with username and key and run 8 copies of this notebook, with each copy combining 4 notebooks from 1.2).The dataset is public and you can see it [here](https://www.kaggle.com/datasets/shlomoron/leap-tfrecs-combined-1-s-ds). 
### For the high-resolution dataset:  
When I created the high-res dataset I already had the experience from the low-res one, so bullets 1.1-1.3 are combined into a single notebook. Here too I give a copy and not the original one, from the same reason I mentioned in 1.3. See the notebook [here](https://www.kaggle.com/code/shlomoron/leap-download-data-1-xx-public). I run 44 copies of this notebook, with notebook_id between 1 to 44. Each notebook download 5[batches]\*100[grids]\*21600[atmospheric columns] for a total of 10,800,000 samples per notebook resulting in aa dataset of ~60GB in size. See the first dataset [here](https://www.kaggle.com/datasets/shlomoron/leap-tfrecs-1-xx). 
## 2. Create auciliary data sets
There are several datasets that I need for training.
2.1. Grid metadata. This is for latitude/longtitude data of the grid indices, since I use lat/lon as auxiliary loss. See notebook [here](https://www.kaggle.com/code/shlomoron/leap-grid) and dataset [here](https://www.kaggle.com/datasets/shlomoron/leap-gdata-ds).
2.2. Mean, std and min/max of all columns in Kaggle train and test set. Well, it slightly more complicated since for features that differ over the 60 levels I calculate also mean/std/min/max for all the levels combined, and is separate between 60-levels features (cols) and the features that are the same for all 60 levels (col_not). If it was not clear enough, I hope it will be after you read the code. I need this data for normalization during training: (x-mean(x))/std(x) and min/max are needed for log and soft clipping of the feature. It will become clear later in the data preparation part.

