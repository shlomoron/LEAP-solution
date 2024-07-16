# LEAP-solution
This repository contains my solution my solution (1st place, solo gold medal) for the Kaggle competition of [LEAP (ClimSim)](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).  

# The steps to reproduce my solution  
## 1. Creating the datasets
### For the low-resolution dataset:
#### 1.1. Download the data:
See [here](https://www.kaggle.com/code/shlomoron/leap-download-data-1) a notebook that downloads 1/32 of the data. This notebook need to be repeated 32 times by changing the index from 0 to any integer up to 31 in this line:  
'files = [all_files[i] for i in range(0, len(all_files), 32)]'  
For example, for index=5, this line will read:  
'files = [all_files[i] for i in range(5, len(all_files), 32)]'  
You can access all the notebooks by changing the notebook index in the link (the number at the end of the link).  
Please note that the notebook index does not correspond to the index in the above line for shuffling reasons. For example, notebook #2 uses the index of 16.  
#### 1.2. Encode the downloaded data to TFRecords.  
See the notebook [here](https://www.kaggle.com/code/shlomoron/leap-data-to-tfrecs-1-s). There is a limit to daily public notebook publishing, so not all of them are public yet. In any case, they are all copies of the first one.  
#### 1.3. Create kaggle datasets of TFRecords by combining the output of the notebooks from 1.2.  
I created each dataset from 4 notebooks for eight datasets in total. See dataset creation notebook [here](https://www.kaggle.com/code/shlomoron/leap-tfrec-combined-1-s-public). This time, I only give a copy (not the original notebook) since this notebook prints username and Kaggle key; the latter should be kept private. In any case, the dataset creation notebook cannot be directly linked to the dataset (if I want, I can delete the dataset and create a new one with the same name from another, different notebook) so if you want to make sure I did not hide any leak in the dataset; you will have to check it manually or recreate it with this notebook (you will need to provide your own 'kaggle.json' with username and key and run eight copies of this notebook, with each copy combining four notebooks from 1.2). The dataset is public; you can see it [here](https://www.kaggle.com/datasets/shlomoron/leap-tfrecs-combined-1-s-ds).  
### For the high-resolution dataset:  
When I created the high-res dataset, I already had experience with the low-res one, so bullets 1.1-1.3 are combined into a single notebook. Here, too, I give a copy, not the original one, for the same reason I mentioned in 1.3. See the notebook [here](https://www.kaggle.com/code/shlomoron/leap-download-data-1-xx-public). I run 44 copies of this notebook, with notebook_id between 1 to 44. Each notebook downloads 5[batches]\*100[grids]\*21600[atmospheric columns] for a total of 10,800,000 samples per notebook, resulting in a dataset of \~60GB in size per notebook (and \~60*44~2.6TB in total. I wanted to download more but started to get afraid that the Kaggle system would decide that I was a bot and ban me, lol. Also, each high-res model I train uses approximately only 1TB or so of this data, so it's enough for the number of models I trained). See the first dataset [here](https://www.kaggle.com/datasets/shlomoron/leap-tfrecs-1-xx).  
## 2. Create auxiliary data sets  
There are several datasets that I need for training.  
2.1. Grid metadata. This is for latitude/longitude data of the grid indices since I use lat/lon as auxiliary loss. See notebook [here](https://www.kaggle.com/code/shlomoron/leap-grid) and dataset [here](https://www.kaggle.com/datasets/shlomoron/leap-gdata-ds).  
2.2. Mean, std, and min/max of all columns in Kaggle train and test set. Well, it is slightly more complicated since for features that differ over the 60 levels, I also calculate mean/std/min/max for all the levels combined. I  also separate between 60-level features (cols) and the features that are the same for all 60 levels (col_not). If it was not clear enough, I hope it will be after you read the code. I need mean/std data for normalization during training: (x-mean(x))/std(x), and I need min/max for log and soft clipping of the feature. Here, too, I don't give the original notebook since it is messy. Instead, [here](https://www.kaggle.com/datasets/shlomoron/leap-msm-ds) is the dataset, [here](https://www.kaggle.com/code/shlomoron/leap-msm-public) is a clean version of a notebook to create this dataset, and [here](https://www.kaggle.com/code/shlomoron/leap-msm-compare) is a notebook showing that the given notebook output the same values as in the dataset.  
2.3. Absolute min/max values for the targets. In 2.2, I calculated min/max over the Kaggle data, but later on, when I performed soft clipping on the high-res targets, I wanted to be more precise and use the absolute min/max over the complete HF low-res dataset. I find these values [here](https://www.kaggle.com/code/shlomoron/leap-y-minmax). I save them to a dataset in the next bullet (2.4).  
2.4. The normalization values of sample_submission, both old and new. I called them 'stds' since the old ones were 1/std. I save them to a dataset [in this notebook](https://www.kaggle.com/code/shlomoron/leap-sample-submission-stds), which I also use to save the min/max from 2.3 (the dataset was already in my pipeline at his point so I just piggybacked on it). The dataset is [here](https://www.kaggle.com/datasets/shlomoron/leap-sample-submission-stds-ds).  
## 3. Find gcp path for the TFRecords data.
Once upon a time, when the TPU instances in Kaggle were the old (deprecated) ones, one would have to feed the data through a GCP bucket of TFRecords. Luckily, public Kaggle dataset are stored on GCP buckets and can be accesses directly using GCP path. Nowadays it's not necessary anymore on Kaggle and you can just attach the datasets to the notebook. However, when training on massive amount on data it's still preferable to use GCP path, since attaching ~3TB to a kaggle notebook can take a very long time to download. On a side note, if you are using Colab it's still the easiest way regardless of dataset size. There is a caveat- GCP path changes every few days, so you must run the GCP-path notebooks before you start a training session, because you don't want the GCP path to change in the middle of your training (trust me). Here are the notebooks for the [low-res datasets](https://www.kaggle.com/code/shlomoron/leap-gcp-path-tfrecs-s) and the [high-res datasets](https://www.kaggle.com/code/shlomoron/leap-gcp-path-tfrecs-hr).  
## 4. Training
I used multiple data representations (i.e. instead of finding the best way to normalize the data, I normalized it in several ways and sent all the representation to the model), soft clipping for the features, soft clipping for high-res targets, MAE loss with auxiliary spacetime loss of lat/lon and sin/cos of day/year cycles. Confidense loss (the model try to predict the loss for each target). The model itself is 12-layers squeezeformer, with minor modifications (mainly those I saw in 2nd solution to Ribonanza). No dropout layers. Dimentions were 256/384/512 (I trained several slightly different models). Also, a wide prediction head (dim 1024/2048) of swish layer followed by swiGLU block. Scheduler was half-cosine decay, AdamW, max LR 1e-3 and weight decay = 4*LR. Some models were trained only on the low-res dataset, and some were trained on low-res+high-res (usually low:high had 2:1 samples ratio, but some models had slightly different ratio). The training was carrie on TPU in Colab and Kaggle, with training time of 12-48 hours for most models. Inference was done of Kaggle p100 GPUs.
