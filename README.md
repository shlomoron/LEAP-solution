# LEAP-solution
This repository contains my solution (1st place, solo gold medal) for the Kaggle competition of [LEAP (ClimSim)](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).  

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
There are several additional datasets that I need for training.  
#### 2.1. Grid metadata.  
This is for latitude/longitude data of the grid indices since I use lat/lon as auxiliary loss. See notebook [here](https://www.kaggle.com/code/shlomoron/leap-grid) and dataset [here](https://www.kaggle.com/datasets/shlomoron/leap-gdata-ds).  
#### 2.2. Mean, std, and min/max of all columns in Kaggle train and test set.  
Well, it is slightly more complicated since for features that differ over the 60 levels, I also calculate mean/std/min/max for all the levels combined. I  also separate between 60-level features (cols) and the features that are the same for all 60 levels (col_not). If it was not clear enough, I hope it will be after you read the code. I need mean/std data for normalization during training: (x-mean(x))/std(x), and I need min/max for log and soft clipping of the feature. Here, too, I don't give the original notebook since it is messy. Instead, [here](https://www.kaggle.com/datasets/shlomoron/leap-msm-ds) is the dataset, [here](https://www.kaggle.com/code/shlomoron/leap-msm-public) is a clean version of a notebook to create this dataset, and [here](https://www.kaggle.com/code/shlomoron/leap-msm-compare) is a notebook showing that the given notebook output the same values as in the dataset.  
#### 2.3. Absolute min/max values for the targets.  
In 2.2, I calculated min/max over the Kaggle data, but later on, when I performed soft clipping on the high-res targets, I wanted to be more precise and use the absolute min/max over the complete HF low-res dataset. I find these values [here](https://www.kaggle.com/code/shlomoron/leap-y-minmax). I save them to a dataset in the next bullet (2.4).  
#### 2.4. The normalization values of sample_submission, both old and new.  
I called them 'stds' since the old ones were 1/std. I save them to a dataset [in this notebook](https://www.kaggle.com/code/shlomoron/leap-sample-submission-stds), which I also use to save the min/max from 2.3 (the dataset was already in my pipeline at his point so I just piggybacked on it). The dataset is [here](https://www.kaggle.com/datasets/shlomoron/leap-sample-submission-stds-ds).  
## 3. Find the GCP path for the TFRecords data.
Once upon a time, when the TPU instances in Kaggle were the old (deprecated) ones, one would have to feed the data through a GCP bucket of TFRecords. Luckily, public Kaggle datasets are stored on GCP buckets and can be accessed directly using the GCP path. Nowadays, it's not necessary anymore on Kaggle, and you can just attach the datasets to the notebook. However, when training on massive amounts of data, it's still preferable to use the GCP path since attaching ~3TB to a Kaggle notebook can take a very long time to download. On a side note, if you use Colab, the GCP path is still the easiest way, regardless of dataset size. There is a caveat- the GCP path changes every few days, so you must run the GCP-path notebooks before you start a training session because you don't want the GCP path to change in the middle of your training (trust me). Here are the notebooks for the [low-res datasets](https://www.kaggle.com/code/shlomoron/leap-gcp-path-tfrecs-s) and the [high-res datasets](https://www.kaggle.com/code/shlomoron/leap-gcp-path-tfrecs-hr).  
## 4. Training, inference, submit  
I used multiple data representations (i.e., instead of finding the best way to normalize the data, I normalized it in several ways and sent all the representations to the model), soft clipping for the features, soft clipping for high-res targets, MAE loss with auxiliary spacetime loss of lat/lon and sin/cos of day/year cycles. Confidence loss (the model tries to predict the loss for each target). The model is a 12-layer squeezeformer with minor modifications (mainly those I saw in the [2nd solution to Ribonanza](https://github.com/hoyso48/Stanford---Ribonanza-RNA-Folding-2nd-place-solution)). There are no dropout layers. Dimensions were 256/384/512 (I trained several slightly different models). Also, a wide prediction head (dim 1024/2048) of swish layer followed by swiGLU block. Scheduler was half-cosine decay, AdamW, max LR 1e-3, and weight decay = 4*LR. I trained some models only on the low-res dataset and some on low-res+high-res (usually low: high had a 2:1 sample ratio, but some models had slightly different ratios). The training was carried out on TPU in Colab and Kaggle, with a training time of 12-48 hours for most models. Inference was done on Kaggle p100 GPUs.  
As an example, I trained (after the competition ended and I cleaned my code) a low-res+high-res, 384 dim, four epochs model in Kaggle. This model achieves 0.79081/0.78811 public/private LB. All the models I ensembled in my winning solutions are similar, up to minor differences that I will list in the ensemble section. I did the training on Kaggle in four parts (I trained most of my models in several parts due to the Kaggle/Colab notebook's time limit).  
[Training part 1](https://www.kaggle.com/code/shlomoron/leap-training-1)  
[Training part 2](https://www.kaggle.com/code/shlomoron/leap-training-2)  
[Training part 3](https://www.kaggle.com/code/shlomoron/leap-training-3)  
[Training part 4](https://www.kaggle.com/code/shlomoron/leap-training-4)  
[Inference](https://www.kaggle.com/code/shlomoron/leap-training-4-inference)  
[Submit](https://www.kaggle.com/code/shlomoron/leap-training-4-submit)  
Note: training 2/3/4 are copies of training 1 with the following changes:  
1.  In block 16, instead of generating a random scrambling rand_seed for the high-res dataset, I change it to the seed generated in training 1. From:  

rand_seed = np.random.randint(10000)  

to:  

'''  
rand_seed = np.random.randint(10000)  
print(rand_seed)  
'''  
rand_seed = 2336  

2. In block 17, the running index start from the current_epoch-1, in order to load new high-res data. From:

for i in range(11):  

to:  

for i in range(1, 11):  

for epoch 2, range(2, 11) for epoch 3 etc.

3. In block 32, LR_SCHEDULE start from (current_epoch-1)*10 (since each epoch is divided to ten dub-epochs in the training). From:

LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50, num_training_steps = N_EPOCHS)
 for step in range(N_EPOCHS)]  

to:  

LR_SCHEDULE = [lrfn(step, num_warmup_steps=N_WARMUP_EPOCHS, lr_max=LR_MAX, num_cycles=0.50, num_training_steps = N_EPOCHS)
 for step in range(10, N_EPOCHS)]  

For epoch 2, range(20, N_EPOCHS) for epoch 3 etc.  

4. In block 37, load the last part checkpoints by:  

  checkpoint_old = tf.train.Checkpoint(model=model,optimizer=optimizer)  
  checkpoint_old.restore('/kaggle/input/leap-training-1/ckpt_folder/ckpt-1')  

  For training 2, checkpoint_old.restore('/kaggle/input/leap-training-2/ckpt_folder/ckpt-1') for training 3 atc.  

  ## 5. Ensemble  
  My winning submission was an ensemble of 13 models, 5 were trained only on the low-red sata and 8 were trained also on the high-res data. My ensemble notebook [is here](https://www.kaggle.com/code/shlomoron/leap-ensemble), and as you can see, it got 0.79410/0.79123 public/private LB scores. The ensembled prediction are [here for the low-res model](https://www.kaggle.com/datasets/shlomoron/leap-model-preds-lr) and [here for the high-res models](https://www.kaggle.com/datasets/shlomoron/leap-model-preds-hr). The models were blended with equal weigths. 
  ###5.1 Models details
  All of the models were similar to the example model I trained in bullet 4, with small changes for each model that I list in this bullet (the models are listed with the same name as in the predictions datasets linked above). Each epoch is over all HuggingFace low-res data except 200K samples that are used for local validation, and are the same validation samples in all my models. For the high-res models, each epoch includes, in addition to the low-res data, also high-res samples with 2:1 low-high ratio, except for models where I will state another ratio explicitly. The high-res data differ from epoch to epoch (as opposed to the low-res data which is repeated in each epoch). Each epoch was divided to 10 sub-epochs, and usually the last sub-epoch predictions were used, unless I state otherwise.  
  1. 0p0-all30epochs: Trained only on low-res data, 3 epochs.  
  2. 0p0-all60epochs: Trained only on low-res data, 6 epochs.  
  3. 0p0-all90epochs ensemble-2: Trained only on low-res data, 9 epochs. An equal-eweights sub-ensemble of the predictions with the model weights from sub-epoch 58 and sub-epoch 90 (last).
  4. 0p0-9epochs-E90: Trained only on low-res data, 9 epochs. Sub-epoch 90.
  5. 0p0-512dim-noSpaceTime-4epochs-E40: Trained only on low-res data, 4 epochs, model dim 512, prediction head dim 2048, no spacetime auxiliary loss, sub-epoch 40.
  6. 0p0-384dim-noSpaceTime-3epochs-E30: Trained only on low-res data, 3 epochs, no spacetime auxiliary loss, sub-epoch 30.
  7. 
