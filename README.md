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
1.  In block 16, instead of generating a random scrambling rand_seed for the high-res dataset, I changed it to the seed generated in training 1. From:  
rand_seed = np.random.randint(10000)  
to:  
'''  
rand_seed = np.random.randint(10000)  
print(rand_seed)  
'''  
rand_seed = 2336  

2. In block 17, the running index starts from the current_epoch-1, in order to load new high-res data. From:  
for i in range(11):  
to:  
for i in range(1, 11):  
for epoch 2, range(2, 11) for epoch 3 etc.  

3. In block 32, LR_SCHEDULE start from (current_epoch-1)*10 (since each epoch is divided into ten dub-epochs in the training). From:
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
  My winning submission was an ensemble of 13 models; I trained five on the low-red data, and eight on the low-res+ high-res data. My ensemble notebook [is here](https://www.kaggle.com/code/shlomoron/leap-ensemble); as you can see, it got 0.79410/0.79123 public/private LB scores. The ensemble predictions are [here for the low-res model](https://www.kaggle.com/datasets/shlomoron/leap-model-preds-lr) and [here for the high-res models](https://www.kaggle.com/datasets/shlomoron/leap-model-preds-hr). I blended the models with equal weights.  
All of the models were similar to the example model I trained in bullet 4, with minor changes for each model that I list in this bullet (I list the models with the same name as in the predictions datasets linked above). Each epoch is over all Hugging Face low-res data except 200K samples that I used for local validation and are the same validation samples in all my models. For the high-res models, each epoch includes, in addition to the low-res data, also high-res samples with a 2:1 low-high ratio. The high-res data differ from epoch to epoch (as opposed to the low-res data, which is repeated in each epoch). I divided each epoch into ten sub-epochs; I used the last sub-epoch predictions unless I stated otherwise. I give the weights used for inference [here for the low-res models](https://www.kaggle.com/datasets/shlomoron/leap-model-weights-lr) and [here for the high-res models](https://www.kaggle.com/datasets/shlomoron/leap-model-weights-hr), and for each model in the following list I link also an inference notebook. Finally, [this notebook](https://www.kaggle.com/code/shlomoron/leap-preds-validation) shows that the predictions obtained in the given inference notebooks are the same as those used in my winning ensemble.
  1. 0p0-all30epochs: Trained only on low-res data, three epochs. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-0p0-3epochs-e30-inference-p). LB 0.78895/0.78599.  
  2. 0p0-all60epochs: Trained only on low-res data, six epochs. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-0p0-6epochs-e60-inference-p). LB 0.78795/0.78388.  
  3. 0p0-all90epochs ensemble-2: Trained only on low-res data, nine epochs. An equal-weights sub-ensemble of the predictions with the model weights from sub-epoch 58 and sub-epoch 90 (last). [inference notebook for sub-epoch 58](https://www.kaggle.com/code/shlomoron/leap-0p0-9epochs-e58-inference-p), [inference notebook for sub-epoch 90](https://www.kaggle.com/code/shlomoron/leap-0p0-9epochs-e90-inference-p). LB 0.78831/0.78426.  
  4. lr-0p0-4e-512dim: Trained only on low-res data, four epochs, model dim 512, prediction head dim 2048, no spacetime auxiliary loss. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-0p0-512dim-nost-4epochs-e40-inference). LB 0.78797/0.78548.  
  5. lr-0p0-3e-384dim: Trained only on low-res data, three epochs, no spacetime auxiliary loss. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-0p0-384dim-nost-3epochs-e30-inference-p). LB 0.78944/0.78547.  
  6. hr-0p0-4e: Trained on low-res+high-res data. Four epochs. No high-res target soft clipping. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-4epochs-e40-inference-p). LB 0.79039/0.78855.  
  7. 0p0-4e-256dim-fr: Trained on low-res+high-res data. Four epochs. Model dimensions 256. High-res targets soft clipping rescale_factor = 1.2 [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-256dim-4epochs-fr-e40-inference-p). LB 0.78981/0.78642.  
  8. hr-0p0-3e: Trained on low-res+high-res data. Three epochs. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-3epochs-e30-inference-p). LB 0.79033/0.78717.  
  9. hr-0p0-7e-256dim: Trained on low-res+high-res data. Seven epochs. Model dimensions 256. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-256dim-7epochs-e70-inference-p). LB 0.78903/0.78622.  
  10. 0p0-4e-no-aux: Trained on low-res+high-res data. Four epochs. No spacetime auxiliary loss. No high-res target soft clipping. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-noaux-4epochs-e40-inference-p). LB 0.78988/0.78818.  
  11. 0p0-5e-nospacetime: Trained on low-res+high-res data. Five epochs. No spacetime auxiliary loss. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-nospacetime-5epochs-e50-inference-p). LB 0.79159/0.78869.  
  12. hr-0p0-5e-noconfidencehead: Trained on low-res+high-res data. Five epochs. No spacetime auxiliary loss. No confidence head. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-noconfidenceh-5epochs-e50-inference-p). LB 0.78945/0.78631.  
  13. hr-0p0-3e-nospacetime: Trained on low-res+high-res data. Three epochs. No spacetime auxiliary loss. [inference notebook](https://www.kaggle.com/code/shlomoron/leap-hr-0p0-nospacetime-3epochs-e30-inference-p). LB 0.79036/0.78780.  

I know that the model's names do not follow the same convention and can be a bit confusing, and I am sorry about that.  
Note: to infer new data, all you need to do is to provide the new data in the same format as the given test.csv and link to the new csv file in block 41, in the line:
test_df = pl.read_csv('/kaggle/input/leap-atmospheric-physics-ai-climsim/test.csv'

# Summary/write-up
This competition was an amazing experience and wonderfull sandbox to test ideas and experiments with varius technuqes and architectures in-depth. Thanks to Kaggle for this amazing platform, to colab for providing me cheap TPU to train on, to Google as the owner of both Kaggle and Colab and also for providing us Tensorflow. When I think about it, I trained on TPU (google) on tensorflow framework (google) on Colab (google) for a competition held on Kaggle (google). It's all Google from start to end. You are amazing and I salute you.  
Also huge thanks for the competition host for providing us this amazing competition and doing so much to keep it on track, even when facing all sort of unexpected LEAK problems.  
And thank you the Kaggle community, and especially all the people active and helpful on the forum and in the code section, you makes the experience so much better. I love you.  

I know that as 1st place, and moreover, considering the significan gap inscores between my solutiuon to the next ones, a lot of eyes would be on me to make sure I did not exploit any LEAK. Hence, I made a lot of effor to provide a complete Kagge pipeline, from downloading the data from HF, through TFRecords encoding, training, inferring and submission, with a training example that resulted in a single model 0.79081/0.78811 public/private LB. By following my pipeline ste by step, you can slso construct the dataset and train your own ~0.79+ public LB model from scratch, ensuring no hiding of leaky pseudo-labels in the train set or nefarius test-set reverse engineering in inference. All of this by simply copying and running a series of Kaggeld notebooks. Admittedly, it's a LOT of notebooks- over 100 notebooks if you intend do download end encode to TFRecords all the data yourself- but with the massive ampount of data in this competition, there is no way around it. See the complete piupeline and extra details [in my github](https://github.com/shlomoron/LEAP-solution).
## Context section
[Business context](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim).  
[Data context](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data).  
## 1. Overview of the Approach  
### 1.1. model
In one word, squeezeformer. This would not surprise anyone who followed some of the more similar competitions on Kaggle in the last year, in particular [ASLFR](https://www.kaggle.com/competitions/asl-fingerspelling) and [Ribonanza](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding). I used the same modified squeezeformer blocks that I saw in the [2nd solution to Ribonanza](https://github.com/hoyso48/Stanford---Ribonanza-RNA-Folding-2nd-place-solution) by hoyso48. The only changes I remember doing is that I deleted the first LayerNorm in the 1Dconv block and added an ECA layer. However, there may have been more minor changes that I forgot. My Tensorflow implementation was guided by hoyso48 PyTorch implementation, so once again, I give him my thanks.  
I used 12 block models with dimensions of 256/384/512. Before the squeezeformer blocks, I have a linear dense layer followed by batchNorm as an encoder from the input data to the model dimensions. After the squeezeformer blocks, I have a prediction head of swish dense followed by a GLUMlp block (swiGLU followed by linear dense), both with head dimensions of 1024/2048, depending on the model. For more details, check my code.  
At first, I used droput layers, which helped greatly when the training data was in the order of 1M samples. However, after I saw comments in the forum about dropout being unnecessary, I experimented again when I scaled up to ~40M-80M samples, and it was indeed unnecessary when ensembling (although it still allows for higher scores for single models, at the cost of twitch or trice epochs). Since removing it allows for much faster training, I also chose to drop the dropout altogether. Scheduler was half-cosine decay, AdamW, max LR 1e-3, and weight decay = 4*LR.
### 1.2. Loss  
I once read that the most important part of a DL model is the loss function. I was a bit skeptical then- sure, it is important, but it's not exactly complicated to choose the appropriate loss, right? Say, if our metric is R-squared, the best loss will obviously be...MSE?
#### 1.2.1. Use MAE  
It performs better than MSE in every way- it converges faster, is better at convergent, and is more stable. If I need to choose one 'secret' of this competition for a high score, aside from the notorious leaks, of course, it is this.
#### 1.2.2. Auxiliary loss  
We had explicit spacetime data for the train set but not for the test set. Auxiliary loss is a common practice in such cases. I used MAE on the normalized latitude/longitude and the sin/cos on the day/year cycles (if it is unclear, please look at my code).
A side not on the auxiliary loss- some people speculated, in the last week of the competition, that using explicit spacetime data, even that of the training set, is considered a forbidden leak. So, first, such use was [confirmed by the host to be legit](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/519772#2919233), as long as there is no 'hacking' of the test set (which can be done by using the auxiliary predictions to pseudo-label the test set- a thing I did not do). Second, in the last week, I trained several models without auxiliary spacetime loss, and my no-spacetime ensemble of 6 models got 0.79355/0.79092. So I win anyway; auxiliary spacetime loss is unnecessary and may not even be helpful (since my winning ensemble, while with a slightly better score, is also 13 models).  
#### 1.3.3. Confidence head  
The [3rd place solution in Ribonanza](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460403) got a suspiciously good score for a particular model. I won't explain why it was suspicious; you will need to know Ribonanza's details for the context. In any case, this suspicious model had two unique things going for it- a strange architecture and a confidence head. I still need to experiment more with said strange architecture, but for the confidense head, it was easy to try in this competition and surprisingly effective. The idea is simple- I also predict the loss of each target. I also used MAE for the confidence loss. I have one model I trained without confidence head, and it got LB 0.78945/0.78631, so I think I would also won without it. Then again, a model with the seme specifications, but with confidense head was my best model with LB 0.79159/0.78869. So, yeah. Surprisingly effective. And yes, this second model can win 1st place in this competition by itself (0.78869 private, although not 1st in public).
#### 1.3.4 Masked loss  
I masked out the targets that are zeroed in the submission or those for which we use the ptend trick (ptend_q0002_2-ptend_q0002_26).  
For those new at LEAP- please read [this post](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484) for ptend trick context.  

Final thoughts on the loss function: well, it is probably the most important part of a deep learning model. lol.  
### 1.3. Data preparation
#### 1.3.1 High-res data  
You probably have noticed already that I used high-res data. Yes, it help. Although I would have won also without it- my only low-res data ensemble of five models got LB 0.79299/0.78951. But high-res definitely helped, although it somtimes require a spaciel soft-clipping tratment as you will see. Also, training with high-res is a bit less stasble, hence I usually used larger batch sizes (1024/2048 compared to 512 for low-res only models). I blended high res data with ratio of 2:1 low:high. Lower than that the performance is weaker, higher than that the gains are small compared to extra training time required.  
#### 1.3.2 Multiple data representation
This is a trick I learned from 1st solution [at ASLFR](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485). Without going into too much details, the situation in ASLFR was that the data could be normalized in two different ways. I remember that I tried both ways, found out what is better and stuck with it. Then the competition ended, and guess what? The 1st place just normalized in the two possible ways, concatenated the two representations (with a few extra steps in between, read their summary for the full details) and sent it to the model.  
Let me first separate between the features that are spread ove the 60 height level, whi9ch I call X_col, and the features that are the same for all the levels, which I call X_col_not.  
For X_col_not, I used only one representation, which is the somple normalization (x-mean)/std.   
For X_col, I used three representations. The first is norm_1, the same like X_col_not, where each feature in each level is normalized with its own mean/std. i.e., for state_t, then we have (state_t_1-mean(state_t_1))/std(state_t_1), (state_t_2-mean(state_t_2))/std(state_t_2) etc. In my code, I call this representation x_col_not_norm (for x_col_not) and x_col_norm (for X_col).  
The second representation normalize each feature by the total mean and std over all the levels. i.e., for state_t, then we have (state_t_1-mean(state_t))/std(state_t), (state_t_2-mean(state_t))/std(state_t) etc. In my code, I call this representation x_total_norm.  
Finally, the third representation is:  

```  
x_col_norm_log = tf.where((x_col_norm-x_col_norm_min+1)>=1, tf.math.log(x_col_norm-x_col_norm_min+1),
                                    -tf.math.log(1+1-(x_col_norm-x_col_norm_min+1)))
```

Which is the kisd of thing that trying to explain with word would never be as clear as just looking at the code.  
After you looked at the code and understood it, you maybe wonder why not use just:  

```  
x_col_norm_log = tf.math.log(x_col_norm-x_col_norm_min+1)
```

Why all the extra step with the tf.cond? See, I had a problem. I calculated x_col_norm_min only with Kaggle data, and then I scaled up my code to all HF data (which have values lower than Kaggle data x_col_norm_min) but did not wanted to change the normalization constant because it break inference pipeline, and then I would have to use a different pipelines for my old models and new  models. Yeah sometimes I'm a bit lasy. Proud of it. And it turned out to be an excellent chpice when I included also high-res data.  
#### 1.3.3 Wind
$wind = \sqrt{(state_u)^2+(state_v)^2}$  
It just made sense and I wanted to include at least one 'physically justified' thing in the model (silly me, yes). It did not really helped but also did not hurt the model, so it stayed. I used only the first normalization for WIND, with:  
mean(wind) = mean(mean(state_u), mean(state_v))  
and with:  
std(wind) = sum(std(state_u), std(state_v)  
All in all, the feature dimension is:  
9[col_features]/*60[levels]*3[representations]+60[wind_levels]+16[not_col features] = 1696  
#### 1.3.4 Features soft clipping 1  
After normalization, the data has some extreme values (~±3000). This problem exists only for the first normalization. The model actually handled it easily, but I preferred to play on the safe side. So, for x_col_norm and for WIND, I applied the following soft clipping:  

```  
cutoff = 30
square_cutoff = cutoff**0.5
x_col_norm = tf.where(x_col_norm>cutoff, x_col_norm**0.5+cutoff-square_cutoff, x_col_norm)
x_col_norm = tf.where(x_col_norm<-cutoff, -tf.math.abs(x_col_norm)**0.5-cutoff+square_cutoff, x_col_norm)
```

#### 1.3.5 Features soft clipping 2  
In addition to the first soft clipping, I applied a second soft-clipping to deal with extreme values from the high-res set. This clipping was applied on all the representation, including WIND, and after the first soft clipping (1.3.3) was applied for the relevant features:  

```
cutoff_2 = 86.0
log_cutoff = tf.math.log(cutoff_2)
x_col_norm = tf.where(x_col_norm>cutoff_2, tf.math.log(x_col_norm)+cutoff_2-log_cutoff, x_col_norm)
x_col_norm = tf.where(x_col_norm<-cutoff_2, -tf.math.log(-x_col_norm)-cutoff_2+log_cutoff, x_col_norm)
```

cutoff_2 was chosen to be such that the soft clipping would affect only high-res data.  
#### 1.3.6 Targets soft clipping  
This was done only for the high-res targets, each target was soft clipped if it was too extreme compared to low-res corresponding target min/max (this differ from 1.3.4/1.3.5 in that the soft-clipping range was different for each rarget). In code:  

```  
rescale_factor = 1.1
if x['res'] == 0:
    y_norm = tf.where(y_norm<norm_y_min*rescale_factor, norm_y_min*rescale_factor-tf.math.log(1-y_norm+norm_y_min*rescale_factor), y_norm)
    y_norm = tf.where(y_norm>norm_y_max*rescale_factor, norm_y_max*rescale_factor+tf.math.log(1+y_norm-norm_y_max*rescale_factor), y_norm)
```
### 1.4 Post-processing  
#### 1.4.1. Downcast and Upcast  
A speciel care should be taken when movig between FP64 and FP32 and vice versa. My TFRecords were encoded with the original values in FP64. After I processed the data (see 1.3) the values were downcast to FP32 before I transferred them to the model. Then I upcasted the predictions to FP64, and only then I apply de-normalization in order to get the values for submission:  

```
preds = preds + mean_y.reshape(1,-1)*stds
preds[:, np.where(stds_new == 0)] = 0
preds = preds/np.where(stds>0, stds, 1)
```

#### 1.4.2. Mean for bad targets  
This is very simple:  

```
metrics = np.asarray([sklearn.metrics.r2_score(val_labels[:, i], preds[:, i]) for i in range(368)])
for i in range(len(metrics)):
    if metrics[i]<0:
        preds[:,i] = 0
```

In reality it was eventually unnecessary, because the only bad targets were among those zeroed out in the submission or in the ptend trick range.

#### 1.4.3. Ptend trick  
Obviously. If you are new at LEAP, look [here for details](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484).  

## 2. Details of the submission  
### 2.1 Ensembling  
My winning ensemble included 13 models, each a bit different (see full details in my github, 'The steps to reproduce my solution' bullet 5). The best model (11) was LB 0.79159/0.78869, the worst (2) was LB 0.78795/0.78388. Both were best/worst both ob public and on private LB. The full ensemble was LB 0.79410/0.79123. In addition, my low-res-data only ensemble of 5 models have LB 0.79299/0.78951, and my no-spacetime-auxilliary-loss ensemble of 6 models have 0.79355/0.79092. When you read my solution, you maybe tried to find out the 'secret sauce' that got me the 1st place, but it really was the combinations that made the difference. Every single technuqe that I used, I think I could still get to 1st place without it.  
### 2.2 The helpful techniques
This is a short summary of the methods I wrote about already in depth above. Squeeseformer, wide GLUMlp prediction head, no dropout, MAE, auxiliary timespace loss, confidence head, masked loss, multiple data representation, high-res data, features and targets soft-clipping,careful downcast/upcast.
### 2.3 What didn't work  
Various model architectures (pure transformer, other conv/transformer combinations, Unet, droput, smaller models, larger models, other optimizers) in short, a lot og hyper-parameters that were less optimal. Log-normalization (i.e. log(x), not my log(1+x) representation which deal with different issues). MSE, MSE/MAE varius combination, weighted loss function (check my Ribonanza solution for details), probaly other not-very-important things that I don't remember already.  
### 2.4 Hardware
I trained on Kaggle and Colab TPU, and inferred on Kaggle P100 gpu. Compute-wise, my experiments and training are at least 200 bucks in Colab compute units, and probably less than 300 bucks in total. If it sounds like a lot compared to your 'free' personal machine, please consider the electicity cost of using an RTX4090...

## 3. Sources
This is me favorite part, a bit thank you for all the recources that helped me!  
[Ptend trick](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484) and also [here originally](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/499896#2791290).  
[Ribonanza 2nd solution by Hoiso48](https://github.com/hoyso48/Stanford---Ribonanza-RNA-Folding-2nd-place-solution) for Squeezeformer architecture guidance and insights.  
[Ribonanza 3rd place solution](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding/discussion/460403) for confidence head method.  
[ASLFR 1st solution](https://www.kaggle.com/competitions/asl-fingerspelling/discussion/434485) for multiple data representation method.  
[Dropout is unnecessary](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/514020#2884414) and [also here](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/514020#2885043).  

In addition, I used the low-res and high-res data [from HF](https://huggingface.co/LEAP).  
