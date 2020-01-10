---
layout: git
title: Denoising with Generative Models
published: true
description: Semester Project - EPFL
github: 'https://github.com/Billotais/Denoising-with-Generative-Models'
---
Semester Project by [Loïs Bilat](mailto:lois@bilat.xyz) at VITA Lab - EPFL - Fall 2019 

Supervised by [Alexandre Alahi](mailto:alexandre.alahi@epfl.ch) and [Brian Sifringer](mailto:brian.sifringer@epfl.ch).

**NOT FINISHED YET, MISTAKES TO BE EXPECTED**



# Table of Contents
1. [Introduction](#introduction_)
2. [Model](#model_)
	- [Original Architecture](#original_model_)
    - [GAN](#gan_)
    - [Auto-encoder](#autoencoder_)
    - [Collaborative GAN](#collaborative_gan_)
    - [Conditional GAN](#conditional_gan_)
3. [Code](#code_)
4. [Experiments](#experiments_) 
5. [Potential Improvements](#potential_improvements_) 
6. [Conclusion](#conclusion_) 
7. [Sources](#sources_) 

<a name="introduction_"></a>



# Introduction

**Description of the problem**

The quality of audio recordings from a mobile device has gotten better over the years, but there are still a lot of factors that can decrease the quality. Among others, the size and the quality of the microphone sensor, as well as its location relative to the audio source can have a non-negligable impact. We also cannot forget potential ambient noise (e.g. voices, rain, traffic) and reverberation that can be cause by the size and shape of the room.

It would be usefull if we could somehow correct all those issues by using software tools that could take a low quality audio file, and improve it as if the audio was recorded using high quality equipement in a perfectly silent environement. More precisly, given a music sample of any length and recorded using low quality equipment in a noisy environment (therefore it might have a low resolution, some noise and some reverberation), we want to output a higher resolution version of the same audio sample, with some of the noise and reverberation removed. If the resulting music file sounds better to the human ear than the original, the transformation is considered successful. 

**Why is it important ?**

If we want high-quality recording on our mobile devices, we need some software solutions. as we might not be able to improve the hardware quality of the microphone due to physical limitations. It is also hard to control the environment where we want to do our recording. This type of technology could then be used by smartphone manufacturers to let the users create studio-grade quality recordings.

Moreover, If we are able to improve the quality of an audio signal, we might also be able to improve the quality of other types of signal (e.g. an electromagnetic signal). 

For instance, it could be used to improve the precision of the LIDAR technology that can be very useful for autonomous cars.

**Goal of the project**

Most of the research on audio denoising is done on speech, as it is probably the domain where such techniques might be the most usefull (intelligent assistants, speech-to-text, etc). Therefore, there are very few papers that tried these techniques on music, and when they did it often was as an afterthought, and not the primary goal of their project.

The goal of this project was to study the state of the art in audio denoising and super-resolution, and then improve it by adding some other ideas coming from image processing research. At the time of the start of the project, the state of the art in speech super-resolution seemed to be _Speech Denoising with Deep Feature Losses_ (complete reference at the end), and the state of the art for speech denoising seemed to be _Adversarial Audio Super-Resolution with Unsupervised Feature Losses_. The plan was to first implement this second paper as a baseline, and then try to improve it. Unfortunatly, it proved more challening than expected to implement this model, as some details were not clear enough to implement it correctly. 

Therefore, the paper _Audio Super-Resolution using Neural Nets_ was chosen as a baseline for this project, as the paper was very clear and a github repository was also available for more details. 



<a name="model_"></a>

# Model

The model proposed here is a convolutional autoencoder with skip connections, also known as *U-net*. Starting from this base model, a few improvements were then added, such as a discriminator network to transform the model into a GAN and another autoencoder to further improve the learning process.

After this, techniques like Collaborative GAN, Conditional GAN **and Patch GAN** were also implemented.

<a name="original_model_"></a>

## Original model

**Architecture**

The original architecture is, as mentionned before, a convolutional autoencoder with skip connections. It consists of $B$ downsampling blocks, one bottleneck block, $B$ upsampling blocks and a final convolutional layer. There are stacking residual connections between a downsampling and an upsampling block at the same level, and an additive residual connection between the input and the final block.

![architecture.png]({{site.baseurl}}/img/vita/architecture.png)


Each domnsampling block consists of a convolutional block, and a ReLU block. These have a stride of 2, the number of channels outputed by each convolutional block is given by the array `[126, 256, 512, 512, 512, ...]`, and the size of filters is given by the array `[63, 33, 17, 9, 9, 9, ...]`. The ReLU block is more precisely a Leaky rectified linear block with a slope of 0.2 on the negative side.

The bottleneck block is the same as a downsampling block, but with a dropout with probabilty 0.5 after the convolutional layer.

In the upsampling blocks, the convolutional layer uses the same filter sizes as the downsampling blocks, but in reversed order. The number of channels outputed by the convolution is double the one in the corresponding downsampling block, and we have a stride of 1. We then have a dropout of 0.5 and a LeakyReLU with a slope of 0.2. Following this, we have the *DimShuffle* operation, more precisely the Sub-pixel operation, that takes some data of shape $N\times C \times \ W$ and transform it into data of shape $N\times C/2 \times \ 2W$ by interleaving elements from two channels together. 


![Subpixel operation]({{site.baseurl}}/img/vita/subpixel.png)

Finally, we have the stacking block that takes the output of the corresponding downsampling block and concatenates it on the channel dimension with the output of the sub-pixel block.

After all the upsampling blocks, we finish with a final block that makes of convolution with 2 outputs channels, a filter size of 9 and a stride of 1, then the subpixel operation (this will reshape our data so that we are left with only one channel, and then we add the output of this to our input data to get the output of the network (this is the additive skip connection). What this connection implies is that the netork doesn't learn how to create denoised sound directly, but rather it learns how to create "denoising data" that when added to the audio itself improves it.

To better understand this architecture, you can see here a schema of the network 


![Detailed architecture]({{site.baseurl}}/img/vita/detailed.png)


and you can see how the stacking connections are used. In the upsampling block, the goal of the convolution is to merge the data from the previous upsampling block and from the corresponding downsampling block, whereas the sub-pixel operation's goal is simply to reshape it so that it has the correct shape to be concatenated later.

Moreover, you can see here for a toy example with depth 4 and where inputs have shape (1, 1024), what are the sizes of each layer and the parameters of the convolution. 

```
           input : (1, 1024) -----------------------------------------
                                                                     |   
                                                                     |
                                                                     |
        (1, 1024)->conv(128,63,2)->(128, 512)-------------------     |
                                                               |     |
                                                               |     |
                                                               |     |
      (128,512)->conv(256,33,2)->(256, 256)---------------     |     |
                                                         |     |     |
                                                         |     |     |
                                                         |     |     |
    (256,256)->conv(512,17,2)->(512, 128)----------      |     |     |
                                                   |     |     |     |
                                                   |     |     |     |
                                                   |     |     |     |
  (512,128)->conv(512,9,2)->(512, 64)---------     |     |     |     |
                                             |     |     |     |     |
                                             |     |     |     |     |
                                             |     |     |     |     |
(512,64)->conv(512,9,2)->(512,32)            |     |     |     |     |
                                             |     |     |     |     |
                                             |     |     |     |     |
  (512,32)->(512x2,9)->(1024,32)-sp->(512,64):     |     |     |     |
                                             =     |     |     |     |
                                         (1024,64) |     |     |     |
                                                   |     |     |     |
    (1024,64)->(512x2,17)->(1024,64)-sp->(512,128)-:     |     |     |
                                                   =     |     |     |
                                               (1024,128)|     |     |
                                                         |     |     |
      (1024,128)->conv(256x2,33)->(512,128)-sp->(256,256):     |     |
                                                         =     |     |
                                                     (512,256) |     |
                                                               |     |
        (512,256)->conv(128x2,63)->(256,256)-sp->(128.512)-----:     |
                                                               =     |
                                                           (256,512) |    
                                                                     |
           (256,512)->conv(2,9)->(2,512)-sp->(1, 1024)---------------+
                                                                     =	
                                                                 (1, 1024)
```


**Training**

We train this network using the $L2$ loss

$$\mathcal{L}_G = \mathcal{L}_{L2} = \frac{1}{W}\sum_{i=1}^W \left\|x_{h,i} - G(x_l)_i\right\|$$

where $x_h$ is the high quality audio signal, and $x_l$ the low quality signal.

It is also possible to use the $L1$ loss

$$\mathcal{L}_G = \mathcal{L}_{L1} = \frac{1}{W}\sum_{i=1}^W |x_{h,i} - G(x_l)_i |$$

by simply changing a parameter in the command.

<a name="gan_"></a>

## GAN

To transform our system into a Generative Adversarial Network (GAN), we need to add a discriminator network, whose goal is to classify given samples as *real* or *fake*. In our case, we want that the original high quality audio should be classify as *real*, and the genereated improved audio should be classified as *fake*. The goal of our first network, called generator here, is to create improved samples that will be classified as *true* by the discriminator.

The architecture of the discriminator is basically the first half ot the generator network, with a Batch normalization layer added between the convolutional and ReLu layers. At the end, everything is sent into a linear layer and a sigmoid activation function that will input one value between 0 and 1, the probability that a given sample is *real*. This Discriminator is trained with the following loss function

$$\mathcal{L}_D = - [\log D(x_h) + \log(1-D(G(x_l)))]$$

When training the model, we first train the generator and discriminator separatly for a while, and once the loss of the discriminar is low enough, we change the generator loss $L_G$ to

$$\mathcal{L}_G = \mathcal{L}_{L2} + \lambda_{adv}\mathcal{L}_{adv}$$

with

$$\mathcal{L}_{adv} = - \log D(G(x_l))$$

meaning that the generator will not only look at its own loss (i.e. how far are we from the target sample), but it will also try to generate more realistic samples to fool the discriminator.

<a name="autoencoder_"></a>

## Autoencoder

To improve the model further, we added another network, with an autoencoder architecture, that will also contribute to the loss function of our generator by computing the distance between our generated and target data, but in the latent space created at the bottleneck of this autoencoder. The architecture is the same as the generator, but the residual connections are removed (and therefore some parameters for the number of channels and filters and adapted consequently).

This autoencoder is trained using the $\mathcal{L}_{L2}$ loss, on the *identity task* (meaning that the target is the same as the input). The goal of this is to find a lower dimnesion representation of our data (at the bottlneck), that can give some useful information to our optimisation problem.

The new loss for the generator is now 

$$\mathcal{L}_G = \mathcal{L}_{L2} + \lambda_{adv}\mathcal{L}_{adv} + \lambda_f \mathcal{L}_f$$

with 

$$\mathcal{L}_f = \frac{1}{C_f W_f} \sum_{c=1}^{C_f} \sum_{i=1}^{W_f} \left\| \psi (x_h)_{i.c} - \psi(G(x_l))_{i,c}\right\|$$

Where $\psi(x)$ is the output of the network at the bottlneck layer, and $C_f$ and $W_f$ are the number of channels and the width of the data at the bottleneck

<a name="collaborative_gan_"></a>

## Collaborative GAN

In a normal GAN architecture, once the model is trained, we only use the generator to create our final samples. Collaborative GAN is a technique where the discriminator is also used during the sampling phase, to hopefully get better results. Collaborative GAN is used only once the original training of the model is completly done. I will present here the modified version of this method, adapted from the original paper.

There are two main part to this method :

### Collaborative sampling

This step happens when you want to generate a sample, but also want to improve it using information provided by the discriminator. Since this happens after the training, we want to freeze the parameters of both the generator and the discriminator.

Suppose we have an input sample, i.e. a low quality audio sample. We send it as an input to the generator, to generate an improved version. We then take the output, and give it to the discirminator. If the discriminator classifies it as "real" with a high enough confidence (in the code *0.5* is used, but higher values may be required for better results), we stop, as we consider our sample is good enough. However, if our sample is classified as "fake", we will try to improve it. This is illustrated here:

![Collaborative GAN]({{site.baseurl}}/img/vita/collab_gan.png)

For this, we will look at the internal activations of the generator network, in particular the activations at one layer (we call it $x_l$). $x_l$ corresponds to the activation at the output of the l$l$th upsampling block of our generator. We then compute the gradiant of this tensor of activations relative to the loss of the discriminator, and then update the value of $x_l$ by gradiant descent. The values of $x_l$ are then once again propagated to the end of the network, which will give us a new sample. (Note that during this second pass with the modified $x_l$, the values provided by the skip connections come from the original propagation of x. 

Finally, we repeat this while our sample is not classified as "real" by the discriminator. We nevertheless stop after a maximum number of iterations to not spent to much time with problematic samples.

To sum up : 

- Compute $y = G(x)$, a generated sample
- Compute $p = D(y)$, the confidence of the sample being "real"
- If $p<\eta$ :
	- Compute the gradiant $\frac{d \mathcal{L}_D}{d x_l}$, where $x_l$ is the internal activation at layer $l$ of the upsampling blocks.
    - Update $x_l$ by gradiant descent, i.e. $x_l = x_l - \lambda \frac{d \mathcal{L}_D}{d x_l}$
    - Do another forward pass $G(x)$, but when reaching layer $l$, replace the values by the new value of $x_l$.
    - Continue the forward pass until we have a new generated sample.
    - If it is still classified as "fake", and we have not reached the maximum number of iterations, repeat.
- If $p>\eta$ : we have our sample, nothing else to do.

This whole algorithm is done at the complete end of our pipeline, when we generate samples for our output audio file. 

However, our discriminator, despite being able to classifiy samples as real and fake, it is not really capable of helping the refining process. For this reasoon, we have to train it beforehand. This step is called discriminator shaping.

### Discriminator shaping

Discriminator shaping happens after the normal training, but before the generation of the final audio file using collaborative sampling. I works as follows :

For a given number of iterations, and given a pair $(x_l, x_h)$ of low quality and high quality audio, we train the discriminator with $x_h$ given as "real" data, and the collaboratively sampled data generated from $x_l$ as "fake" data.

<a name="conditional_gan_"></a>

## Conditional GAN

Another technique that can be used to improve the results of our model is conditional generative adversarial network (also known as CGAN). This technique is for instance used by [pix2pix](https://phillipi.github.io/pix2pix/), where the goal is to transform an image into another (often starting from a schematic images, i.e. only the borders, or maybe a segmentation map, and trying to create a realistic image correspondign to this input). 

This was implemented in this project since this idea (image to image instead of random latent data to image) is actually guite similar to what we are trying to do here. Indeed, we already start from an image, and therefore this model might be more appropriated that the usual generator model.

The basic idea of CGAN is this: You take the same model as for our previous GAN (a generator, and a discriminator that takes improved samples and classifies them as *real* or fake*), but you edit the discriminator so that it takes both the improved data, but also the original data as an input (WE simply concatenate them on the channel dimension). 

This change means that now the goal of the discriminator is to ask "Is this a real-sounding improved sample, given what the original data is ?", instead of only "Is this a real-sounding improved sample ?" (therfore the name *conditional* GAN).

You can see here an illustration of this change, taken from the original pix2pix paper.

![Conditional gan]({{site.baseurl}}/img/vita/cgan.png)

Here they use images, but the idea is the same. Your input x (the drawn shoe, respectively the low quality audio file) is given to the generator to create an improved version (the grey shoe / improved audio). Then, you give both the generated sample (grey shoe / improved audio) and the original data (drawn show / low quality audio) to the discriminator). Moreover, when training the discriminator, you will also give two samples as the input, namely the original image (drawn shoe / low quality audio) and the target image (brown shoe / high quality audio).

The architecture for this conditional discriminator is exactly the same as the original discriminator, with simply a different input shape. The rest of the network doesn't change.


# Preprocessing

Since this model should work with any audio file provided, we need to do some preprocessing beforehand. There are two parts for this. 

## Audio degradation

Since the goal of this project is to improve audio quality, we need to create our dataset. We use the MAESTRO Dataset (as described later), which constists of only high quality audio files. We therefore need to degrade those files to have our training data pairs. Three type of pre-processing are supported; Noise, Downsampling and reverberation. All of those are done using the `sox` library, and can be chained together.

For the downsampling, we have two arguments, *target_res* and *input_res*. If we name our pair of input/output data samples (x, y), we can create y by simply downsampling our audio to *target_rate*. To create x, we first downsample our data to *input_res*, and then upsample it to *target_rate* using simple linear interpolation. We do it thi way so we can have a symetrical network, which is necessary for our skip connections to work properly.

## Audio split

Since our network takes as input some data with a given size, we need to split our original audio file into mutliple small segments. We do this using a sliding window with a stride (usually a window of 2048 and a stride of 1024). This gives us some redundency in the data, and this way any part of the music is seen twice, at a different postion in the input. 

During the evaluation phase, when we want to reconstruct an audio file, we need to put samples of the music back together. The naive way would be to split the audio file using a stride of the same value as the window size (i.e. no overlap), improve each sample individually, and then concatenate all the blocks together. This, however, doesn't give us good results, and we can hear a distincitve noise at the junction between two samples. This happens because our network handles the border of a sample differently than data in the middle of the sample (we have to use some padding on the border, so the sound is distorded).

Therefore, for the evaluation phase, we still split the data using a sliding window with some overlap. When putting the blocks together, we will only keep a part of each samples, and the borders will be croped. This can be seen in the following illustration.

![Illustration of audio reconstruction]({{site.baseurl}}/img/vita/merge.png)


With this technique, we don't have any audio artefact at the junction between two samples.

<a name="code_"></a>

# Code

You can find on this [github page](https://github.com/Billotais/Denoising-with-Generative-Models) all the code used for this project. Required libraries are the following 

```
matplotlib
torch
torchaudio
numpy
pandas
graphviz
```

and can be found in `requirements.txt`. `pandas` and `graphviz` are not required in a standard execution of the code.

## How to run
```
usage: main.py [-h] [-c COUNT] [-o OUT] [-e EPOCHS] [-b BATCH] [-w WINDOW]
               [-s STRIDE] [-d DEPTH] -n NAME [--dropout DROPOUT]
               [--train_n TRAIN_N] [--test_n TEST_N] [--load LOAD]
               [--continue CONTINUE] [--dataset DATASET]
               [--dataset_args DATASET_ARGS] [--data_root DATA_ROOT] --rate
               RATE --preprocessing PREPROCESSING [--gan GAN] [--ae AE]
               [--collab COLLAB] [--lr_g LR_G] [--lr_d LR_D] [--lr_ae LR_AE]
               [--scheduler SCHEDULER]

optional arguments:
  -h, --help            show this help message and exit
  -c COUNT, --count COUNT
                        number of mini-batches per epoch [int], default=-1
                        (use all data)
  -o OUT, --out OUT     number of samples for the output file [int],
                        default=500
  -e EPOCHS, --epochs EPOCHS
                        number of epochs [int], default=10
  -b BATCH, --batch BATCH
                        size of a minibatch [int], default=32
  -w WINDOW, --window WINDOW
                        size of the sliding window [int], default=2048
  -s STRIDE, --stride STRIDE
                        stride of the sliding window [int], default=1024
  -d DEPTH, --depth DEPTH
                        number of layers of the network [int], default=4,
                        maximum allowed is log2(window)-1
  -n NAME, --name NAME  name of the folder in which we want to save data for
                        this model [string], mandatory
  --dropout DROPOUT     value for the dropout used the network [float],
                        default=0.5
  --train_n TRAIN_N     number of songs used to train [int], default=-1 (use
                        all songs)
  --load LOAD           load already trained model to evaluate [bool],
                        default=False
  --continue CONTINUE   load already trained model to continue training
                        [bool], default=False, not implemented yet
  --dataset DATASET     type of the dataset[simple|type], where 'type' is a
                        custom dataset type implemented in load_data(),
                        default=simple
  --dataset_args DATASET_ARGS
                        optional arguments for specific datasets, strings
                        separated by commas
  --data_root DATA_ROOT
                        root of the dataset [path], default=/data/lois-
                        data/models/maestro
  --rate RATE           Sample rate of the output file [int], mandatory
  --preprocessing PREPROCESSING
                        Preprocessing pipeline, a string with each step of the
                        pipeline separated by a comma, more details in readme
                        file
  --gan GAN             lambda for the gan loss [float], default=0 (meaning
                        gan disabled)
  --ae AE               lambda for the audoencoder loss [float], default=0
                        (meaning autoencoder disabled)
  --collab COLLAB       Enable the collaborative gan [bool], default=False
  --lr_g LR_G           learning rate for the generator [float],
                        default=0.0001
  --lr_d LR_D           learning rate for the discriminator [float],
                        default=0.0001
  --lr_ae LR_AE         learning rate for the autoencoder [float],
                        default=0.0001
  --scheduler SCHEDULER
                        enable the scheduler [bool], default=False

```

For instance, the following command

```
main.py --count -1 --out 1000 -e 10 --batch 32 --window 2048 --stride 1024 \\
        --depth 8 --train_n -1  --name gan_10 --data_root /data/lois-data/models/maestro \\
        --rate 10000 --lr_g 0.0001 --lr_d 0.0001 --gan 0.0001 \\
        --preprocessing "sample 5000 10000"
```

will run the model for 10 epochs, using minibatches of 32 samples. The network will have a depth of 8, and we split the data into sub-samples of 2048 of width, and some overlap (stride of 1024). We train on all avalailable training data, but create an improved version for only one output file. Our data is stored in the `/data/lois-data/maestro` folder. Our target rate is 10kHz, the learning rate for the generator and the discriminator is 0.0001, the lambda used for the discriminator part of the composite loss is 0.0001. Finally, we want to do some  upsampling from 5kHz to 10kHz.

Note that you need to create a `out` folder inside then `src` directory. All the results will be saved inside.

### Preprocessing

You can apply the following types of preprocessing by putting the followijng arguments as a string for the `--preprocessing` option: 

- Downsample : `"sample low_res high_res"`, if you want your input data to have a resolution of `low_res` Hz, and the target data to have resolution `high_res` data. Do not forget to also put the value of `high_res` for the `--rate` argument.
- Noise : `"noisetype variance volume"`, where `noisetype` can be among `["whitenoise", "pinknoise", "brownnoise", "tpdfnoise"]`, `volumne` the intensity of the noise (good values are around 0.001). You can add some variance if desired, so that for each audio file the noise level will be a little bit different.
- Reverberation : `"reverb variance reverberance hf_damping room_scal stereo_depth pre_delay wet_gain=0"` to apply some reverberation with various parameters. Default values can be found in the code.

You can also apply different preprocessing one after the other, by concatenating the commands with a comma in between, e.g. `--preprocessing "sample 5000 10000,whitenoise 0 0.002"`

### Evaluation 

By default, once the model has finished training, it will take a file and try to improve it. This file will be `out.wav`. If you want to evaluate other files, you can run the original command, but with the `--load filename` argument added. This will look for a model save file `out/name/models/model.tar` (don't forget to rename the model file you want to use).
 
## Code structure

- `main.py` 
	- `init()` : Contains the parsing of the arguments, 
    - `init_net()` : Initialize the generator, discriminator and autoencoder networks
    - `load_data()` : preprocesses the datafiles, load them into memory and return dataloaders
    - `pipeline()` : call above functions, then train/evalutate the model. Create output file.
- `train.py` 
	- `train()` : All the code to trian the model, tracks the evolution, draw some plots, etc
    - `make_train_step()` : used by `train()` to define one step of the training process.
- `test.py` 
	- `test()` : All the code to test the model, tracks the evolution, draw some plots, etc
    - `make_test_step()` : used by `test()` to define one step of the testing process.
- `network.py` 
	- `class Generator` : Implementation of the Generator Network
    - `class Discriminator` : Implementation of the Discriminator Network
    - `class AutoEncoder` : Implementation of the AutoEncoder Network
    - `class ConditionalDiscriminator` : Implementation of the Conditional Discriminator network.
    - other classes are juste submodules fused by the other main modules.
- `files.py` 
	- `class SimpleFiles` : simple representation of a dataset, with methods to get train/test/val data
    - `class MAESTROFiles` : specific representation for the MAESTRO Dataset, with methods to get train/test/val data
- `datasets.py`
	- `class AudioDataset` : Pytorch dataset class that takes a filename and loads it as a pair `(input,target)` into memeory
- `preprocessing.py`
	- `preprocess()` : Given a file name and a preprocessing chain, returns pair of file names corresponding to the outputs of the preprocessing.
    - `sample, noise, reverb` :  implementation of the options accepted by the preprocess chain
    
- `utils.py` : various functions used in multiple places in the code




<a name="experiments_"></a>

# Experiments

**MAESTRO Dataset**

This dataset consists of more than 200 hours of recorded piano, in high quality (44.1kHz), and in the wav format. Each file is also availible in MIDI format, but those were not used as we wanted to use some more realistic data. Due to the huge amount of data, no all of it was used (only ~3.3GB).

It was the highest quality data that we found, but it had the problem of having a little bit of background noise (since it was recorded in a concert hall, you can hear poeple mobing chairs and so on). What could maybe be done to not have this noise in the high quality data is to generate realistic piano sound from the midi file (either artificially, or even with one of those piano that play midi files for you), but it is clearly out of scope for this project.

**Metrics**

To measure the performance of the model, the log-spectral distance (LSD) is used. This is defined as follows

$$LSD(x,y) =  \frac{1}{L} \sum_{l=1}^L \sqrt{\frac{1}{K} \sum_{k=1}^K (X(l,k) - \hat{X}(l,k))^2}$$

where $X$ and $\hat{X}$ are the log-spectral power magnitudes of $y$ and $x$, respectively. These are defined as $X=\log{\left\| S\right\|^2}$, where $S$ is the short-time Fourier transform (STFT) of the signal. $l$ and $k$ are used to index frames and frequencies, repectively. For the following results, frames of length 1024 were used.

## Results

For the following experiments, as there are a lot of paramters, only a few of them are changed while the others stay fixed. 

For all the experiment, we do super-resolution from 5kHz to 10kHz, with no noise added. Approximatly 1.4 GB of raw data was used, corresponding to a little over 2 hours of music. 

For each experiment, the data is split into samples of 4096 of length, with an overlap of 2048. The main network is initialized with a depth of 8 (i.e. 8 downsampling blocks and 8 upsampling blocks). The discriminator and the autoencoder will have the same depth aswell. Each experiment is for 10 epochs, using mini-bacthes of size 32. For the generator, the L2 loss is used by default. 

For all the experiments, the same file is used as the "test data" to create the plots that will follow. Moreover, the measure the metrics, another file is used, but once again always the same. 



<p>Low quality audio</p>
Low quality audio, i.e. what we want to improve, the input
<audio controls>
  <source src="https://bilat.xyz/audio/in.wav" type="audio/wav">
Your browser does not support the audio element.
</audio>

High quality audio, i.e. what we want to achieve, the target
<audio controls>
  <source src="https://bilat.xyz/audio/target.wav" type="audio/wav">
Your browser does not support the audio element.
</audio>

As you can hear, the degradation in the first file is clear. It sound like it was recorded in a box, and it is actually due to the loss of high freqencies following the down-sampling of the file.





<a name="potential_improvements_"></a>

# Discussion

When creating a neural network, one important aspect it to chose the appropriate loss. There are a few types of losses that can be considered. 

First, the reconstruction loss. This one is the simple loss that we usually think of at first. It looks at how far away are we from the target data, in a "mathematical" way. We often use the mean squared error, but other types of distance exist.

The perceptual loss also looks at the distance between two signals, but at a higher level. For instance, we can look at the difference in frequencies between the two signals, and create a loss function from this. We still compute the loss between two accurate representations of the data, but we look at it from a different angle. We could have for instance used the Log-spectral distance (LSD) used in the metrics, and it might have been helpful, especially for super-resolution where it is important to recreate high frequencies.

The adverarial loss is the one that is implemented by the discriminator. The adversarial network tries to learn that the generator creates fake samples, and meanwhile the generator tries to create samples that are classified as real by the discriminator. This is a kind of minimax game, and there are also multiple possible loss. 

Finally, there is also the style loss that is often use in image processing to create images "in the style of" another image. This time, it looks at very high features of the data, and compute the loss from this. In the context of this project, style loss doesn't seem very usefull. Indeed, the style of a music file would be something like the pace of the song, the tonalities, the rythm, the instruments or maybe the compositor. However here this is basically the same for all the data, as all of it is classical music. 

On the other hand, if we were to expand this project to different types of music, it could be usefull to prevent the generated audio from sounding too different, e.g. if the input is a guitar melody, you don't want it to sound like a violin after removing the noise.

The idea behind the additional autoencoder is actually to try to automatically find those specific losses. It tries to find a low dimension representation of the data, which could be anything from just a low resolution version of the audio, or features about "sadness" or "color" or "speed" of the music. This is the goal, but since the architecture of this autoencoder is still quite simple, it is probable that is not able to learn such complicated features. 


<a name="conclusion_"></a>

# Conclusion

Audio denoising is a very complicated task, and as we have seen with the results, it is really hard to get a result that really sounds better. Unlike with images and videos, were a little artefact or defect will maybe go unseen by the untraine eye, something similar in an audio file is immediatly noticeable, especially with music file with little variation (everybody knows what a piano is supposed to sound like, so any variation will be noticable). With more general sound, like speech mixed with background music, it can maybe be less problematic as there is anyway more variation in the original audio. 

Nevertheless, this project can hopefully be of help for anybody that want to start working with audio denoising. It can serve as a starting point for the implementation of many advanced machine learning or processing methods that might be able to solve this problem. 

<a name="sources_"></a>

# Sources

**How to Develop a Conditional GAN (cGAN) From Scratch**

By Jason Brownlee, 2019, [link](https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/)

**Image-to-Image Translation with Conditional Adversarial Networks**

By Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros, 2018, [link](https://arxiv.org/pdf/1611.07004.pdf)

**Pytorch Tutorial and Documentation**

Pytorch official website, [link](https://pytorch.org/)

**PyTorch-GAN**

Collection of code samples implementing GAN in pytorch. [link](https://github.com/eriklindernoren/PyTorch-GAN)

**Speech Denoising with Deep Feature Losses**

By François G. Germain, Qifeng Chen, and Vladlen Koltun, 2018, [link](https://arxiv.org/pdf/1806.10522.pdf)

**Recurrent Neural Networks for Noise Reduction in Robust ASR**

by Andrew L. Maas1, Quoc V. Le, Tyler M. O’Neil, Oriol Vinyals, Patrick Nguyen, Andrew Y. Ng, 2012, [link](http://www1.icsi.berkeley.edu/~vinyals/Files/rnn_denoise_2012.pdf)


**Raw Waveform-based Speech Enhancement by Fully Convolutional Networks**

By Szu-Wei Fu, Yu Tsao, Xugang Lu, Hisashi Kawai, 2017, [link](https://arxiv.org/pdf/1703.02205.pdf)

**SEGAN: Speech Enhancement Generative Adversarial Network**

By Santiago Pascual, Antonio Bonafonte, Joan Serrà, 2017, [link](https://arxiv.org/pdf/1703.09452.pdf), [github](https://github.com/santi-pdp/segan). 

**Audio Super-Resolution using Neural Nets**

By Volodymyr Kuleshov, S. Zayd Enam, and Stefano Ermon, 2017, [link](https://arxiv.org/pdf/1708.00853.pdf), [website](https://kuleshov.github.io/audio-super-res/#), [github](https://github.com/kuleshov/audio-super-res)

**Adversarial Audio Super-Resolution with Unsupervised Feature Losses**

By Sung Kim, Visvesh Sathe, 2018, [link](https://openreview.net/forum?id=H1eH4n09KX)

**Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset**
 
Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck, 2019, [link](https://magenta.tensorflow.org/datasets/maestro)

A summary of every reading and additional sources can be found [here](https://bilat.xyz/vita-notes).
