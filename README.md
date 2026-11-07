# Benchmarking Generative AI Models for Deep Learning Test Input Generation
  <p align="justify">This repository contains the source code and test generation input data of the paper “Benchmarking Generative AI Models for Deep Learning Test Input Generation” accepted at ICST 2025.</p> 

## Motivation:

 <p align="justify">The objective of this research is to benchmark and combine different GenAI models with test input generators, assessing their effectiveness, efficiency, and quality of the generated test images, in terms of domain validity and label preservation</p>

## Repository Structure:
Every folder follows the same order: **dataset → model → mutation level**. Datasets are `mnist`, `svhn`, `cifar10`, `imagenet_pizza`, `imagenet_teddy` (the two ImageNet classes of the paper); models are `vae`, `gan`, `sd`; levels are `high`, `low`.

```
tig/                    unified test input generator (python -m tig), see below
tests/                  pytest suite + golden files
legacy/<dataset>/       original scripts used for the paper
    classifier/         model.py, train.py, weights/
    vae/                tig_vae.py, model.py, train.py, weights/, samples/
    gan/                tig_gan.py (ImageNet: tig_biggan.py), train_cdcgan.py, weights/, samples/
    sd/                 tig_sd.py, generate_images.py
generated_images/<dataset>/<model>_<level>/     generated test inputs (.npy) + summary sheet
evaluation_results/
    rq1/<dataset>/      <model>_rq1.py scripts and <model>/ data
    rq2_rq3/            <dataset>_<model>_<level>.csv
    human_study/        MTurk results <dataset>.csv, survey/ pages and URL generator
    mutation_extent/    csv/ ranges and scripts/ that compute them
    npy_to_png/         converters from .npy to images
documentation/          Stable Diffusion fine-tuning guide and parameters
```

## Unified CLI (`tig/`)

The `tig` package runs every generator through one genetic algorithm:

```
python -m tig --model vae --dataset mnist --mutation high --seed 0 --num-images 100
```

- `--model vae|gan|sd`, `--dataset mnist|svhn|cifar10|imagenet_pizza|imagenet_teddy`; combinations not ported yet print the legacy script to use.
- Ported: VAE and GAN on every dataset (conditional DCGAN on MNIST/SVHN/CIFAR-10, BigGAN-deep-256 on ImageNet). Stable Diffusion still runs from `legacy/<dataset>/sd/`.
- `--mutation low|high`: the initial perturbation is `0.0002` (low) or `0.002` (high) times the latent range `max - min`, the rule behind every value in the legacy scripts.
- MNIST weights are in the repo. Put the Google Drive checkpoints below under `weights/<dataset>/` (e.g. `weights/svhn/svhn_convend.pth`, `weights/svhn/SVHN_vggnet.pth`, `weights/imagenet/imagenet_vae_model_epoch_41.pth`); the exact paths are in `tig/config.py`.
- ImageNet: `pip install pytorch-pretrained-biggan scipy`. BigGAN and the VGG19-BN classifier download their weights on first use (~800 MB). The ImageNet VAE starts from real images: put them in `data/imagenet_pizza/<any_folder>/*.jpg` (or `imagenet_teddy`), e.g. the ImageNet validation images of that class.
- The same `--seed` gives the same results. Output: `results/<model>_<dataset>_<mutation>_s<seed>/summary.csv` plus one `.npy` per image.
- Tests: `python -m pytest tests` (unit tests, compatibility with the legacy networks, golden regression tests).

 ## Getting Started to RUN GenAI-Tigs:
 ### STEP-1: Configure the environment:

 ### STEP-2: Download Pretrained Classifier Checkpoints:
 To evaluate the performance of the classifier under the test generator, you can obtain the pretrained weight checkpoints from the provided link.
 
  -Mnist_classifier_ckpt: already in `legacy/mnist/classifier/weights/`
  
  -SVHN_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1vLS_9TT4ncrAfP3LVAOQzw-zdKUgoPBb/view?usp=sharing)
  
  -Cifar10_classifier_ckpt:[Download ckpt here](https://drive.google.com/file/d/1sxG5En1Vc1pEFhedebO8fRcvbb1NNE_y/view?usp=sharing)
  
  -Imagenet_classifir_ckpt:For Imagenet, we used pretrained classifier weights vgg-19-bn  directly from the PyTorch repository [see pytorch Link](https://drive.google.com/uc?export=download&id=YOUR_DIRECT_DOWNLOAD_LINK_ID)
  
  -To make changes to hyperparameters or to train the classifier from scratch, execute the following command:
  
    python3 legacy/mnist/classifier/train.py    (similar for other datasets)


### STEP-3: Execute Any One of the Three Generative AI Models Based on Test Input Generators
 
 ### 1. VAE:
 
 We have trained the Variational Autoencoder (VAE) on all four datasets: MNIST, SVHN, CIFAR-10, and ImageNet. You can download the pretrained weights for all four models from the following link.

- Mnist_vae_ckpt:already in `legacy/mnist/vae/weights/`
- SVHN_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/13D8DXRQ41pNv29jZDuWKjjUXMaXlpeG1/view?usp=sharing)
- Cifar10_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/1dLYUewBnDfOh6qsy8REWFbb57pktKg6k/view?usp=sharing)
- Imagenet_vae_ckpt:[Download ckpt here](https://drive.google.com/file/d/1iM9Sp7l7zc5o_B5ZukQ4RP8fmkScdFBw/view?usp=sharing)
  
 Run the script by using a command.
 
 To train the VAE from scratch, run the following command:

```
python legacy/train_vae.py --dataset mnist 
```

Replace mnist with svhn, cifar10, or imagenet to train on a different dataset.

After downloading the checkpoints, run the following command to collect misbehavior-inducing inputs, run the command:

```
python -m tig --model vae --dataset mnist
```
### 2. GAN:

We have trained Conditional GANs for three datasets: MNIST, SVHN, and CIFAR-10. The pretrained weights for these models are available in their respective dataset directories under Repository structure. 

-For ImageNet, we have chosen pytorch BigGAN as the Conditional GAN model and are utilizing its pretrained weights.A detail about configuration and environment settings [here](https://github.com/lukemelas/pytorch-pretrained-gans/tree/main)

 We utilize the 256x256 size Deep-BigGAN model with the specified pretrained weights by executing the following command:
 
 ```
 G = make_gan(gan_type='biggan', model_name='biggan-deep-256')
```
 We set the truncation value to 1.0 to produce images with greater variation.
 To run the tig CDCGAN for a specific dataset, use the following command:

```
python -m tig --model gan --dataset mnist --mutation low
```
Replace mnist with svhn or cifar10. For ImageNet (BigGAN) use `legacy/imagenet/gan/tig_biggan.py`.

### 3. Stable Diffusion Setup and Script Execution
#### How to Fine-tune Stable Diffusion? 
Fine-tune stable diffusion using the khoya-ss platform on four different datasets. For a detailed description, please [click here](documentation/fine_tune_stable_diffusion.md).
#### Download SD weights
Download the fine-tuned model weights from [this link](https://drive.google.com/file/d/1FauJR7XbPt_g0W4r-LPIbv7si79JHh4V/view?usp=sharing). 

#### configuration to excute Sd_tig

To run the test generator for Stable diffusion. You are required to install the following setup steps:
### Create a Virtual Environment and install packages to run the SD-based generator script

First, please make sure you have Conda installed. If not, you can download and install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

 **1. Create a Conda virtual environment**:
   Open a terminal or Anaconda Prompt and run the following command to create a new Conda virtual environment:
   
   ```
   conda create --name stable_diffusion_env python=3.10
   ```
   
 **2. Installing desired packages**
     The requirements setup is already provided; run requirement-sd.txt using pip.
   
   ```
     pip install -r requirements-sd.txt
   ```

 **3. Run SD-based Generator**
     To run the script of the SD generator, run the following command:

```bash
python3 legacy/mnist/sd/tig_sd.py
```     
     
  Similar mnist is replaced with other dataset names, svhn, cifar10, and imagenet, according to their dataset name.
     
 > Note: keep upgrading diffusers and transformers to avoid errors.


### REFERENCE:
Maryam, M., Biagiola, M., Stocco, A., & Riccio, V. (2024). Benchmarking Generative AI Models for Deep Learning Test Input Generation. In Proceedings of the 18th IEEE International Conference on Software Testing, Verification and Validation (ICST 2025)




