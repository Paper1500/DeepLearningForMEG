# README: Temporally Rich Deep Learning Models for Magnetoencephalography

This repository contains the code for the paper under review at [TMLR](https://openreview.net/forum?id=zSeoG5dRHK). Please note, that this is a temporary repository and will be updated with a permanent link once the paper is accepted. You should not rely on direct links to this repository.

### Abstract

Deep learning has been used in a wide range of applications, but it has only very recently been applied to Magnetoencephalography (MEG). MEG is a neurophysiological technique used to investigate a variety of cognitive processes such as language and learning, and an emerging technology in the quest to identify neural correlates of cognitive impairments such as those occurring in dementia.
Recent work has shown that it is possible to apply deep learning to MEG to categorise induced responses to stimuli across subjects.
While novel in the application of deep learning, such work has generally used relatively simple neural network (NN) models compared to those being used in domains such as computer vision and natural language processing.
%
In these other domains, there is a long history in developing complex NN models that combine spatial and temporal information.
We propose more complex NN models that focus on modelling temporal relationships in the data, and apply them to the challenges of MEG data.
We apply these models to an extended range of MEG-based tasks, and find that they substantially outperform existing work on a range of tasks, particularly but not exclusively temporally-oriented ones. We also show that an autoencoder-based preprocessing component that focuses on the temporal aspect of the data can improve the performance of existing models.

### Code

There are four main dependencies for this project:

- `hydra`: For configuration management
- `numpy`: For numerical operations
- `torch`: PyTorch deep learning framework
- `pytorch_lightning`: PyTorch Lightning framework
- `mne`: Only required for loading and preprocessing MEG data

However, the preprocessing should be done in a separate environment for compatibility issues.
The details of settings this up and running the preprocessing script are provided in the `meg` folder.
After the processing has been completed the training and evaluation with the instructions described here.

### Environment

#### Setting up the environment

To replicate the exact environment the experiments were run in, make use of the `environment.lock.yml` file.
For a more general setup that should be suitable for most users, use the `environment.yml` file.
We take an optimistic view for forward compatibility and if you encounter issues running the code using the `environment.yml`, compare the packages you have installed with the ones in the `environment.lock.yml` file.

To create an environment using conda:

   ```bash
   conda env create -f environment.yml --name <ENV_NAME>
   ```

   After setting up the environment, activate it:

   ```bash
   conda activate <ENV_NAME>
   ```

**Note**: Replace `<ENV_NAME>` with the name of the environment specified in the `environment.lock.yml` file.

#### Running the code

There are two main entry points for the code: `main.py` and `train_encoder.py`.
The former is used for training and evaluating the models, while the latter is used for pretraining the autoencoder.

### Default Configuration

The script comes with a default configuration (`conf/config.yaml`) with the following settings here are some of the settings that you might want to change.

```yaml
defaults:
  - model: timeres # the model to use. one of: timeres | timeconv | transformer | varcnn | lfcnn
  - datamodule: camcan # the dataset to use. either camcan or mous

downsampler: # the autoencoder that is used as downsampler. 
  enabled: False 
  frozen: True # if false, will fine-tune the autoencoder.
  path: # the path to the autoencoder checkpoint. Can be left empty if not enabled.

trainer:
  n_models: 3 # the number of models to train.
  use_gpu: True # if True, will use the GPU.
  fast_dev_run: False # if True, will run a single batch of training and testing.
  min_epochs: 3 # the minimum number of epochs to train for.
  max_epochs: 40 # the maximum number of epochs to train for.
  
  logger:
    _target_: pytorch_lightning.loggers.wandb.WandbLogger
    entity: <YOUR_ENTITY> # your wandb entity
    project: <YOUR_PROJECT> # your wandb project
    
system:
  data_dir: data # the path to the root data directory.
```

### How to Use

**Training a model**: To train a model, use the following command:

After modifying the configuration file, run the script using the following command:

   ```bash
   python main.py
   ```

Alternatively, you can specify individual parameters using the command line. For example, to train a model using the `mous` dataset and the `lfcnn` model, use the following command:

   ```bash
   python main.py datamodule=mous model=lfcnn
   ```

**Training an autoencoder**: Training the autoencoder is similar to training a model, but does not take the model parameter. To train the autoencoder on the CamCAN dataset, use the following command:

   ```bash
   python train_encoder.py datamodule=camcan
   ```
