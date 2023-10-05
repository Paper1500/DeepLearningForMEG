## README

This folder contains the code for generating datasets from raw data. It contains the following:

- `create_datasets.py`: The main script for generating datasets.
- `environment.lock.yml`: The specific conda environment that was used to create the dataset.
- `environment.yml`: The file that was used to create the the conda environment.
- `conf`: The directory containing the configuration files for the script.

### Overview

This script, leveraging the `hydra` configuration framework, serves to generate datasets using the `MegDataset` class. It takes in multiple configurations to determine input and output directories, the name of the dataset, its variant, the branch, and a few other parameters.

### Prerequisites

We take an optimistic view for forward compatibility and at the time of writing, the `environment.yml` creates a working environment. However, if you encounter issues running the code, compare the packages you have installed with the ones in the `environment.lock.yml` file.

1. **Conda Environment**: Before running this script, make sure to set up the conda environment using the provided `environment.yml` file in the root directory. To do this:

   ```bash
   conda env create -f environment.yml --name <ENV_NAME>
   ```

   After setting up the environment, activate it:

   ```bash
   conda activate <ENV_NAME>
   ```

   **Note**: Replace `<ENV_NAME>` with the name of the environment specified in the `environment.lock.yml` file.

### Default Configuration

The script comes with a default configuration (`config.yaml`) with the following settings:

```yaml
build:
  input_directory: data # the path to the root input directory
  output_directory: output # the path to the root output directory
  test_run: False # if True, limits the number of subjects.

data:
  dataset: camcan # either 'mous' or 'camcan'
  downsampling: [8] # the downsampling factor(s) to use.
  max_subjects: 1000 # the maximum number of subjects to include.
  random_seed: 42 # the random seed to use.
  sample_generation: numpy # the sample generation method to use.
  window_normalization: baseline # the window normalization method to use.
  window_size: 600 # the window size to use.
  include_baseline: False # if True, includes the baseline period in the window.
```

### How to Use

1. **Configuration**: Use the provided default configuration or create a custom one under the `conf` directory.

2. **Running the Script**: To run the script, use the following command:

After modifying the configuration file, run the script using the following command:

   ```bash
   python create_datasets.py
   ```

Alternatively, you can specify individual parameters using the command line. For example, to create a dataset with the first 10 subjects from the `camcan` dataset, use the following command:

   ```bash
   python create_datasets.py data.dataset=camcan data.max_subjects=10 build.input_directory=raw_data
   ```
