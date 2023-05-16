# Combining deep generative and predictive models for  peptides discovery

This is the code for "Combining deep generative and predictive models for peptides discovery" paper.

# Conda Environment Setup

We use `conda` to install the dependencies for CVAE and TCPP from the provided `CVAE.yaml` and  `TCPP.yaml` file, which can give you the exact python environment we run the code for the paper:

> **NOTE**: we [also](https://github.com/mattragoza/liGAN) highly recommend using [mamba](https://mamba.readthedocs.io/en/latest/) instead of vanilla conda for managing your conda environments. Mamba is a drop-in replacement for conda that is:

- Faster at solving environments (>10x in my experience)
- Better at resolving conflicts
- More informative when something goes wrong.



conda env create -f environment.yaml 

conda env create -f TCPP.yaml 

# Dataset


The processed data for training CAVE and TCPP are provided in the corresponding folders. Note that in the TCPP project, the `train.csv`   and  `test.csv`  files in the data folder are used to train the TCPP model, and the test dataset in the test directory should use the files generated in the save directory of the CAVE project.

In addition, the original peptide library of the experimental builds is in the cluster project. You can implement the clustering for the data if required.

It is important to note that the data provided is encrypted as the experimental data is expensive and difficult to access. Please contact the authors to provide decrypted data if required.



# Quickstart

# Step 1: Generation of active peptides by CVAE 

Run the following command in the CVAE project

train

```
python train.py --prop_file = "data/L4_aas.txt" 
--save_dir="save/L4_aas" 
--num_prop="1"



```



sampling

```

python sample.py --num_iteration="100"
 --save_file ="save/model_115.ckpt-115"  --result_filename ="result"



```

# Step 2: Active peptides screening by TCPP model

Run the following Scripts in the TCPP project.

train

Model training can be started by running the  `train.py`  script.

test

Model testing can be started by running the  `test.py`  script.
