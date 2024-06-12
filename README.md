# From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers
Repo for the paper ```From Interpolation to Extrapolation: Complete Length Generalization for Arithmetic Transformers```. Attention Bias Calibration is a method of attention biasing that enables near-perfect accuracies (>99%) on sequences up to 10 times longer than the maximum training length in certain arithmetic tasks. For more information see [Arxiv link](https://arxiv.org/abs/2310.11984)


## File hierarchy
The files are partitioned into two main parts:
1. src: This contains all the source code for the project
2. exp: This contains all the experiments and plots generated

In the exp folder, there are 3 files:
config.ini: The configuration for that particular experiment
train.log: Generated upon running, this file stores the accuracy data with respect to epochs
models: This file stores all the trained epochs in the form of a .ckpt file, which could be directly loaded via PyTorch

# How to Use

## Installation
Python: 3.10.4
Library requirements: see requirements.txt

## Running the Code

### Train for Interpolation

To create and run a model, follow the steps below:
1. Create a folder containing the config.ini file
2. Specify the parameters in the config file
3. Switch to that folder: 
```bash
cd ./exp/your_folder_name
```

4. use the command below to start training:

```bash
CUDA_VISIBLE_DEVICES=(your_device) nohup torchrun --rdzv-endpoint (your_specified_port) ../../src/train.py  (config_file_path) (number_of_epochs) > {output_log.log} 2>&1 &
```

Example:

```bash
CUDA_VISIBLE_DEVICES=1 nohup torchrun --rdzv-endpoint 0.0.0.0:29544 ../../src/train.py  ./head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini 256 > train.log 2>&1 &
```

You might need to change the port number when running multiple experiments at once.

See the ```./exp``` folder for more details.

### Create Bias

To create the attention bias matrix, follow the steps below:
1. First train a model with LOAD_WINDOW set to false
2. Change the paths in the getBias.py program to the model you just trained
3. Run ```getBias.py``` with the following command: 
```bash
python ./src/calcBias.py \
    (path of config file goes here) \
    (path of model goes here) \
    (window plots save path goes here, we are using ./exp/plots/) \
    (attention bias save path goes here, we are using ./exp/windows/) \
    -g (GPU id goes here)
```
Example:

```bash
python ./src/calcBias.py \
    ./exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120/head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini \
    ./exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120/model/epoch-256-loss-0.0000000097-acc-0.0000000000.ckpt \
    ./exp/plots/ \
    ./exp/window_saves/ \
    -g 0
```

4. The bias tensors should be stored in your specified attention bias save path, and the attention plots should be saved in the plot save path.

### Retrain with bias
  
5. Create a new model using the steps mentioned previously, this time set ```LOAD_WINDOW``` to true and fill in the paths with the matrices saved in step 4
6. Train the new model, it should achieve good accuracy much faster
7. Evaluate the model by running ```eval.py``` with the following command: 
```bash
python ./src/eval.py \
    (path of config file goes here) \
    (path of model goes here) \
    -g (GPU id goes here)
```

Example:
```bash
python ./src/eval.py \
    ./exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex2/head-8-enc-3-dec-3-emb-128-batch-512-drop-0.3.ini \
    ./exp/h8enc3dec3emb128batch512drop0.3add_vanilla_len120_ex2/model/epoch-4-loss-0.0053801495-acc-0.0000000000.ckpt \
    -g 0
```

Examples are also included in the ```./exp``` folder, in the two folders with the _ex suffix

The new model should be able to extrapolate in the tasks we provided.
Note that in order for an effective bias to be created, the model must first successfully interpolate, that is, it must achieve a high enough accuracy before creating the bias matrix for it to work.

In some cases, when retraining with the bias on, it is necessary to make sure that the model does not overfit. From my experience the optimal model epoch is around 4.


Email: shaoxiongduan@gmail.com. Contact me if you have questions.
