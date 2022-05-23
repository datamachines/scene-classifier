# Scene Classifier

To accomplish the task of identifying surrounding environments we trained a scene classifier using a neural network. 

PyTorch and Jupyter were leveraged via our `jupyter_to` docker container to develop, train and test our model.  
DockerHub registry: https://hub.docker.com/r/datamachines/jupyter_to 

We used MIT CSAIL's Places365 library of data and model architectures to train our scene classifier neural network (https://github.com/CSAILVision/places365). 

The Places365 dataset (http://places2.csail.mit.edu/) is an image dataset of 365 different scenes. We took a subset of the data to included only 49  outdoor scenes. The class names are listed in `subset_data_classes.txt`

## Running the model

To run the trained model on a single image:

```
python run.py <test image> --num_classes 49
```

The run script calls the list of class names, so you must include the number of classes as an argument.


## Training (requires GPU)

### Downloading the data
Our subsetted dataset (places_data.zip 3.63 GB) contains 274948 training images, and 5500 validation images.

To download the data run `make get_data` in your local directory (outside of the container).


### Start up the GPU container

To build the GPU version of the container: https://hub.docker.com/r/datamachines/jupyter_cto

Run the container using a docker run command, i.e.:

```
docker run -v `pwd`:/dmc --rm --gpus device=all -p 8888:8888 datamachines/jupyter_cto:11.3.1_2.8.0_4.5.5-20220318
```

When this command is run, it will create a link for you to access Jupyter in a browser on your local host. 

### Train from scratch

The default base model is set to resnet18.

To retrain from scratch use `train_placesCNN.py` from https://github.com/CSAILVision/places365 (this file uses CUDA)

To use GPU compiled TensorFlow and Opencv, Keras, NumPy, pandas, and PyTorch, all from within the Jupyter Notebook, use the CUDA version of our `jupyter_to` container. https://hub.docker.com/r/datamachines/jupyter_cto

```
python train_placesCNN.py places_data --num_classes 49
```

It will save the best model as `resnet18_best.pth.tar` and the latest model as `resnet18_latest.pth.tar`


### Train as a feature extractor

To retrain the network as a feature extractor (a transfer learning method that freezes all the layers except the last layer of the model) use `retrain.py`. This offers a faster training solution, but generally lower accuracy performance. 

```
python retrain.py places_data --num_classes 49
```

`retrain.py` will save a model as `resnet18_model_weights.pth.tar`


