# DenseSharp (WIP)
3D Deep Learning from CT Scans Predicts Tumor Invasiveness of Subcentimeter Pulmonary Adenocarcinomas

DenseSharp Networks are 3D DenseNet-based deep neural networks with multi-task learning the nodule classification labels and segmentations. 

# Code Structure
* [`mylib/`](mylib/):
    * [`dataloader/`](mylib/dataloader): PyTorch-like datasets and dataloaders for Keras.
    * [`models/`](mylib/models): 3D *DenseSharp* and *DenseNet* models togethor with the losses and metrics.
    * [`utils/`](mylib/utils): plot and multi-processing utils.
* [`explore.ipynb`](explore.ipynb): plots and basic views of networks.
* [`training.py`](training.py): #TODO

# 3D Nodule Mesh Plots
The 3D mesh plots are used for illustration interactively.
![3d nodule mesh pltos](3dmesh.gif)

The helper functions are provided in [`mylib/utils/plot3d.py`](mylib/utils/plot3d.py).
See [`explore.ipynb`](explore.ipynb) for the demo code. 
Control the mesh step by setting `step_size`.
 

# Requirements
* Python 3 (Anaconda 3.6.3 specifically)
* TensorFlow==1.4.0
* Keras==2.1.5
* To plot the 3D mesh, you may also need [`plotly`](https://plot.ly/python/) installed.
Higher versions should also work (perhaps with minor modifications).


# Data samples
Unfortunately, our dataset is not available publicly considering the patients' 
privacy, and restrictions apply to the use. 
However, you can still run the code using the sample dataset 
([download](https://drive.google.com/open?id=1c-suZobPIH-DSE99zspPb098jEiDqRGa)).
Please note, the data in the dataset is just for demonstration.

Unzip the sample dataset, then modify the `"DATASET"` field in `mylib/dataloader/ENVIRON`.

The *DenseSharp* Networks are parameter-efficient multi-task learning 
networks for classification and segmentation. It's generally designed for 3D data,
with these two task labels. You can run the code on your own data if you process
your dataset following the sample data format.

# LICENSE
The code is under Apache-2.0 License.

The sample dataset is just for demonstration, neither commercial nor 
academic use is allowed.

