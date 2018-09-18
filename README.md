# DenseSharp (WIP)
3D Deep Learning from CT Scans Predicts Tumor Invasiveness of Subcentimeter Pulmonary Adenocarcinomas

DenseSharp Networks are 3D DenseNet-based deep neural networks with multi-task learning the nodule classification labels and segmentations. 

# Code Structure
* [`mylib/`](mylib/):
    * ...
* [`explore.ipynb`](explore.ipynb)
* [`training.py`](training.py):

# Requirements
* Python 3 (Anaconda 3.6.3 specifically)
* TensorFlow==1.4.0
* Keras==2.1.5
* To plot the 3D mesh, you may also need [`plotly`](https://plot.ly/python/) installed.
Higher versions should also work (perhaps with minor modifications).


# Data samples
Unfortunately, our dataset is not available publicly considering the patients' 
privacy. However, you can still run the code using the sample dataset 
([download](...)). Please note, the data in the dataset is just for demonstration. 

Generally, the *DenseSharp* Networks are parameter-efficient multi-task learning 
networks for classification and segmentation. It's generally designed for 3D data,
with these two task labels. You can run the code on your own data if you process
your dataset following the sample data format.

# LICENSE
The code is under Apache-2.0 License.

The sample dataset is just for demonstration, neither commercial nor 
academic use is allowed.

