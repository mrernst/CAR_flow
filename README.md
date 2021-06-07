# CAR_flow
## Occluded Object Recognition Codebase

<p align="center">
  <img src="https://github.com/mrernst/CAR_flow/blob/master/img/OSCAR_mnist.png" width="375">

CAR_flow stands for Convolutional Architectures with Recurrence, tensorflow implementation. It is the codebase used for the two conference publications [1, 2]. 
If you make use of this code please cite either one or both:
 

[1] **Ernst, M. R., Triesch, J., & Burwick, T. (2019). Recurrent connections aid occluded object 626 recognition by discounting occluders. In I. V. Tetko, V. Kurkova, P. Karpov, & F. Theis (Eds.), Artificial neural networks and machine learning ICANN 2019:Image processing (pp. 294-305). Springer International Publishing. https://doi.org/10.1007/978-3-030-30508-6_24**

[2] **Ernst M.R., Triesch J., Burwick T. (2020). Recurrent Feedback Improves Recognition of Partially Occluded Objects. In Proceedings of the 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 327-332**

## Getting started with the repository

* Download the OSCAR Datasets from Zenodo and put the in their respective folders in /datasets [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3540900.svg)](https://doi.org/10.5281/zenodo.3540900)
* Convert data to tfrecord format using the provided scripts
* Configure the config.py file
* Start an experiment on a slurm cluster using run_engine.py 

### Prerequisites

* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [scikitlearn](http://scikit-learn.org/)
* [matplotlib](https://matplotlib.org/)
* [tensorflow](https://www.tensorflow.com)
* [tensorflow-plot](https://github.com/wookayin/tensorflow-plot)


### Directory structure

```bash
.
├── datasets                          
│   ├── cifar10                      # CIFAR10
│   ├── cifar100                     # CIFAR100
│   ├── imagenet                     # ImageNet
│   ├── mnist                        # MNIST
│   ├── fashionmnist                 # fashion-MNIST
│   ├── osmnist                      # OS-MNIST
│   ├── osycb                        # OS-YCB
├── network_engine                    
│   ├── run_engine.py                # Builds experiment substructure, launches on slurm cluster        
│   ├── utilities             		    
│   │   ├── afterburner.py            # Combines experiment files post-hoc
│   │   ├── distancemetric.py         
│   │   ├── helper.py                 # Helper functions
│   │   ├── tfevent_handler.py        
│   │   ├── tfrecord_handler.py       
│   │   ├── visualizer.py             # Visualization functions
│   │   ├── networks
│   │   │   ├── advancedrcnn.py       # Dynamic network builder
│   │   │   ├── buildingblocks.py     # Predefined modules to stack networks
│   │   │   ├── predictivecodingnet.py      # predictive coding network
│   │   │   ├── preprocessor.py       # Data augmentation framework
│   │   │   ├── simplemrcnn.py        # Networks akin Spoerer et. al. 2017
│   │   │   ├── simplercnn.py         # Networks with multiplicative conn.

│   ├── engine.py                     # Main Program
│   ├── config.py             		  # Experiment Parameters 
│   ├── run_engine.py                 # Setup and Run Experiments
├── experiments                   
├── LICENSE                           # MIT License
├── README.md                         # ReadMe File
└── requirements.txt                  # conda/pip requirements
```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite SORN folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.7](https://www.python.org/downloads/release/python-371/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `recurrentnetworks`) and install the [required packages](requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.7`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`


### Animation of changing weights during training

<p align="center">
  <img src="https://github.com/mrernst/CAR_flow/blob/master/img/BK_weights.gif" width="375">

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
