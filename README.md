# CAR_flow
## Occluded Object Recognition Codebase

<p align="center">
  <img src="https://github.com/mrernst/CAR_flow/blob/master/img/OSCAR_mnist.png" width="375">

CAR_flow stands for Convolutional Architectures with Recurrence, a tensorflow implementation. It is the codebase used for the two conference publications [1, 2]. 
If you make use of this code please cite either one:
 

[1] **Ernst, M. R., Triesch, J., & Burwick, T. (2019). Recurrent connections aid occluded object 626 recognition by discounting occluders. In I. V. Tetko, V. Kurkova, P. Karpov, & F. Theis (Eds.), Artificial neural networks and machine learning ICANN 2019:Image processing (pp. 294-305). Springer International Publishing. https://doi.org/10.1007/978-3-030-30508-6_24**

[2] **Ernst M.R., Triesch J., Burwick T. (2020). Recurrent Feedback Improves Recognition of Partially Occluded Objects. In Proceedings of the 28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 327-332**

## Getting started with the repository

* Download the (OSCAR Datasets)[https://zenodo.org/badge/DOI/10.5281/zenodo.3540900.svg]

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
│   ├── run_engine.py             		        
│   ├── engine.py
│   ├── config.py             		                   		        
│   ├── utilities             		    
│   │   ├── afterburner.py            # description
│   │   ├── distancemetric.py         # description
│   │   ├── helper.py                 # description
│   │   ├── tfevent_handler.py        # description
│   │   ├── tfrecord_handler.py       # description
│   │   ├── visualizer.py             # description
│   ├── engine.py                     # Main Program
│   ├── parameters.py                 # Experiment Parameters
│   ├── run_engine.py                 # Setup and Run Experiments
├── experiments                   
├── LICENSE                           # Apache 2.0 License
├── README.md                         # ReadMe File
└── requirements.txt                  # conda/pip requirements
```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite SORN folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.6](https://www.python.org/downloads/release/python-360/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `occludedobjectrecognition`) and install the [required packages](requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.6`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`

By installing these packages in a virtual environment, you avoid dependency clashes with other packages that may already be installed elsewhere on your computer.

## Experiments

<p align="center">
  <img src="https://github.com/mrernst/CAR_flow/blob/master/img/BK_weights.gif" width="375">

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
