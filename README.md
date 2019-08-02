# Project Saturn

```python
#                                           _.oo.
#                  _.u[[/;:,.         .odMMMMMM'
#               .o888UU[[[/;:-.  .o@P^    MMM^
#              oN88888UU[[[/;::-.        dP^
#             dNMMNN888UU[[[/;:--.   .o@P^
#            ,MMMMMMN888UU[[/;::-. o@^
#            NNMMMNN888UU[[[/~.o@P^
#            888888888UU[[[/o@^-..
#           oI8888UU[[[/o@P^:--..
#        .@^  YUU[[[/o@^;::---..
#      oMP     ^/o@P^;:::---..
#   .dMMM    .o@^ ^;::---...
#  dMMMMMMM@^`       `^^^^
# YMMMUP^
#  ^^
```


Project Saturn aims to be a clean rewrite of a toolbox for occluded object recognition with recurrent convolutional neural networks (ReCoNNet)

## Getting started with the repository


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
├── datasets                       # description
│   ├── run_single.py              # description
│   ├── sorn.py                    # description
│   ├── stats.py                   # description
│   ├── synapses.py                # description
├── network engine                 # description
│   ├── utilities             		  # description
│   │   ├── run_single.py          # description
│   │   ├── run_single.py          # description
│   │   ├── run_single.py          # description
│   │   ├── run_single.py          # description
│   ├── sorn.py                    # description
│   ├── stats.py                   # description
├── README.md
├── LICENSE
├── requirements.txt               # description
└── utils                          # description
    ├── backup.py
    └── bunch
```

### Installation guide

#### Forking the repository

Fork a copy of this repository onto your own GitHub account and `clone` your fork of the repository into your computer, inside your favorite SORN folder, using:

`git clone "PATH_TO_FORKED_REPOSITORY"`

#### Setting up the environment

Install [Python 3.6](https://www.python.org/downloads/release/python-360/) and the [conda package manager](https://conda.io/miniconda.html) (use miniconda). Navigate to the project directory inside a terminal and create a virtual environment (replace <ENVIRONMENT_NAME>, for example, with `sorn_env`) and install the [required packages](https://github.com/delpapa/SORN_V2/blob/master/requirements.txt):

`conda create -n <ENVIRONMENT_NAME> --file requirements.txt python=3.6`

Activate the virtual environment:

`source activate <ENVIRONMENT_NAME>`

By installing these packages in a virtual environment, you avoid dependency clashes with other packages that may already be installed elsewhere on your computer.

## Experiments


## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details
