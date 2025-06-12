# Genetic Programming for Learning Interpretable Traffic Signal Control Policies with Intersection Communications

![Document](https://img.shields.io/badge/docs-in_progress-violet)
![Implementation](https://img.shields.io/badge/implementation-python-blue)

[![DOI](https://img.shields.io/badge/ComGPL-UnderReview-royalblue)]()
[![Py_version](https://img.shields.io/badge/python-3.11-green)](https://www.python.org/)
![License](https://img.shields.io/badge/License-None-lightgrey)

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![CityFlow](https://img.shields.io/badge/CityFlow-05bca9?style=for-the-badge)](https://cityflow-project.github.io/)
![TSC](https://img.shields.io/badge/TSC-EA4C89?style=for-the-badge)
![Explainable](https://img.shields.io/badge/Explainable-7400b8?style=for-the-badge)
![Interpretable](https://img.shields.io/badge/Interpretable-028090?style=for-the-badge)

---
This is a example code for paper `ComGPL`.

The testing platform for the algorithm is [CityFlow](https://cityflow-project.github.io/) and
`ComGPL` is adapted as an intelligent agent in [LibSignal](https://github.com/DaRL-LibSignal/LibSignal).

## Usage
This project does not have complex third-party dependencies!
### Download
Please execute the following command to install and configure our environment.
```shell
git clone https://github.com/Rabbytr/comgpl.git
```
If you don't have Git on your computer, you can download the [zip](https://github.com/Rabbytr/comgpl/archive/refs/heads/main.zip) file directly.

### Requirements
You can use pip to directly install the following dependencies.
```shell
pip install -r requirements.txt
```
```text
deap==1.4.1
numpy==2.0.0
libsumo==1.20.0
gym==0.26.2
pyyaml==6.0.1
pathos==0.3.2
```

To install CityFlow simulator:

> To ensure that CityFlow installs successfully, please make sure that [CMake](https://cmake.org/) is installed on your computer and C++ can be compiled on your computer

```shell
git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
pip install -e .
```

> The [pybind11](https://github.com/pybind/pybind11) in source code of CityFlow is unfortunately a static version. If you encounter installation errors, you may consider upgrading the pybind11 version included in CityFlow.

### Run ComGPL

After installing the dependencies mentioned above, you can directly run `ComGPL`.
```shell
python run_comgpl.py
```

To quickly get feedback on running and keep you from getting bored :joy:, the parameters are relatively small. 
Once the code runs correctly, please set appropriate parameters or those recommended in the paper.
