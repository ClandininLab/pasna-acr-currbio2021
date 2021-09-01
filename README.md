# pasna-acr-currbio2021
Original data and analysis code for Carreira-Rosario et al. "Mechanosensory input during circuit formation shapes Drosophila motor behavior through Patterned Spontaneous Network Activity", Current Biology 2021.

# Prerequisites

The analysis code **pasna2021** only supports Python3, so in the commands below, the **pip** and **python** commands should refer to a Python3 install.  You can either install Python3 directly or through a package manager like Conda.

# Data
1. Open a terminal, and note the current directory, in which the command below will clone the code from GitHub.
2. Clone [pasna-acr-currbio2021](https://github.com/ClandininLab/pasna-acr-currbio2021):
```shell
> git clone https://github.com/ClandininLab/pasna-acr-currbio2021.git
> cd pasna-acr-currbio2021
> cd data
```
3. Inside the **data** directory, there are four folders corresponding to data from four main experiment groups. Each group except **wt** contains a control (**ctl**) folder and an experimental (**exp**) folder. Within each folder, there is a **.xlsx** file containing summary information for each embryo. The fluorescence time course of each embryo is in a separate **.csv** file.

# Analysis Code Installation

1. Go back to the top **pasna-acr-currbio2021** directory. 
2. Install **pasna2021**:
```shell
> pip install -e .
```

If you get a permissions error when running the **pip** command, you can try adding the **--user** flag.  This will cause **pip** to install packages in your user directory rather than to a system-wide location.

# Running the Example Analysis Code

1. If you do not have an installation of Jupyter, install Jupyter Lab or Notebook.
```shell
> pip install jupyterlab
```
2. In a terminal tab, navigate to the examples directory and open one of the Jupyter notebooks. The notebooks contain all the analysis performed for the paper.
```shell
> cd examples
> jupyter notebook analyze_wt.ipynb
```
