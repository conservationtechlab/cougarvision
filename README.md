# cougarvision
Tools to automatically analyze images and videos from telemetering field cameras and to take responsive action.


## Setting up Conda Environment

[Instructions to install conda] (https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

The file **cougarvision_env.yml** describes the python version and various dependencies with specific version numbers. To activate the enviroment

```
conda env create -f cougarvision_env.yml

conda activate cougarvision_env

conda env list

```

The first line creates the enviroment from the specifications file which only needs to be done once. 

The second line activates the environment which may need to be done each time you restart the terminal.

The third line is to test the environment installation where it will list all dependencies for that environment
