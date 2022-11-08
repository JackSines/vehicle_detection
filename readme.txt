This folder contains all of the files necessary to set up and test the software


INSTRUCTIONS:

install miniconda using one of the installation files included (32 and 64 bit depending on your system)
it can also be downloaded here: https://docs.conda.io/en/latest/miniconda.html (the python 3.8 32 or 64 bit version)

UNTICK all options during this installation

open the anaconda prompt (search for 'anaconda prompt (miniconda3)' in windows)

create the environment: type 'conda env create -f PATH\env.yaml' for example, 'conda env create -f downloads\901354\env.yaml'

once created, copy the two folders (application and yolo) into the new environment located at C:\users\USER\miniconda3\envs\obj_det\

activate the environment in the anaconda prompt: type 'activate obj_det'

navigate to location: type 'cd miniconda3\envs\obj_det\Application'

run main: type 'python main.py'
