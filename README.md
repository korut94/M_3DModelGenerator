# M_3DModelGenerator
Simple Gradio App, based on Hunyuan3D-2 model, to generate 3D models from images.

# Getting Started
## Requirements
Before install the app, be sure to have `Python 3` and to install [CUDA Toolkit v12.9](https://developer.nvidia.com/cuda-12-9-0-download-archive) with the environment variable `CUDA_HOME` properly pointing at the installation path (required by PyTorch).
## Install
### Windows
Double-click on the batch file:
```
install.bat
```
### Linux
In the command line, run the python script:
```
python ./Scripts/install.py
```
## Run the app
Once the installation process is terminated succesfully, the working directory will containt: `Hunyuan3D-main` directoy, python virtual environment directory `vevn` and the script `run_app.bat`.
To run the application:
### Windows
Double-click on the batch file:
```
run_app.bat
```
### Linux
In the command line, activate the virtual environment in `venv` directory: 
```
./venv/Scripts/activate
```
and run the python script
```
python ./app.py
```
The app's web-page will be opened on your default browser.
