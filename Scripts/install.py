import os
import subprocess
import tempfile
import urllib.request
import venv
import zipfile

# PyTorch
pytorch_cuda_url = "https://download.pytorch.org/whl/cu129"

# Hunyuan3D-2
hunyuan_url = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2/archive/refs/heads/main.zip"
hunyuan_name = "Hunyuan3D-2-main"

# Working directories
temp_dir = tempfile.gettempdir()
work_dir = os.getcwd()
# Define the virtual environment directory
venv_dir = os.path.join(work_dir, "venv")

hunyuan_zip_path = os.path.join(temp_dir, f"{hunyuan_name}.zip")

try:
    print("Creating virtual environment...")
    venv.create(venv_dir, with_pip=True)

    # Determine pip path
    pip_path = os.path.join(venv_dir, "Scripts", "pip.exe") if os.name == "nt" else os.path.join(venv_dir, "bin", "pip")
    print(f"Using pip at {pip_path}")

    print("Installing PyTorch...")
    subprocess.check_call([pip_path, "install", "torch", "torchvision", "--index-url", pytorch_cuda_url])

    print(f"Download {hunyuan_url}...")
    urllib.request.urlretrieve(hunyuan_url, hunyuan_zip_path)

    print(f"Extracting Hunyuan3D-2...")
    with zipfile.ZipFile(hunyuan_zip_path, 'r') as zip_ref:
        zip_ref.extractall(work_dir)

    print(f"Installing Hunyuan3D-2...")
    hunyuan_path = os.path.join(work_dir, hunyuan_name)
    if os.path.exists(hunyuan_path):
        requirements_path = os.path.join(hunyuan_path, "requirements.txt")
        subprocess.check_call([pip_path, "install", "-r", requirements_path])
        subprocess.check_call([pip_path, "install", "-e", hunyuan_path])
    else:
        print(f"No ${hunyuan_path} found.")
        raise RuntimeError(f"No ${hunyuan_path} found.")

    venv_activate_path = os.path.join(venv_dir, "Scripts", "activate.bat")

    print("Creating run script...")
    with open('run.bat', 'w') as file:
        file.write(
            '@echo off\n'\
            'setlocal\n'\
            f'call {venv_activate_path}\n'\
            'powershell -Command "python ./app.py"\n'\
            'pause\n'\
            'endlocal\n'
        )

    print("Installation completed successfully.")

finally:
    # Delete the Hunyuan3D-2 zip file from the temporary directory
    if os.path.exists(hunyuan_zip_path):
        print(f"Cleaning up...")
        os.remove(hunyuan_zip_path)
