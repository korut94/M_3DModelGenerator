import os
import shutil

# List of items to delete
items_to_delete = ['Hunyuan3D-2-main', 'venv', 'run_app.bat']

for item in items_to_delete:
    if os.path.exists(item):
        if os.path.isdir(item):
            shutil.rmtree(item)
            print(f"Directory '{item}' has been deleted.")
        else:
            os.remove(item)
            print(f"File '{item}' has been deleted.")
    else:
        print(f"'{item}' does not exist.")
