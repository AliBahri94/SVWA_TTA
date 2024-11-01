import os
import subprocess
import time
import glob

def get_last_saved_file(main_path):
    # Assuming the saved files are in the specified directory and named as *.ply
    files = glob.glob(os.path.join(main_path, "*.ply"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime) 
    latest_number = latest_file.split("/")[-1][:-4]  
    return latest_number     
 
def run_script_with_file(main_path, dataset, filename=None):
    cmd = ["python", "./datasets/create_corrupted_dataset.py", "--main_path", main_path, "--dataset", dataset]  
    if filename:
        cmd.extend(["--continue_from", filename])  
    process = subprocess.Popen(cmd)
    return process

def main():
    main_path = "/export/livia/home/vision/Abahri/projects/MATE/MATE/data/"      
    dataset = "scanobjectnn"

    process = run_script_with_file(main_path, dataset, "1")
    while True:
        process.wait()
        if process.returncode == -11:  # -11 is the signal code for segmentation fault
            print("Segmentation fault detected.")  
            last_file = get_last_saved_file("/export/livia/home/vision/Abahri/projects/MATE/MATE/data/scanobjectnn/main_split/test_meshes/")
            if last_file:
                print(f"Restarting script with {last_file}")
                process = run_script_with_file(main_path, dataset, last_file)  
            else:
                print("No saved file found. Exiting.")
                break
        else:
            print("Script completed successfully.")  
            break

if __name__ == "__main__":
    main()