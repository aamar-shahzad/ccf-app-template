import os

# Get the current directory
current_directory = os.getcwd()
print("Current Directory:", current_directory)
# slect workspace folder
workspace_folder = os.path.join(current_directory, "workspace")
print("Workspace Folder:", workspace_folder)
# read a file from the workspace of sandbox_0 folder
file_path = os.path.join(workspace_folder, "sandbox_0", "0.pem")
# print the file path
print("File Path:", file_path)
# print content of the file
with open(file_path, "r") as file:
    print(file.read())
