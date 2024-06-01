import os
import git
import requests
import shutil

# Configuration
local_solution_folder = r'D:\Solution'
repo_url = "https://github.com/YashDhirajOza/ALGO.git"
local_repo_folder = r'D:\Projects\ALGO'  # Changed to a directory on the D: drive
github_api_url = "https://api.github.com/repos/YashDhirajOza/ALGO/contents/"

# Function to clone the repository if it does not exist
def clone_repo_if_needed(repo_url, local_repo_folder):
    # Ensure the parent directory exists
    parent_dir = os.path.dirname(local_repo_folder)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    
    if not os.path.exists(local_repo_folder):
        print(f"Cloning repository from {repo_url} to {local_repo_folder}")
        git.Repo.clone_from(repo_url, local_repo_folder)
    else:
        print(f"Repository already exists at {local_repo_folder}")

# Function to fetch the list of files in the GitHub repository
def get_github_files(api_url):
    response = requests.get(api_url)
    if response.status_code != 200:
        raise Exception(f"Failed to get the repository contents: {response.status_code}")
    return [file['name'] for file in response.json()]

# Function to copy new files to the local repository folder
def copy_new_files_to_repo(solution_folder, repo_folder, github_files):
    new_files = []
    for file_name in os.listdir(solution_folder):
        if file_name not in github_files:
            src_path = os.path.join(solution_folder, file_name)
            dst_path = os.path.join(repo_folder, file_name)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                new_files.append(file_name)
    return new_files

# Function to commit and push new files to the repository
def commit_and_push_changes(repo, files_to_commit):
    if files_to_commit:
        repo.index.add(files_to_commit)
        repo.index.commit('Add new solutions')
        origin = repo.remotes.origin
        origin.push()
        print(f"Committed and pushed {len(files_to_commit)} files.")
    else:
        print("No new files to commit.")

# Main execution
def main():
    clone_repo_if_needed(repo_url, local_repo_folder)
    repo = git.Repo(local_repo_folder)
    github_files = get_github_files(github_api_url)
    new_files = copy_new_files_to_repo(local_solution_folder, local_repo_folder, github_files)
    commit_and_push_changes(repo, new_files)

if __name__ == "__main__":
    main()
