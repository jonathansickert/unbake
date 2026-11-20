"""Fetch the dataset from Hugging Face into data/ (unbake/paths.py)."""
from huggingface_hub import snapshot_download

from unbake.paths import DATA

if __name__ == "__main__":
    snapshot_download(repo_id="JonathanSickert/unbake", repo_type="dataset", local_dir=DATA)
