"""Upload baked_env to the Hugging Face dataset repo. Run from the repo root after `huggingface-cli login`."""
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(repo_id='JonathanSickert/unbake', folder_path='baked_env', repo_type='dataset')
