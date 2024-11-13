from huggingface_hub import HfApi, HfFolder, Repository

# Set your Hugging Face username
username = "Happyb"

# Set the local path to your trained model
model_path = "/scratch/gpfs/awettig/wura/checkpoints/Lugha-Llama"

# Set the name of your private repository
repo_name = "private-model"

# Authenticate with Hugging Face
hf_token = HfFolder.get_token()
api = HfApi()

# Create a new private repository
repo = api.create_repo(
    repo_name,
    private=True,
    repo_type="model",
    organization=username
)

# Push the model to the new repository
repo = Repository(local_dir=model_path, clone_from=f"{username}/{repo_name}")
repo.push_to_hub()

print(f"Model uploaded to private repository: {username}/{repo_name}")
