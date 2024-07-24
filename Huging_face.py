# huggingface-cli login

from huggingface_hub import HfApi, HfFolder
import os

api = HfApi()
token = HfFolder.get_token()

#print(f"Current model token: {token}")
# api.create_repo(repo_id="Happyb/afroMeta-Llama-3-8B", private=True)

# Push files to your repository
# api.upload_file(
#    path_or_fileobj="../../awettig/scaling_wura/checkpoints/",
#    path_in_repo="Meta-Llama-3-8B_wura_data-packed_bsz256_steps3000_lr6e-5_warmup0.05_afr+amh+eng+fra+hau+ibo+kin+mlg+nya+orm+por+sna+som+sot+swa+tir+xho+yor+zul",
#    repo_id="Happyb/afroMeta-Llama-3-8B",
#    token=token
# )
#file_path = os.path.abspath("../../awettig/scaling_wura/checkpoints/Meta-Llama-3-8B_wura_data-packed_bsz256_steps3000_lr6e-5_warmup0.05_afr+amh+eng+fra+hau+ibo+kin+mlg+nya+orm+por+sna+som+sot+swa+tir+xho+yor+zul")
# directory = "../../awettig/scaling_wura/checkpoints/"
file_path = "/../awettig/scaling_wura/checkpoints/Meta-Llama-3-8B_wura_data-packed_bsz256_steps3000_lr6e-5_warmup0.05_afr+amh+eng+fra+hau+ibo+kin+mlg+nya+orm+por+sna+som+sot+swa+tir+xho+yor+zul"

print(f"Checking file at: {file_path}")

if os.path.isfile(file_path):
    print("File Found!")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="Meta-Llama-3-8B_wura_data-packed_bsz256_steps3000_lr6e-5_warmup0.05_afr+amh+eng+fra+hau+ibo+kin+mlg+nya+orm+por+sna+som+sot+swa+tir+xho+yor+zul",
        repo_id="Happyb/afroMeta-Llama-3-8B",
        token=token
    )
else:
    print("File does not exist. Please check the path.")


#directory = os.path.abspath("../../awettig/scaling_wura/checkpoints/")
#print(f"Checking files in: {directory}")

#for filename in os.listdir(directory):
#    print(filename)

#if os.path.isfile(filepath):
#    print("File exists!")
#else:
#    print("File does not exist. Check the path.")

# for filename in os.listdir(directory):
#    filepath = os.path.join(directory, filename)
#    if os.path.isfile(filepath):

"""
api.upload_file(
    path_or_fileobj=filepath,
    path_in_repo="Meta-Llama-3-8B_wura_data-packed_bsz256_steps3000_lr6e-5_warmup0.05_afr+amh+eng+fra+hau+ibo+kin+mlg+nya+orm+por+sna+som+sot+swa+tir+xho+yor+zul",
    repo_id="Happyb/afroMeta-Llama-3-8B",
    token=token
)

"""
