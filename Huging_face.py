huggingface-cli login

from huggingface_hub import HfApi, HfFolder

api = HfApi()
token = HfFolder.get_token()

api.create_repo(repo_id="Happyb/afroMeta-Llama-3-8B", private=True)

# Push files to your repository
api.upload_file(
    path_or_fileobj="../../awettig/scaling_wura/checkpoints/",
    path_in_repo="Meta-Llama-3-8B_wura_data-packed_bsz256_steps3000_lr6e-5_warmup0.05_afr+amh+eng+fra+hau+ibo+kin+mlg+nya+orm+por+sna+som+sot+swa+tir+xho+yor+zul",
    repo_id="Happyb/afroMeta-Llama-3-8B",
    token=token
)