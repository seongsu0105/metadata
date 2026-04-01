import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# 1. 설정
load_dotenv()  # .env 파일을 읽어옵니다.
token = os.getenv("HF_TOKEN")
login(token=token)
repo_id = "seongsu0105/blossom-8b-adapter"  # 저장소 이름
local_folder_path = "./result/2"  # 파일들이 들어있는 폴더 경로

# 2. 허깅페이스 API 호출
api = HfApi()

# 3. 저장소 생성 (이미 있으면 넘어가고, 없으면 만듭니다)
api.create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)

# 4. 폴더 내 모든 파일 업로드 (safetensors, json 등 한꺼번에)
api.upload_folder(
    folder_path=local_folder_path, repo_id=repo_id, repo_type="model", token=token
)

print(f" 업로드 완료! 저장소: https://huggingface.co/{repo_id}")
