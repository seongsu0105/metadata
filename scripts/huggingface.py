import argparse
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi, login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="학습 산출 폴더를 Hugging Face 모델 저장소로 업로드"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=os.getenv("HF_REPO_ID", "seongsu0105/blossom-8b-adapter"),
        help="업로드 대상 Hugging Face repo id (예: username/model-name)",
    )
    parser.add_argument(
        "--local-folder",
        type=str,
        default=os.getenv("HF_LOCAL_FOLDER", "./result/2"),
        help="업로드할 로컬 폴더 경로",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="repo가 없을 때 private 모델 저장소로 생성",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN 이 비어 있습니다. 환경 변수/Secrets를 확인하세요.")

    folder = os.path.abspath(args.local_folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"업로드 폴더를 찾을 수 없습니다: {folder}")

    login(token=token)
    api = HfApi()

    api.create_repo(
        repo_id=args.repo_id,
        token=token,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=folder,
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
    )
    print(f"업로드 완료: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
