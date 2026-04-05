"""
배치 메타데이터 생성 CLI.

실행 (scripts 디렉터리에서):
  python main.py

구성 모듈: batch_config, batch_pdf, batch_llm, batch_postprocess, batch_record, batch_pipeline
"""

from batch_pipeline import process_all

if __name__ == "__main__":
    process_all()
