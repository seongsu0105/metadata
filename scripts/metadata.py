import os
import re
import json
from PyPDF2 import PdfReader

# 경로 설정
base_dir = os.path.dirname(__file__)
pdf_dir = os.path.abspath(os.path.join(base_dir, "..", "pdfdata"))
data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))
meta_dir = os.path.abspath(os.path.join(base_dir, "..", "metadata"))

os.makedirs(data_dir, exist_ok=True)
os.makedirs(meta_dir, exist_ok=True)


# 전처리 함수
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"의안\s*번호.*?\d+", "", text)
    text = re.sub(r"발의연월일[:\s]*\d{4}.*?\d+", "", text)
    text = re.sub(r"법률제\s*\d+호?", "", text)

    text = re.sub(r"\(.*?\)", "", text)

    return text.strip()


# 청킹 함수
def split_document(text, file_id):
    parts = re.split(r"(제안이유|주요내용|개정내용|목적|배경|법률안)", text)

    result = []
    chunk_index = 0

    for i in range(1, len(parts), 2):
        section = parts[i]
        content = clean_text(parts[i + 1])

        sentences = re.split(r"(?<=[.!?])\s+", content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            result.append(
                {
                    "text": sentence,
                    "metadata": {
                        "file_id": file_id,
                        "section": section,
                        "chunk_index": chunk_index,
                    },
                }
            )

            chunk_index += 1

    return result


all_chunks = []

for file_name in os.listdir(pdf_dir):
    if not file_name.endswith(".pdf"):
        continue

    pdf_path = os.path.join(pdf_dir, file_name)
    file_id = os.path.splitext(file_name)[0]

    print(f"처리중: {file_name}")

    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"

    txt_path = os.path.join(data_dir, f"{file_id}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    chunks = split_document(text, file_id)
    all_chunks.extend(chunks)


output_path = os.path.join(meta_dir, "metadata.jsonl")

with open(output_path, "w", encoding="utf-8") as f:
    for item in all_chunks:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n완료! 총 청크 수: {len(all_chunks)}")
