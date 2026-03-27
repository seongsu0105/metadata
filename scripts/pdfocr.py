from PyPDF2 import PdfReader
import os

base_dir = os.path.dirname(__file__)

file_path = os.path.abspath(
    os.path.join(base_dir, "..", "pdfdata", "2217418_의사국 의안과_의안원문.pdf")
)

print("파일 경로:", file_path)
print("존재 여부:", os.path.exists(file_path))

reader = PdfReader(file_path)

text = ""

for page in reader.pages:
    content = page.extract_text()

    if content:
        text += content + "\n"
    else:
        print("⚠️ 이 페이지는 텍스트 추출 안됨 (OCR 필요)")

file_name = os.path.splitext(os.path.basename(file_path))[0]
output_file_name = file_name + ".txt"

# data 폴더 없으면 생성
output_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, output_file_name)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print("저장 완료:", output_path)