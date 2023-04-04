from glob import glob
from PyPDF2 import PdfReader

data_path = 'data/*'
data_files = glob(data_path)


def parsePDF(file):
    raw_text = ''
    reader = PdfReader(file)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text
