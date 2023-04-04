from glob import glob
import pickle

from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from utils.pdf_loader import parsePDF

# local import
from utils.openai_client import embeddings_, OPENAI_API_KEY

data_path = 'data/*'
data_files = glob(data_path)

raw_data = []
data_source = []
for data in data_files:
    res = parsePDF(data)
    raw_data.append(res)
    data_source.append(data)


# splitting data into smaller contexts
text_splitter = CharacterTextSplitter(chunk_size=3000, separator='\n')

docs = []
metadata = []
for i, d in enumerate(raw_data):
    splits = text_splitter.split_text(d)
    docs.extend(splits)
    metadata.extend([{"source": data_source[i]}] * len(splits))


embeddings = embeddings_(openai_api_key=OPENAI_API_KEY)
# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, embeddings, metadatas=metadata)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
