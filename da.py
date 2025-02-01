import faiss
import pickle
import numpy as np

# Specify the paths to your .pkl and .faiss files
datapkl = r"C:\Users\Shriya Deshpande\Downloads\index.pkl"
datafaiss = r"C:\Users\Shriya Deshpande\Downloads\index.faiss"

# Load the pickle file
with open(datapkl, 'rb') as file:
    metadata = pickle.load(file)

# Assuming metadata is a tuple containing (vector_ids, original_data)
vector_ids, original_data = metadata

# Load the FAISS index
index = faiss.read_index(datafaiss)

#print(dir(vector_ids))
#print(vars(vector_ids))

original_data = vector_ids._dict
document_content = {k: v.page_content for k, v in original_data.items()}

#print(document_content)

with open('input2.txt', 'w', encoding='utf-8') as f:
    for key, value in document_content.items():
        f.write(f"page_content:{value}\n")