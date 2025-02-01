import pickle
import faiss
import numpy as np

# 1. Load the FAISS index from the .faiss file
index = faiss.read_index(r'C:\Users\Shriya Deshpande\Downloads\index.faiss')

# 2. Load the mapping of indices to original data from the .pkl file
with open(r'C:\Users\Shriya Deshpande\Downloads\index.pkl', 'rb') as file:
    index_to_data = pickle.load(file)

# 3. Get the dimensionality of the vectors in the FAISS index
index_dimensionality = index.d
print(f"Index dimensionality: {index_dimensionality}")

# 4. Define a query vector with the correct dimensionality
query_vector = np.array([[1.0] * index_dimensionality], dtype=np.float32)  # Adjust to match the dimensionality

# 5. Perform the similarity search on the FAISS index
k = 5  # Number of nearest neighbors to retrieve
D, I = index.search(query_vector, k)

# 6. Debugging: Print the results of the search
print("Distances:", D)
print("Indices:", I)

# 7. Retrieve the original data corresponding to the indices with error handling
original_data = []
for i in I[0]:
    if i in index_to_data:
        original_data.append(index_to_data[i])
    else:
        print(f"Warning: Index {i} not found in index_to_data")

print("Original Data:", original_data)