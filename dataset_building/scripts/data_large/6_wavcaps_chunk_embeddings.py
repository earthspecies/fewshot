import pickle
import os

# Load the embeddings dictionary from the pickle file
with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Calculate the number of embeddings per chunk
num_embeddings = len(embeddings)
chunk_size = num_embeddings // 5  # Base size of each chunk

# Split the dictionary into 5 chunks
embedding_items = list(embeddings.items())
chunks = [embedding_items[i * chunk_size: (i + 1) * chunk_size] for i in range(5)]

# Adjust the last chunk to include any remaining items (if num_embeddings % 5 != 0)
if num_embeddings % 5 != 0:
    chunks[-1].extend(embedding_items[5 * chunk_size:])

# Save each chunk to a separate pickle file
output_dir = "embedding_chunks"
os.makedirs(output_dir, exist_ok=True)

for i, chunk in enumerate(chunks):
    chunk_dict = dict(chunk)  # Convert each chunk back to a dictionary
    chunk_fp = os.path.join(output_dir, f"wavcaps_embeddings_chunk{i + 1}.pkl")
    with open(chunk_fp, 'wb') as f:
        pickle.dump(chunk_dict, f)
    print(f"Saved chunk {i + 1} with {len(chunk_dict)} embeddings to {chunk_fp}")
