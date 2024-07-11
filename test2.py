import numpy as np

# Sample data for testing
embeddings = np.random.rand(9, 10)  # 9 nodes, 10-dimensional embeddings
nodes = np.arange(1, 10)  # Node IDs from 1 to 9

# Initialize X with indices starting from 1 instead of 0
X = dict(zip(range(embeddings.shape[0]), embeddings))
print(X)
# Create X with incremented indices
X = {str(int(nodes[key])): X[key] for key in X}

# Print X
print(X)
