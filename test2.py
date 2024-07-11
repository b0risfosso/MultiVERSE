import numpy as np

# Sample data for testing
embeddings = np.random.rand(9, 10)  # 9 nodes, 10-dimensional embeddings
nodes = np.arange(1, 10)  # Node IDs from 1 to 9

# Ensure indices of X start from 1 instead of 0
X = dict(zip(range(1, embeddings.shape[0] + 1), embeddings))
X = {str(int(nodes[key-1])): X[key] for key in X}

# Print the resulting dictionary
print("X contents:")
for key, value in X.items():
    print(f"Key: {key}, Value: {value}")

# Verify the indices
expected_keys = [str(i) for i in range(1, 10)]
actual_keys = list(X.keys())
assert expected_keys == actual_keys, f"Expected keys {expected_keys} but got {actual_keys}"
print("Test passed: Keys are incremented correctly from 1 to 9")


print(X)