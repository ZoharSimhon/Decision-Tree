import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from functools import lru_cache

# Function to load data from file
def load_data(filename):
    # Load data from a text file, assuming each row is a data point and the last column is the label
    data = np.loadtxt(filename, dtype=int)
    X = data[:, :-1]  # Features (all columns except the last)
    y = data[:, -1]   # Labels (last column)
    return X, y

# Helper function to calculate binary entropy
def binary_entropy(p):
    # Handle edge cases where p is 0 or 1 (no entropy)
    if p == 0 or p == 1:
        return 0
    # Calculate binary entropy
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

# Helper function to calculate error
def calculate_error(y_true, y_pred):
    # Calculate the proportion of incorrect predictions
    return np.mean(y_true != y_pred)

# Optimized Brute Force Decision Tree
def optimized_brute_force_tree(X, y, k):
    n_samples, n_features = X.shape
    
    # Define a recursive function with caching to build the decision tree
    @lru_cache(maxsize=None)
    def build_tree(depth, sample_indices):
        # Convert sample_indices to a frozenset for caching
        sample_indices = frozenset(sample_indices)
        
        # Base case: if maximum depth is reached or all labels are the same, return the majority class
        if depth == k or len(set(y[list(sample_indices)])) == 1:
            majority_class = Counter(y[list(sample_indices)]).most_common(1)[0][0]
            return majority_class, majority_class
        
        best_error = float('inf')
        best_tree = None
        
        # Iterate over all features to find the best split
        for feature in range(n_features):
            left_indices = frozenset(i for i in sample_indices if X[i, feature] == 0)
            right_indices = sample_indices - left_indices
            
            # Skip invalid splits
            if not left_indices or not right_indices:
                continue
            
            left_subtree, left_error = build_tree(depth + 1, left_indices)
            right_subtree, right_error = build_tree(depth + 1, right_indices)
            
            # Calculate the weighted error of the split
            total_error = left_error * len(left_indices) / len(sample_indices) + \
                          right_error * len(right_indices) / len(sample_indices)
            
            # Update the best split if the current one is better
            if total_error < best_error:
                best_error = total_error
                best_tree = [feature, left_subtree, right_subtree]
            
            # Early stopping if a perfect split is found
            if best_error == 0:
                break
        
        # Combine subtrees if they are the same class
        if isinstance(best_tree, list) and isinstance(best_tree[1], (int, float)) and isinstance(best_tree[2], (int, float)) and best_tree[1] == best_tree[2]:
            return best_tree[1], best_error
        
        return best_tree, best_error

    initial_indices = frozenset(range(n_samples))
    best_tree, best_error = build_tree(0, initial_indices)
    
    # Combine subtrees to simplify the tree
    combine_subtrees(best_tree)
    
    return best_error, best_tree

# Binary Entropy Decision Tree
def binary_entropy_tree(X, y, k):
    # Recursive function to split the node
    def split_node(X_node, y_node, depth):
        # Base case: if maximum depth is reached or all labels are the same, return the majority class
        if depth == k or len(set(y_node)) == 1:
            return Counter(y_node).most_common(1)[0][0]
        
        best_feature = None
        best_entropy = float('inf')
        
        # Iterate over all features to find the best split
        for feature in range(X_node.shape[1]):
            left_mask = X_node[:, feature] == 0
            right_mask = ~left_mask
            
            # Skip invalid splits
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            left_prop = np.mean(y_node[left_mask])
            right_prop = np.mean(y_node[right_mask])
            
            left_entropy = binary_entropy(left_prop)
            right_entropy = binary_entropy(right_prop)
            
            # Calculate the total entropy of the split
            total_entropy = (np.sum(left_mask) * left_entropy + np.sum(right_mask) * right_entropy) / len(y_node)
            
            # Update the best split if the current one is better
            if total_entropy < best_entropy:
                best_entropy = total_entropy
                best_feature = feature
        
        # If no valid feature was found, return the majority class
        if best_feature is None:
            return Counter(y_node).most_common(1)[0][0]
        
        left_mask = X_node[:, best_feature] == 0
        right_mask = ~left_mask
        
        # Recursively split the left and right child nodes
        return [best_feature,
                split_node(X_node[left_mask], y_node[left_mask], depth + 1),
                split_node(X_node[right_mask], y_node[right_mask], depth + 1)]
    
    # Build the tree starting from the root
    tree = split_node(X, y, 0)
    
    # Predict function to traverse the tree and make predictions
    def predict(tree, sample):
        if not isinstance(tree, list):
            return tree
        if sample[tree[0]] == 0:
            return predict(tree[1], sample)
        else:
            return predict(tree[2], sample)
    
    # Make predictions for all samples
    predictions = np.array([predict(tree, sample) for sample in X])
    error = calculate_error(y, predictions)
    
    # Combine subtrees to simplify the tree
    combine_subtrees(tree)
    
    return error, tree

# Improved plotting function for the decision tree
def plot_tree(tree, title):
    def plot_node(ax, node, x, y, dx, dy):
        if isinstance(node, (int, float)):  # Leaf node
            color = 'lightcoral' if node == 0 else 'lightblue'
            ax.text(x, y, f'Leaf: {node}', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, ec='black'))
            
        elif isinstance(node, list) and len(node) == 3:  # Internal node
            ax.text(x, y, f'Feature {node[0]}', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgray', ec='black'))
            
            # Left child
            ax.annotate('0', xy=(x, y), xytext=(x-dx, y-dy),
                        arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
            plot_node(ax, node[1], x-dx, y-dy, dx/2, dy)
            
            # Right child
            ax.annotate('1', xy=(x, y), xytext=(x+dx, y-dy),
                        arrowprops=dict(arrowstyle='<-'), ha='center', va='center')
            plot_node(ax, node[2], x+dx, y-dy, dx/2, dy)
            
        else:
            color = 'lightcoral' if node == 0 else 'lightblue'
            ax.text(x, y, f'Class: {node}', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', fc=color, ec='black'))

    # Create a plot
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_node(ax, tree, 0.5, 1, 0.25, 0.2)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Function to combine subtrees if they are the same class
def combine_subtrees(tree):
    if isinstance(tree, list) and len(tree) == 3:
        left_value, left_remove = combine_subtrees(tree[1])
        right_value, right_remove = combine_subtrees(tree[2])
        
        # Update tree if left or right subtree should be combined
        if left_remove:
            tree[1] = left_value            
        if right_remove:
            tree[2] = right_value            
        
        # Combine subtrees if both sides are the same class
        if left_value != -1 and left_value == right_value:
            return (left_value, True)
        return (-1, False)   
    else:
        return (tree, False)

# Main execution
if __name__ == "__main__":
    # Load data from file
    X, y = load_data('vectors.txt')

    # Set the maximum depth of the tree
    k = 2  

    # Run optimized brute-force method
    bf_error, bf_tree = optimized_brute_force_tree(X, y, k)
    bf_success_rate = (1 - bf_error) * 100
    print(f"Brute-force success rate: {bf_success_rate:.2f}%")

    # Run binary entropy method
    be_error, be_tree = binary_entropy_tree(X, y, k)
    be_success_rate = (1 - be_error) * 100
    print(f"Binary entropy success rate: {be_success_rate:.2f}%")

    # Plot the decision trees
    plot_tree(bf_tree, f"Brute-force Decision Tree (k={k+1})\n\n")
    plot_tree(be_tree, f"Binary Entropy Decision Tree (k={k+1})\n\n")
   
