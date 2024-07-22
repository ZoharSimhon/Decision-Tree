import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from functools import lru_cache
import time  # Import the time module for performance measurement

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

# Brute Force Decision Tree
def brute_force_tree(X, y, k):
    n_samples, n_features = X.shape
    
    # Helper function to calculate error for a given tree
    def calculate_tree_error(tree, X, y):
        predictions = np.array([predict(tree, sample) for sample in X])
        return np.mean(predictions != y)
    
    # Prediction function for a given tree
    def predict(tree, sample):
        if isinstance(tree, (int, float)):
            return tree
        feature, left, right = tree
        if sample[feature] == 0:
            return predict(left, sample)
        else:
            return predict(right, sample)
    
    # Define a recursive function with caching to build the decision tree
    @lru_cache(maxsize=None)
    def build_tree(depth, sample_indices):
        sample_indices = frozenset(sample_indices)
        data = X[list(sample_indices)]
        labels = y[list(sample_indices)]
        
        # Base case: if maximum depth is reached or all labels are the same, return the majority class
        if depth == k or len(set(labels)) == 1:
            return int(np.mean(labels) >= 0.5)
        
        best_tree = None
        min_error = float('inf')
        
        # Iterate over all features to find the best split
        for feature in range(n_features):
            left_indices = frozenset(i for i in sample_indices if X[i, feature] == 0)
            right_indices = sample_indices - left_indices
            
            # Skip invalid splits
            if not left_indices or not right_indices:
                continue
            
            left_subtree = build_tree(depth + 1, left_indices)
            right_subtree = build_tree(depth + 1, right_indices)
            
            tree = [feature, left_subtree, right_subtree]
            error = calculate_tree_error(tree, data, labels)
            
            # Update the best split if the current one is better
            if error < min_error:
                min_error = error
                best_tree = tree
            
            # Early stopping if a perfect split is found
            if min_error == 0:
                break
        
        return best_tree if best_tree is not None else int(np.mean(labels) >= 0.5)

    initial_indices = frozenset(range(n_samples))
    best_tree = build_tree(0, initial_indices)
    
    # Calculate the overall error of the tree
    best_error = calculate_tree_error(best_tree, X, y)
    
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

def run_algo(X, y, k, decision_algorithm, algo_name):
    # Start the timer
    start_time = time.time()
    
    # Run the decision method
    error, tree = decision_algorithm(X, y, k-1)
    
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    
    
    success_rate = (1 - error) * 100
    print(f"{algo_name} success rate: {success_rate:.2f}%")
    print(f"Execution time: {elapsed_time:.4f} seconds\n")
    
    # Plot the decision trees
    plot_tree(tree, f"{algo_name} Decision Tree (k={k})\n\n")
    
    return success_rate, elapsed_time

def run_decision_tree_comparisons(X, y, ks):
    success_rates_brute = []
    success_rates_entropy = []
    
    elapsed_times_brute = []
    elapsed_times_entropy = []
    
    for k in ks:
        print(f"\nRunning comparisons for k={k}...\n")
        brute_force_success, brute_force_time = run_algo(X, y, k, brute_force_tree, "Brute Force")
        print(brute_force_success, brute_force_time)
        entropy_success, entropy_time = run_algo(X, y, k, binary_entropy_tree, "Binary Entropy")
        
        success_rates_brute.append(brute_force_success)
        success_rates_entropy.append(entropy_success)
        
        elapsed_times_brute.append(brute_force_time)
        elapsed_times_entropy.append(entropy_time)
    
    def plot_comparisons(brute_force_values, entropy_values, ylabel):
        plt.figure(figsize=(10, 6))
        plt.plot(ks, brute_force_values, label='Brute Force')
        plt.plot(ks, entropy_values, label='Binary Entropy')
        plt.xlabel('Depth (k)')
        plt.ylabel(ylabel)
        plt.title('Decision Tree Success Rates by Depth')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # plot success rates
    plot_comparisons(success_rates_brute, success_rates_entropy, 'Success Rate (%)') 
    
    # plot elapsed times
    plot_comparisons(elapsed_times_brute, elapsed_times_entropy, 'Elapsed Time (s)') 
    
# Main function to run the algorithms and visualize the results
if __name__ == "__main__":
    # Load data from file
    X, y = load_data('vectors.txt')

    # Set the maximum depth of the tree
    ks = [3] 
    # ks = [2,3,4,5,6,7] 

    # run the algorithms
    run_decision_tree_comparisons(X, y, ks)