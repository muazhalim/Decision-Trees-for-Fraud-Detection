# COMP2611-Artificial Intelligence-Coursework#2 - Descision Trees

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.tree import export_text
import warnings
import os

# STUDENT NAME: MUHAMMAD MUAZ BIN HALIM
# STUDENT EMAIL:  sc22mmbh@leeds.ac.uk
    
def print_tree_structure(model, header_list):
    tree_rules = export_text(model, feature_names=header_list[:-1])
    print(tree_rules)
    
# Task 1 [10 marks]: Load the data from the CSV file and give back the number of rows
def load_data(file_path, delimiter=','):
    num_rows, data, header_list=None, None, None
    if not os.path.isfile(file_path):
        warnings.warn(f"Task 1: Warning - CSV file '{file_path}' does not exist.")
        return None, None, None
    # Insert your code here for task 1
    data_frame = pd.read_csv(file_path, delimiter=delimiter)
    data = data_frame.to_numpy()
    header_list = data_frame.columns.tolist()
    num_rows = data_frame.shape[0]

    return num_rows, data, header_list

# Task 2[10 marks]: Give back the data by removing the rows with -99 values 
def filter_data(data):
    filtered_data=[None]*1
    # Insert your code here for task 2
    # Convert data to a numpy array if it's not already one
    data = np.array(data)
    not_negative_99 = (data != -99)
    valid_rows = np.all(not_negative_99, axis=1)
    filtered_data = data[valid_rows]

    return filtered_data

# Task 3 [10 marks]: Data statistics, return the coefficient of variation for each feature, make sure to remove the rows with nan before doing this. 
def statistics_data(data):
    coefficient_of_variation=None
    data=filter_data(data)
    # Calculate the mean and standard deviation for each feature
    # Insert your code here for task 3
    means = np.mean(data, axis=0)
    std_deviation = np.std(data, axis=0)
    coefficient_of_variation = np.divide(std_deviation, means)

    return coefficient_of_variation

# Task 4 [10 marks]: Split the dataset into training (70%) and testing sets (30%), 
# use train_test_split of scikit-learn to make sure that the sampling is stratified, 
# meaning that the ratio between 0 and 1 in the lable column stays the same in train and test groups.
# Also when using train_test_split function from scikit-learn make sure to use "random_state=1" as an argument. 
def split_data(data, test_size=0.3, random_state=1):
    x_train, x_test, y_train, y_test=None, None, None, None
    np.random.seed(1)
    # Insert your code here for task 4
    labels = data[:, -1]
    features = data[:, :-1]

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels)

    return x_train, x_test, y_train, y_test

# Task 5 [10 marks]: Train a decision tree model with cost complexity parameter of 0
def train_decision_tree(x_train, y_train,ccp_alpha=0):
    model=None
    # Insert your code here for task 5
    model = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
    model.fit(x_train, y_train)
    return model

# Task 6 [10 marks]: Make predictions on the testing set 
def make_predictions(model, X_test):
    y_test_predicted=None
    # Insert your code here for task 6
    y_test_predicted = model.predict(X_test)
    return y_test_predicted

# Task 7 [10 marks]: Evaluate the model performance by taking test dataset and giving back the accuracy and recall 
def evaluate_model(model, x, y):
    accuracy, recall=None,None
    # Insert your code here for task 7
    y_predicted = model.predict(x)
    accuracy = accuracy_score(y, y_predicted)
    recall = recall_score(y, y_predicted)
    return accuracy, recall

# Task 8 [10 marks]: Write a function that gives the optimal value for cost complexity parameter
# which leads to simpler model but almost same test accuracy as the unpruned model (+-1% of the unpruned accuracy)
def optimal_ccp_alpha(x_train, y_train, x_test, y_test):
    optimal_ccp_alpha=None

    # Insert your code here for task 8
    unpruned_model = train_decision_tree(x_train, y_train, ccp_alpha=0)
    unpruned_accuracy, _ = evaluate_model(unpruned_model, x_test, y_test)

    ccp_alpha = 0
    step = 0.0001
    optimal_accuracy = unpruned_accuracy
    optimal_ccp_alpha = 0

    while True:
        model = train_decision_tree(x_train, y_train, ccp_alpha =ccp_alpha)
        accuracy, _ = evaluate_model(model, x_test, y_test)

        if accuracy >= unpruned_accuracy - 0.01:
            optimal_ccp_alpha = ccp_alpha
        else:
            break

        ccp_alpha = ccp_alpha + step
    return optimal_ccp_alpha

# Task 9 [10 marks]: Write a function that gives the depth of a decision tree that it takes as input.
def tree_depths(model):
    depth=None
    # Get the depth of the unpruned tree
    # Insert your code here for task 9
    depth = model.get_depth()
    return depth

 # Task 10 [10 marks]: Feature importance 
def important_feature(x_train, y_train,header_list):
    best_feature=None
    # Train decision tree model and increase Cost Complexity Parameter until the depth reaches 1
    # Insert your code here for task 10
    model = train_decision_tree(x_train, y_train, ccp_alpha=0)
    ccp_alpha = 0

    while tree_depths(model) <= 1:
        model = train_decision_tree(x_train, y_train, ccp_alpha =ccp_alpha)
        ccp_alpha += 0.001
    indices = np.argmax(model.feature_importances_)
    best_feature = header_list[indices]
    return best_feature
# Example usage (Template Main section):
if __name__ == "__main__":
    # Load data
    file_path = "DT.csv"
    num_rows, data, header_list = load_data(file_path)
    print(f"Data is read. Number of Rows: {num_rows}"); 
    print("-" * 50)

    # Filter data
    data_filtered = filter_data(data)
    num_rows_filtered=data_filtered.shape[0]
    print(f"Data is filtered. Number of Rows: {num_rows_filtered}"); 
    print("-" * 50)

    # Data Statistics
    coefficient_of_variation = statistics_data(data_filtered)
    print("Coefficient of Variation for each feature:")
    for header, coef_var in zip(header_list[:-1], coefficient_of_variation):
        print(f"{header}: {coef_var}")
    print("-" * 50)
    # Split data
    x_train, x_test, y_train, y_test = split_data(data_filtered)
    print(f"Train set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    print("-" * 50)
    
    # Train initial Decision Tree
    model = train_decision_tree(x_train, y_train)
    print("Initial Decision Tree Structure:")
    print_tree_structure(model, header_list)
    print("-" * 50)
    
    # Evaluate initial model
    acc_test, recall_test = evaluate_model(model, x_test, y_test)
    print(f"Initial Decision Tree - Test Accuracy: {acc_test:.2%}, Recall: {recall_test:.2%}")
    print("-" * 50)
    # Train Pruned Decision Tree
    model_pruned = train_decision_tree(x_train, y_train, ccp_alpha=0.002)
    print("Pruned Decision Tree Structure:")
    print_tree_structure(model_pruned, header_list)
    print("-" * 50)
    # Evaluate pruned model
    acc_test_pruned, recall_test_pruned = evaluate_model(model_pruned, x_test, y_test)
    print(f"Pruned Decision Tree - Test Accuracy: {acc_test_pruned:.2%}, Recall: {recall_test_pruned:.2%}")
    print("-" * 50)
    # Find optimal ccp_alpha
    optimal_alpha = optimal_ccp_alpha(x_train, y_train, x_test, y_test)
    print(f"Optimal ccp_alpha for pruning: {optimal_alpha:.4f}")
    print("-" * 50)
    # Train Pruned and Optimized Decision Tree
    model_optimized = train_decision_tree(x_train, y_train, ccp_alpha=optimal_alpha)
    print("Optimized Decision Tree Structure:")
    print_tree_structure(model_optimized, header_list)
    print("-" * 50)
    
    # Get tree depths
    depth_initial = tree_depths(model)
    depth_pruned = tree_depths(model_pruned)
    depth_optimized = tree_depths(model_optimized)
    print(f"Initial Decision Tree Depth: {depth_initial}")
    print(f"Pruned Decision Tree Depth: {depth_pruned}")
    print(f"Optimized Decision Tree Depth: {depth_optimized}")
    print("-" * 50)
    
    # Feature importance
    important_feature_name = important_feature(x_train, y_train,header_list)
    print(f"Important Feature for Fraudulent Transaction Prediction: {important_feature_name}")
    print("-" * 50)
        
# References: 
# Here please provide recognition to any source if you have used or got code snippets from
# Please tell the lines that are relavant to that reference.
# For example: 
# Line 80-87 is inspired by a code at https://stackoverflow.com/questions/48414212/how-to-calculate-accuracy-from-decision-trees
# Line 112 - 121 is inspired by a code at https://chat.openai.com/share/cc2df295-d13a-4eb4-9b65-20421b14ead6

