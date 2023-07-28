import random

def train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=None):
    """
    Splits the input features and corresponding labels into training and testing sets.

    Parameters:
        x_dataset (list or array-like): The input feature dataset to be split.
        y_dataset (list or array-like): The corresponding label dataset to be split.
        test_size (float, optional): The proportion of the dataset to be used for testing.
                                     Defaults to 0.2 (20%).
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        tuple: Four lists, x_train, x_test, y_train, y_test, representing the training
               and testing sets for input features and corresponding labels.
    """

    if random_state:
        random.seed(random_state)
    
    # Create a list of indices corresponding to the length of the input feature dataset
    data_index = list(range(len(x_dataset)))
    # Shuffle the indices randomly
    random.shuffle(data_index)

    # Calculate the index at which to split the shuffled dataset to get the desired test size
    train_length = int(len(x_dataset) * (1 - test_size))
    # Extract the first train_length elements as the training indices
    x_train_indices = data_index[:train_length]
    # Extract the remaining elements as the testing indices
    x_test_indices = data_index[train_length:]

    # Create the training and testing datasets for input features
    x_train = [x_dataset[i] for i in x_train_indices]
    x_test = [x_dataset[i] for i in x_test_indices]

    # Create the training and testing datasets for corresponding labels
    y_train = [y_dataset[i] for i in x_train_indices]
    y_test = [y_dataset[i] for i in x_test_indices]

    # Return the four datasets as a tuple
    return x_train, x_test, y_train, y_test
