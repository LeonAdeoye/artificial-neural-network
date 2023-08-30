import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from perceptron import prepare_data


def normalize_data(df):
    # Apply Min-Max scaling to normalize the data
    return (df - df.min()) / (df.max() - df.min())


# Calculate correlation matrix manually.
def calculate_correlation_matrix(predictions, actual_values):
    array1 = predictions.view(-1)
    array2 = actual_values.view(-1)

    mean1 = array1.mean()
    mean2 = array2.mean()

    # Calculate the centered arrays
    centered1 = array1 - mean1
    centered2 = array2 - mean2

    # Calculate covariance
    covariance = torch.dot(centered1, centered2) / (len(array1) - 1)

    # Calculate standard deviations
    std_dev1 = torch.sqrt(torch.sum(centered1 ** 2) / (len(array1) - 1))
    std_dev2 = torch.sqrt(torch.sum(centered2 ** 2) / (len(array2) - 1))

    # Calculate the correlation coefficient
    correlation_coefficient = covariance / (std_dev1 * std_dev2)
    print("Correlation Coefficient:", correlation_coefficient.item())


def read_concrete_data():
    # Read the Excel file into a DataFrame
    df = pd.read_csv("concrete.csv")

    # Set pandas display options to show all columns and rows without truncation
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    # Get summary statistics of all columns
    # print(df.describe(include='all'))

    ndf = normalize_data(df)

    # Specify the row number where you want to split
    split_row_number = int(0.75 * len(ndf))

    # Split the normalized DataFrame into training and testing sets based on row number
    train_df = ndf.iloc[:split_row_number]
    # print(train_df.head())
    # print(train_df.shape[0])
    test_df = ndf.iloc[split_row_number:]

    X_train = train_df.drop(columns=['strength'])  # Drop the 'strength' column to get predictors 773 x 8
    # print(X_train.head())
    y_train = train_df['strength']  # Target variable strength 773 x 1 from training data
    X_test = test_df.drop(columns=['strength'])  # 258 x 8 test data to be used later when model is created.
    y_test = test_df['strength']   # 258 x 1 target value from test data

    # Convert data to NumPy arrays
    X_train_array = X_train.to_numpy()
    y_train_array = y_train.to_numpy()
    X_test_array = X_test.to_numpy()
    y_test_array = y_test.to_numpy()

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_array, dtype=torch.float32)
    # print(X_train_tensor)
    y_train_tensor = torch.tensor(y_train_array, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test_array, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_array, dtype=torch.float32).unsqueeze(1)  # Adding extra dimension

    # Neural net class without hidden layer
    class NeuralNetworkWIHTOUTHiddenLayer(nn.Module):
        def __init__(self):
            super(NeuralNetworkWIHTOUTHiddenLayer, self).__init__()
            self.fc1 = nn.Linear(X_train_tensor.shape[1], 100)
            self.fc2 = nn.Linear(100, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Neural net class with hidden layer
    class DeepNeuralNetworkWithHiddenLayer(nn.Module):
        def __init__(self):
            super(DeepNeuralNetworkWithHiddenLayer, self).__init__()
            self.fc1 = nn.Linear(X_train_tensor.shape[1], 100)
            self.fc2 = nn.Linear(100, 5, bias=True)  # New hidden layer with 5 nodes and with bias terms
            self.fc3 = nn.Linear(5, 1, bias=True)  # Output layer taking signals from 5 hidden nodes and with bias terms

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))  # Applying ReLU activation to the hidden layer
            x = self.fc3(x)  # Output layer
            return x

    # Creates an instance of the NeuralNetwork class that you've defined above using PyTorch.
    # The NeuralNetwork class inherits from nn.Module, which is a base class provided by PyTorch for building neural network models.
    model = DeepNeuralNetworkWithHiddenLayer()

    # Define loss function and optimizer
    # The nn.MSELoss() (Mean Squared Error Loss) computes the mean squared difference between the predicted outputs and the actual target values.
    criterion = nn.MSELoss()
    # The optim.Adam() function creates an Adam optimizer that will update the model's parameters during training.
    # It takes the model's parameters and a learning rate (lr) as arguments.
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(4882):  # using 4882 steps or epochs. This can be set arbitrarily.
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    with torch.no_grad():  # Disable gradient tracking during prediction
        model.eval()  # Set the model to evaluation mode
        predictions = model(X_test_tensor)

        print(predictions)
        print(y_test_tensor)
        calculate_correlation_matrix(predictions, y_test_tensor)


if __name__ == '__main__':
    #  read_concrete_data()
    prepare_data()

