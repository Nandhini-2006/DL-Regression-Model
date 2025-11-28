# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate Dataset
np.random.seed(0)
X = np.arange(1, 51).reshape(-1, 1)  # Inputs 1 to 50
e = np.random.normal(0, 5, size=X.shape)  # Random noise
y = 2 * X + 1 + e  # Output

# Convert to PyTorch tensors
X_train = torch.tensor(X, dtype=torch.float32)
y_train = torch.tensor(y, dtype=torch.float32)

# Step 2: Define Neural Network Model
class LinearRegressionNN(nn.Module):
    def __init__(self):
        super(LinearRegressionNN, self).__init__()
        self.linear = nn.Linear(1, 1)  # Input 1, Output 1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionNN()

# Step 3: Define Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Step 4: Train the Model
epochs = 2000
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Step 5: Predict for x = 120
x_test = torch.tensor([[120]], dtype=torch.float32)
y_pred = model(x_test).item()
print(f'Predicted output for x=120: {y_pred:.2f}')

# Optional: Plot the fitted line
plt.scatter(X, y, label='Data')
plt.plot(X, model(X_train).detach().numpy(), color='red', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()


```

### OUTPUT
Training Loss Vs Iteration Plot
Best Fit line plot
Include your plot here

<img width="799" height="856" alt="Screenshot 2025-11-28 183027" src="https://github.com/user-attachments/assets/b4577335-544b-43c5-800e-85745a7c8806" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
