import numpy as np 
import pandas as pd
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


MODEL_PATH = Path(__file__).with_name("first_model.pkl")


def create_data():
    try:
        X, Y = make_moons(n_samples=1000, noise=0.1, random_state=42)
        Y = Y.reshape(1000, 1)
        return X, Y
    except Exception as e:
        print("Error loading dataset:", e)
        raise


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Binary Cross-Entropy Loss (measures classification error)
def compute_loss(y_true, y_pred):
    # Added small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def calculate_accuracy(y_true, y_pred, threshold=0.5):
    binary_predictions = (y_pred >= threshold).astype(int)
    
    correct_predictions = (binary_predictions == y_true)
    
    accuracy = np.mean(correct_predictions)
    
    return accuracy

class Model:
    def __init__(self, input_tuple, output_tuple, number_of_layers=2, learning_rate=0.1):
        self.lr = learning_rate
        self.input_size = int(input_tuple[1])
        self.hidden_size = int(input_tuple[0])
        self.output_size = int(output_tuple[1])

        # for now keeping no of layers as 2
        # Layer 1 parameters (Input -> Hidden)

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        
        # Layer 2 parameters (Hidden -> Output)
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))


    def train(self,x_train,y_train,epochs=1000):
        for epoch in range(epochs):
            prediction=self.predict(x_train)
            losses=compute_loss(y_train,prediction)
            accuracy=calculate_accuracy(y_train,prediction)
            self.backward(x_train,y_train)
            if epoch % 10==0 :
                print(f"calculated loss : {losses},accuracy : {accuracy}")
    

    def predict(self, X):
        # Hidden layer computation
        
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)

        # Output layer computation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)  # Final prediction

        return self.A2

    def backward(self, X, y):
        m = X.shape[0]

        dZ2 = self.A2 - y
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def save(self, filename):
        state = {
            "learning_rate": self.lr,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }

        with open(filename, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            state = pickle.load(f)

        model = cls(
            (state["hidden_size"], state["input_size"]),
            (1, state["output_size"]),
            learning_rate=state["learning_rate"],
        )
        model.W1 = state["W1"]
        model.b1 = state["b1"]
        model.W2 = state["W2"]
        model.b2 = state["b2"]
        return model


if __name__ == "__main__":
    X, Y = create_data()

    if MODEL_PATH.exists():
        first_model = Model.load(MODEL_PATH)
        print(f"Loaded saved model from {MODEL_PATH}")
    else:
        first_model = Model(X.shape, Y.shape)
        first_model.train(X, Y)
        first_model.save(MODEL_PATH)
        print(f"Saved trained model to {MODEL_PATH}")

    print(first_model.predict(X)[:10])
# try: 
#   (X_train, y_train), (X_test, y_test) = mnist.load_data()
#   print('Data Loaded')
# except Exception as e:
#   print("Error loading MNIST:", e)
#   raise

# image = Image.fromarray(X_train[0])
# image.show()
# print(f"Image size: {image.size}, Label: {y_train[0]}")

# Generate the dataset
# noise=0.1 adds random variance so the moons aren't perfectly clean lines
# X, y = make_moons(n_samples=500, noise=0.1, random_state=42)

# X contains the coordinates: shape (500, 2)
# y contains the labels (0 or 1): shape (500,)

# Plotting the dataset
# plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', label='Class 0')
# plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Class 1')
# plt.legend()
# plt.show()
