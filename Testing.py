from Week1_perceptron import MODEL_PATH, Model,calculate_accuracy
from sklearn.datasets import make_moons

model = Model.load(MODEL_PATH)

try:
    X, Y = make_moons(n_samples=2000, noise=0.1, random_state=42)
    Y = Y.reshape(2000, 1)
except Exception as e:
        print("Error loading dataset:", e)
        raise

predictions = model.predict(X)

for i in range(100):
  print(predictions[i]," : ", Y[i])
print("accuracy ",calculate_accuracy(Y,predictions))