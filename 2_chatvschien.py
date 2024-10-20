import h5py
import numpy as np
#from deep.utilities import load_data
from deepGit.utilities import load_data
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go



def printArray(myArray):
    print("*********************")
    print(myArray)
    print("---------------------")
    print(myArray.size)
    print(myArray.ndim)
    print(myArray.shape)
    print("*********************")

X_train, y_train, X_test, y_test = load_data()

print('dimensions de X:', X_train.shape)
print('dimensions de y:', y_train.shape)
print(X_test.shape)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

#printArray(X_train[0])
print(type(X_train[0]))

#plt.imshow(X_train[0], cmap='gray')  # 'gray' colormap for grayscale
#plt.colorbar()  # optional, to show the color scale
#plt.show()

def initialisation(X):
    W = np.random.randn(X_train_flat.shape[1], 1)*0.01
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
#   Z = X * W + b
#   A = 1 / (1+e^-Z)
    Z = X.dot(W) + b
    # Calculate statistics of Z for clipping
    mean = np.mean(Z)
    std_dev = np.std(Z)
    
    # Set clipping limits
    clip_lower = mean - 1.2 * std_dev
    clip_upper = mean + 1.2 * std_dev
    
    # Clip Z
    Z_clipped = np.clip(Z, clip_lower, clip_upper)
    
    # Calculate activation
    A = 1 / (1 + np.exp(-Z_clipped))
    #A = 1 / (1 + np.exp(-Z))
    return A

def logLoss(A, y):
    epsilon = 1e-8 
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A+ epsilon)) #m = len(y)

def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y) #jacobien -- X.T c'est la transposée de X
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    #print(type(dW))
    return (W, b)

def predict(X, W, b):
    A = model(X, W, b)
    # print(A)
    return A >= 0.5


def artificial_neuron(X, y, learning_rate = 0.1, n_iter = 6000):
    # initialisation W, b
    W, b = initialisation(X)

    Loss = []
    history = []

    for i in range(n_iter):
        A = model(X, W, b)
        Loss.append(logLoss(A, y))
        dW, db = gradients(A, X, y)
        #dW=-1
        #db=-1
        W, b = update(dW, db, W, b, learning_rate)
        history.append([W, b, Loss, i])

    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred))

    #printArray(Loss)

    #VISUALISATION 
    #plt.plot(Loss)
    #plt.show()

    return (W, b)

W, b = artificial_neuron(X_train_flat, y_train)
