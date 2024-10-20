import numpy as np
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


amountOfFeatures = 2
sampleSize = 100

X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=4, random_state=42)
y = y.reshape((y.shape[0], 1))

print('dimensions de X:', X.shape)
print('dimensions de y:', y.shape)
print(X[0])
exit()
plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
plt.show()

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)

def model(X, W, b):
#   Z = X * W + b
#   A = 1 / (1+e^-Z)
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def logLoss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A)) #m = len(y)

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
        W, b = update(dW, db, W, b, learning_rate)
        history.append([W, b, Loss, i])

    y_pred = predict(X, W, b)
    print(accuracy_score(y, y_pred))

    #printArray(Loss)

    #VISUALISATION 
    #plt.plot(Loss)
    #plt.show()

    return (W, b)

W, b = artificial_neuron(X, y)


############ PREDIRE UNE NOUVELLE PLANTE
#new_plant= np.array([2,1])
#plt.scatter(X[:,0], X[:, 1], c=y, cmap='summer')
#plt.scatter(new_plant[0],new_plant[1], c='r')
#plt.show()

#print(predict(new_plant, W, b))

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:,0], X[:, 1], c=y, cmap='summer')

x1 = np.linspace(-1, 4, 100)
x2 = ( - W[0] * x1 - b) / W[1]

ax.plot(x1, x2, c='orange', lw=3)
plt.show()



###### VISUALISATION
fig = go.Figure(data=[go.Scatter3d( 
    x=X[:, 0].flatten(),
    y=X[:, 1].flatten(),
    z=y.flatten(),
    mode='markers',
    marker=dict(
        size=5,
        color=y.flatten(),                
        colorscale='YlGn',  
        opacity=0.8,
        reversescale=True
    )
)])

fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()


# In[17]:


X0 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
X1 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
xx0, xx1 = np.meshgrid(X0, X1)
Z = W[0] * xx0 + W[1] * xx1 + b
A = 1 / (1 + np.exp(-Z))

fig = (go.Figure(data=[go.Surface(z=A, x=xx0, y=xx1, colorscale='YlGn', opacity = 0.7, reversescale=True)]))

fig.add_scatter3d(x=X[:, 0].flatten(), y=X[:, 1].flatten(), z=y.flatten(), mode='markers', marker=dict(size=5, color=y.flatten(), colorscale='YlGn', opacity = 0.9, reversescale=True))


fig.update_layout(template= "plotly_dark", margin=dict(l=0, r=0, b=0, t=0))
fig.layout.scene.camera.projection.type = "orthographic"
fig.show()
