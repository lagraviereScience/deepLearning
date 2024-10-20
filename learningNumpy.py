import numpy as np
import matplotlib.pyplot as plt
from scipy import datasets

def printArray(myArray):
    print("*********************")
    print(myArray)
    print("---------------------")
    print(myArray.size)
    print(myArray.ndim)
    print(myArray.shape)
    print("*********************")
    
A = np.array([1,2,3])
printArray(A)

B = np.zeros((3,2))
printArray(B)

#C = np.ones((3,4,64,545,4545))
#printArray(C)

np.random.seed(0    )
D = np.random.randn(3,4)
printArray(D)

E = np.linspace(0, 10, 20)
printArray(E)
print(np.diff(E).size)



A = np.zeros((3,2))
B = np.ones((3,2))

C = np.hstack((A, B))
printArray(C)

D = np.vstack((A, B))
printArray(D)

E = np.concatenate((A,B), axis=0)
printArray(E)
E = np.concatenate((A,B), axis=1)
printArray(E)


printArray(D)
D = D.reshape((12,1))
printArray(D)

def initWeird(m, n):
    A = np.random.randn(m, n)
    print("Array A:")
    printArray(A)

    print("Array B:")
    B = np.ones((m,1))
    printArray(B)
    #B = B.reshape
    A = np.concatenate((A, B), axis=1)
    return A


W = initWeird(3,2)
print("Final Array:")
printArray(W)



B = np.arange(16)
B = B.reshape(4,4)
printArray(B)

B = B[1:3,1:3]
printArray(B)


C = np.arange(25).reshape((5,5))
printArray(C)
C[::2,::2]="99999"
printArray(C)

A = np.random.randint(0,10, [5,5])
printArray(A)
printArray(A < 5)

A[(A<5) & (A>2)]=10
printArray(A)


A = np.random.randint(0,255, [1024,720])
printArray(A)
A[A>200] = 255
printArray(A)

face = datasets.face(gray=True)
h = face.shape[0]
w = face.shape[1]
zoom_face = np.copy(face[h//4 : -h//4, w//4: -w//4])
zoom_face[zoom_face > 150] = 255
zoom_face[zoom_face < 150] = 0
#plt.imshow(zoom_face, cmap=plt.cm.gray)
#plt.show()

newFace = face[::2, ::2]
plt.imshow(newFace, cmap=plt.cm.gray)
plt.show()

