import numpy as np
import pandas as pd
data = pd.read_csv('mnist_train.csv')
data.head()
#splitting the data
m,n = data.shape
data = np.array(data)
m,n = data.shape
np.random.shuffle(data)
test_data = data[0:1000].T
y_test = test_data[0]
x_test = test_data[1:n]
x_test = x_test / 255.
train_data = data[1000:m].T
y_train = train_data[0]
x_train = train_data[1:n]
x_train = x_train / 255.
_,m_train = x_train.shape
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#making the neural network
def init_parameters():
    W1 = np.random.randn(10,784)-0.5
    b1 = np.random.randn(10,1)-0.5
    W2 = np.random.randn(10,10)-0.5
    b2 = np.random.randn(10,1)-0.5
    return W1,b1,W2,b2

def Relu(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1,b1,W2,b2,X):
    Z1 = np.dot(W1,X)+b1
    A1 = Relu(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

def y_vector(Y):
    Y_vector = np.zeros((Y.size, Y.max() + 1))
    Y_vector[np.arange(Y.size), Y] = 1
    Y_vector = Y_vector.T
    return Y_vector

def Relu_deriv(Z):
    return Z > 0

def Back_prop(Z1,A1,Z2,A2,Y,X,W1,W2):
    Y_vector = y_vector(Y)
    dZ2 = A2 - Y_vector
    dW2 = 1/m * np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2)
    dZ1 = np.dot(W2.T,dZ2)*Relu_deriv(Z1)
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m* np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_param(dW1,db1,dW2,db2,W1,W2,b1,b2,alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    return W1,b1,W2,b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X,Y,alpha,iterations):
    W1,b1,W2,b2 = init_parameters()
    for i in range(iterations):
        Z1,A1,Z2,A2 =forward_prop(W1,b1,W2,b2,X)
        dW1, db1, dW2, db2 = Back_prop(Z1,A1,Z2,A2,Y,X,W1,W2)
        W1,b1,W2,b2 = update_param(dW1,db1,dW2,db2,W1,W2,b1,b2,alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
    
W1, b1, W2, b2 = gradient_descent(x_train, x_train, 0.10, 500)
    
    
