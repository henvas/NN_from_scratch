import numpy as np
import mnist
import matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report, confusion_matrix

'''
Didn't see the jupyter notebook example so did some of the parts a little different. For example
we didn't do the bias trick until task 3. We therefore had to update the bias in the training loop. 
Sklearn library was just a helpful tool to check confusion matrix
'''


X_train, Y_train, X_test, Y_test = mnist.load()


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def compute_loss(Y, A, W, lamb):
    N = Y.shape[0]
    loss = -np.mean(np.multiply(np.log(A),Y)+np.multiply(np.log(1 - A), (1 - Y))) + lamb*np.reshape(np.dot(W.T, W),())
    #loss = -(1./N)*(np.sum(np.multiply(np.log(A),Y))+np.sum(np.multiply(np.log(1 - A),(1-Y)))+lamb*np.reshape(np.dot(W.T, W),()))
    return loss


def prediction(W, X, b):
    Z = np.matmul(W.T, X.T) + b
    A = sigmoid(Z)
    return A


#   Split data up in only 2's and 3's, as well as split the training data and validation data
#   Also get the last amount of data for test data
#   Should split this function up :-)
def twos_n_threes_data(X, Y, amount, val=False):
    x_val = np.empty((amount // 10, 784))
    y_val = np.zeros((amount // 10,))
    idx = 0
    valid_idx = 0
    data_idx = 0

    #   If training data, else test data:
    if val:
        Y_2 = np.zeros((int(amount * 0.9),))
        X_2 = np.empty((int(amount * 0.9), 784))

        for i in range(0, len(Y)):
            if Y[i] == 2 or Y[i] == 3:
                if idx == amount:
                    break
                # split 10% into validation data
                if idx % 10 == 0:
                    if Y[i] == 2: y_val[valid_idx] = 1
                    x_val[valid_idx] = X[i]
                    valid_idx += 1
                    idx += 1
                else:
                    if Y[i] == 2: Y_2[data_idx] = 1
                    X_2[data_idx] = X[i]
                    data_idx += 1
                    idx += 1
    else:
        # Get last of the test data
        Y_2 = np.zeros((amount,))
        X_2 = np.empty((amount, 784))
        for i in range(len(Y)-1, 0, -1):
            if Y[i] == 2 or Y[i] == 3:
                if idx == amount:
                    break
                if Y[i] == 2: Y_2[idx] = 1
                X_2[idx] = X[i]
                idx += 1
    Y = Y_2
    X = X_2

    return X, Y, x_val, y_val


X_train, Y_train, X_val, Y_val = twos_n_threes_data(X_train, Y_train, 20000, val=True)
X_test, Y_test, _, _ = twos_n_threes_data(X_test, Y_test, 2000)

#   Normalize to keep our gradients manageable
X_train = X_train/255
X_test = X_test/255
X_val = X_val/255

#   Shuffle training data
perm = np.random.permutation(X_train.shape[0])
X = X_train[perm]
Y = Y_train[perm]


#   Hyper parameters
n_x = X.shape[1]
m = X.shape[0]

learning_rate_init = 1
T = 90
lamb = 0.0001

#   Init weights and bias (tried with different methods to avoid exploding/vanishing gradients)
W = np.random.randn(n_x, 1)*0.01
b = np.zeros((1, 1))

#   Plot lists
epochs = []
err = []
test_err = []
val_err = []
train_acc = []
test_acc = []
val_acc = []
Y_predictions = []
weight_lengths = []
train_iter = 0

for epoch in range(30):
    t0 = time.clock()
    learning_rate = learning_rate_init / (1 + epoch/ T)
    # learning_rate = learning_rate_init

    #for t in range(0, 10):

    #   Forward
    Z = np.dot(W.T, X.T) + b
    A = sigmoid(Z)

    # tn = training data, yn = A

    #   Calculate for test and validation data as well
    A_test = prediction(W, X_test, b)
    A_val = prediction(W,X_val,b)

    loss = compute_loss(Y, A, W, lamb)
    test_loss = compute_loss(Y_test, A_test, W, lamb)
    val_loss = compute_loss(Y_val, A_val, W, lamb)

    #   backwards on the training data only
    Ew = (1/m) * (np.matmul(X.T, (A-Y).T) + 2*lamb*W)
    Eb = (1 / m) * (np.sum(A - Y, axis=1, keepdims=True))

    #   SGD
    W = W - learning_rate * Ew
    b = b - learning_rate * Eb

    #   Accuracies and losses for plotting
    pred_train = (prediction(W, X, b) > .5)[0, :]
    pred_test = (prediction(W, X_test, b) > .5)[0, :]
    pred_val = (prediction(W, X_val, b) > .5)[0, :]

    test_acc.append(100*np.sum(pred_test==Y_test)/len(pred_test))
    val_acc.append(100*np.sum(pred_val==Y_val)/len(pred_val))
    train_acc.append(100*np.sum(pred_train==Y)/len(Y))
    err.append(loss)
    test_err.append(test_loss)
    val_err.append(val_loss)
    weight_lengths.append(W)
    epochs.append(epoch)
    #train_iter += 1

    if (epoch % 1 == 0):
        print("Epoch: %d, Loss: %.8f, Error: %.8f, train acc = %.2f, test acc = %.2f, val acc = %.2f,Time: %.4fs"
              % (epoch, loss, np.mean(err), train_acc[-1], test_acc[-1], val_acc[-1], time.clock() - t0))

print("Final cost:", loss)
print("Final learning rate:", learning_rate)

np.savetxt('weights.txt', W, fmt='%f')

#   Plot accuracy and losses
plt.subplot(2,1,1)
plt.plot(epochs, err)
plt.plot(epochs, test_err)
plt.plot(epochs, val_err)
plt.legend(['Train', 'Test', 'Val'], loc='upper right')
plt.title('Loss')
plt.xlabel('Epochs')

plt.subplot(2,1,2)
plt.plot(epochs, train_acc)
plt.plot(epochs, test_acc)
plt.plot(epochs, val_acc)
plt.title('Accuracies')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test', 'Val'], loc='lower right')
plt.show()

#   Plot weights (not image)
for weights in weight_lengths:
    plt.plot(weights)
plt.title('Weights')
plt.xlabel('Pixels')
plt.show()


#   plot confusion matrix and predictions
Z = np.matmul(W.T, X_test.T) + b
A = sigmoid(Z)


predictions = (A>.5)[0,:]
labels = (Y_test == 1)

print(confusion_matrix(predictions, labels))
print(classification_report(predictions, labels))













