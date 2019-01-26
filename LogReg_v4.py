import numpy as np
import mnist
import matplotlib.pyplot as plt
import time


'''
For task 3 we used some of the methods presented in the jupyter notebook example
so that we didn't have to deal with the bias, unlike the previous tasks. 
'''

X_train, Y_train, X_test, Y_test = mnist.load()
# Cannot print weights and also do the bias trick here
#X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
#X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)


def softmax(Z):
    Z -= np.max(Z)
    return (np.exp(Z).T / np.sum(np.exp(Z),axis=1)).T


def accuracy(W, X, Y):
    A = softmax(np.dot(X,W))
    prediction = np.argmax(A,axis=1)
    acc = sum(prediction == Y) / (float(len(Y)))
    return acc


def compute_loss(Y, A, W, lamb):
    N = Y.shape[0]
    loss = (-1/N) * np.sum(Y * np.log(A))+ lamb*np.sum(W*W)
    return loss


def train_val_split(X, Y, val_percentage):
    """
      Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
      --
      X: [N, num_features] numpy vector,
      Y: [N, 1] numpy vector
      val_percentage: amount of data to put in validation set
    """
    dataset_size = X.shape[0]
    idx = np.arange(0, dataset_size)
    np.random.shuffle(idx)

    train_size = int(dataset_size * (1 - val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val


#   Use the first 20000 images
X_train = X_train[:20000]
Y_train = Y_train[:20000]

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)

#   2000 last images for test
X_test = X_test[X_test.shape[0]-2000:]
Y_test = Y_test[Y_test.shape[0]-2000:]


#   Normalize to keep our gradients manageable
X_train = X_train/255
X_test = X_test/255
X_val = X_val/255

#   No reason to do this, will revert whenever I remember
X = X_train
Y = Y_train

#   Not one-hot encoded labels for checking accuracy later
y_train_acc = Y_train
y_test_acc = Y_test
y_val_acc = Y_val

#   One-hot encoding
digits = 10
examples_train = Y.shape[0]
examples_val = Y_val.shape[0]
examples_test = Y_test.shape[0]
Y = Y.reshape(examples_train, 1)
Y_val = Y_val.reshape(examples_val, 1)
Y_test = Y_test.reshape(examples_test, 1)

Y = np.eye(digits)[Y.astype('int32')]
Y_val = np.eye(digits)[Y_val.astype('int32')]
Y_test = np.eye(digits)[Y_test.astype('int32')]
Y = Y.reshape(examples_train, digits)
Y_val = Y_val.reshape(examples_val, digits)
Y_test = Y_test.reshape(examples_test, digits)


#   Hyper parameters
n_x = X.shape[1]
m = X.shape[0]


learning_rate_init = 1
T = 50
lamb = 0.0001

#   Init weights (tried with different methods to avoid exploding/vanishing gradients)
W = np.random.randn(n_x, digits)*np.sqrt(1/m)

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

for epoch in range(50):
    t0 = time.clock()
    learning_rate = learning_rate_init / (1 + epoch/ T)
    # learning_rate = learning_rate_init

    #for t in range(0, 10):

    #   Forward, Softmax
    Z = np.dot(W.T, X.T)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    A = A.T

    # tn = training data, yn = A

    #   Calculate for test and validation data as well
    A_test = np.exp(np.dot(W.T, X_test.T)) / np.sum(np.exp(np.dot(W.T, X_test.T)), axis=0)
    A_val = np.exp(np.dot(W.T, X_val.T)) / np.sum(np.exp(np.dot(W.T, X_val.T)), axis=0)
    A_test = A_test.T
    A_val = A_val.T


    loss = compute_loss(Y, A, W, lamb)
    test_loss = compute_loss(Y_test, A_test, W, lamb)
    val_loss = compute_loss(Y_val, A_val, W, lamb)

    #   backwards on the training data only
    Ew = (1/m) * np.matmul(X.T, (A-Y)) + 2*lamb*W

    #   SGD
    W = W - learning_rate * Ew

    #   Accuracies and losses for plotting
    test_acc.append(accuracy(W, X_test, y_test_acc))
    val_acc.append(accuracy(W, X_val, y_val_acc))
    train_acc.append(accuracy(W, X_train, y_train_acc))

    err.append(loss)
    test_err.append(test_loss)
    val_err.append(val_loss)
    weight_lengths.append(W)
    epochs.append(epoch)
    #train_iter += 1

    if (epoch % 1 == 0):
        print("Epoch: %d, Loss: %.8f, Error: %.8f, train acc = %.2f, test acc = %.2f, val acc = %.2f,Time: %.4fs"
              % (epoch, loss, np.mean(err), train_acc[-1], test_acc[-1], val_acc[-1], time.clock() - t0))
    if epoch == 49:
        print(Y[0] == (A[0] > .5))
        print(Y[0])
        for p in A[0]:
            if p > 0.5:
                print(1, end=" ")
            else:
                print(0, end=" ")

print("\n")
print("Final cost:", loss)
print("Final learning rate:", learning_rate)


np.savetxt('weights_3.txt', W, fmt='%f')

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


#   Plot final weights for each digit
Nr = 2
Nc = 5
fig, axs = plt.subplots(Nr, Nc)
fig.suptitle('Weights')
images = []
idx = 0
for i in range(Nr):
    for j in range(Nc):
        #   Or maybe just W_print = W[:, idx].reshape(28,28)
        W_print = weight_lengths[idx][:,idx].reshape(28,28)
        idx += 1
        images.append(axs[i, j].imshow(W_print, cmap=plt.get_cmap('seismic')))
        axs[i, j].label_outer()
plt.show()


#   Average weight over training for each digit
fig2, axs2 = plt.subplots(Nr, Nc)
fig2.suptitle('Average weights')
images = []
idx = 0
W_avg = np.empty((784, 10))
avg = []
for digit in range(10):
    for i in range(784):
        for j in range(len(weight_lengths)):
            avg.append(weight_lengths[j][i, digit])
        W_avg[i][digit] = np.mean(avg)
        avg = []

for i in range(Nr):
    for j in range(Nc):
        W_print = W_avg[:, idx].reshape(28,28)
        idx += 1
        images.append(axs2[i, j].imshow(W_print, cmap=plt.get_cmap('seismic')))
        axs2[i, j].label_outer()
plt.show()















