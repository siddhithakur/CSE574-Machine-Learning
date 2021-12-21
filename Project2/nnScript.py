import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import time
# import pandas as pd
# import pickle
# import matplotlib.pyplot as plt

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  (1.0 / (1.0 + np.exp(-z)))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_tmp = np.asarray([])
    test_tmp = np.asarray([])
    for i in range(0, 10):
        # print(mat['train'+str(i)].shape)
        labels = np.ones((mat['train' + str(i)].shape[0], 1))
        labels[:, 0] = i

        columns = mat['train' + str(i)].shape[1] + 1
        train_tmp = np.append(train_tmp, np.concatenate((mat['train' + str(i)], labels), axis=1))

        labels_test = np.ones((mat['test' + str(i)].shape[0], 1))
        labels_test[:, :] = i

        columns_test = mat['test' + str(i)].shape[1] + 1
        test_tmp = np.append(test_tmp, np.concatenate((mat['test' + str(i)], labels_test), axis=1))

    labeled_train = train_tmp.reshape((-1, columns))
    # print(labeled_train)
    labeled_test = test_tmp.reshape((-1, columns))



    # Feature selection
    # Your code here.
    col_same_value = np.all(labeled_train == labeled_train[0], axis=0)
    # print(col_same_value)
    labeled_train = labeled_train[:, ~col_same_value]
    labeled_test = labeled_test[:, ~col_same_value]
    # features=[]
    features = [i for i in range(len(col_same_value) - 1) if col_same_value[i] == False]
    print(features)
    choice = np.random.choice(range(labeled_train.shape[0]), size=(50000,), replace=False)
    ind = np.zeros(labeled_train.shape[0], dtype=bool)
    ind[choice] = True
    rest = ~ind

    train = labeled_train[ind]
    # print(train)
    validation = labeled_train[ind == False]
    train_data = train[:, :-1]
    train_label = train[:, -1]
    train_label = train_label.astype(int)
    train_data = train_data / 255.0
    # print(train_data)
    validation_data = validation[:, :-1]
    validation_data = validation_data / 255.0
    validation_label = validation[:, -1]
    test_data = labeled_test[:, :-1]
    test_data = test_data / 255.0
    test_label = labeled_test[:, -1]
    test_label = test_label.astype(int)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #
    train_plus_bias = np.concatenate((np.ones((training_data.shape[0], 1)), training_data), axis=1)
    aj = np.dot(train_plus_bias, w1.T)
    zj = sigmoid(aj)

    z_plus_bias = np.concatenate((np.ones((zj.shape[0], 1)), zj), axis=1)
    bl = np.dot(z_plus_bias, w2.T)
    ol = sigmoid(bl)
    training_label = training_label.astype(int)

    yl = np.eye(np.max(training_label) + 1)[training_label]

    jw1_w2 = (-1 / training_data.shape[0]) * (
        np.sum(np.multiply(yl, np.log(ol)) + np.multiply((1.0 - yl), np.log(1.0 - ol))))
    delta_l = ol - yl
    n = training_data.shape[0]

    gradient2 = np.dot(delta_l.T, z_plus_bias)

    gradient1 = np.dot(np.transpose(np.dot(delta_l, w2) * (z_plus_bias * (1.0 - z_plus_bias))), train_plus_bias)
    gradient1 = gradient1[1:, :]

    obj_val = jw1_w2 + (lambdaval / (2 * n)) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))

    grad_w1 = (1 / n) * (gradient1 + lambdaval * w1)
    grad_w2 = (1 / n) * (gradient2 + lambdaval * w2)


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    n = data.shape[0]

    data = np.concatenate((np.ones((data.shape[0], 1)), data), axis=1)
    aj = np.dot(data, w1.T)
    zj = sigmoid(aj)

    z_plus_bias = np.concatenate((np.ones((zj.shape[0], 1)), zj), axis=1)
    bl = np.dot(z_plus_bias, w2.T)
    ol = sigmoid(bl)

    labels = np.argmax(ol, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""
start_time = time.time()
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 28

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')





# result_dict={}
# result_dict['selected_features']=[12, 13, 14, 15, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779]
# result_dict['n_hidden']=n_hidden
# result_dict['w1']=w1
# result_dict['w2']=w2
# result_dict['lambda']=lambdaval
# with open('params.pickle', 'wb') as handle:
#     pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Following code is used to train for range of hidden and lambda values
# train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
# lambdavalues = np.arange(0, 60, 10)
# n_hidden_layers = np.arange(4, 30, 4)
#
# time_needed = []
# training_accuracies = []
# test_accuracies = []
# validation_accuracies = []
# n_input = train_data.shape[1]
# lambda_list = []
# n_hidden_list = []
#
# w1_list = []
# w2_list = []
# for lambdavalue in lambdavalues:
#
#     for n_hidden in n_hidden_layers:
#         print((train_label))
#         #  Train Neural Network
#         start_time = time.time()
#         # set the number of nodes in input unit (not including bias unit)
#
#         # set the number of nodes in hidden unit (not including bias unit)
#         # n_hidden = 50
#
#         # set the number of nodes in output unit
#         n_class = 10
#
#         # initialize the weights into some random matrices
#         initial_w1 = initializeWeights(n_input, n_hidden)
#         initial_w2 = initializeWeights(n_hidden, n_class)
#
#         # unroll 2 weight matrices into single column vector
#         initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
#
#         # set the regularization hyper-parameter
#         lambdaval = 0
#
#         args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
#
#         # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
#
#         opts = {'maxiter': 50}  # Preferred value.
#
#         nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#
#         # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#         # and nnObjGradient. Check documentation for this function before you proceed.
#         # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)
#
#         print("--- %s seconds ---" % (time.time() - start_time))
#         # Reshape nnParams from 1D vector into w1 and w2 matrices
#         w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
#         w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
#
#         predicted_label = nnPredict(w1, w2, train_data)
#
#         # find the accuracy on Training Dataset
#
#         print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
#         training_accuracies.append(100 * np.mean((predicted_label == train_label).astype(float)))
#         predicted_label = nnPredict(w1, w2, validation_data)
#
#         # find the accuracy on Validation Dataset
#
#         print('\n Validation set Accuracy:' + str(
#             100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
#         validation_accuracies.append(100 * np.mean((predicted_label == validation_label).astype(float)))
#         predicted_label = nnPredict(w1, w2, test_data)
#
#         # find the accuracy on Validation Dataset
#
#         print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
#         test_accuracies.append(100 * np.mean((predicted_label == test_label).astype(float)))
#
#         time_needed.append(time.time() - start_time)
#         n_hidden_list.append(n_hidden)
#         lambda_list.append(lambdavalue)
#         w1_list.append(w1)
#         w2_list.append(w2)
#

#check the values
# acc_comparisons = pd.DataFrame(list(zip(lambda_list, n_hidden_list, training_accuracies, validation_accuracies, test_accuracies, time_needed)),
#                       columns=['λ', 'hidden_units','Train_Accuracy', 'Validation_Accuracy', 'Test_Accuracy', 'Training_Time'])
# acc_comparisons = acc_comparisons.sort_values(by=['Test_Accuracy'], ascending=False)
#print(acc_comparisons)


# lambda_optimal = acc_comparisons.iloc[0,0]
# hidden_optimal = acc_comparisons.iloc[0,1]
#
#
# plt.figure(figsize=(10,5))
# plt.title('Training Time vs Hidden Units')
# plt.xlabel('Hidden Units')
# plt.ylabel('Training Time')
# plt.xticks( np.arange( 4,34, step=4))
# plt.yticks()
# time_vs_hidden=acc_comparisons[acc_comparisons.λ==optimal_lambda]
# plt.plot(time_vs_hidden.hidden_units, time_vs_hidden.Training_Time,  color='c')
#
# plt.show()
#
# plt.figure(figsize=(10,5))
# plt.title('Accuracy vs Hidden Units for Optimal lambda: '+str(lambda_optimal))
# plt.xlabel('Hidden Units')
# plt.ylabel('Accuracy')
#
# acc_vs_hidden=acc_comparisons[acc_comparisons.λ==lambda_optimal]
# plt.plot(acc_vs_hidden.hidden_units, acc_vs_hidden.Train_Accuracy,  color='g')
# plt.plot(acc_vs_hidden.hidden_units, acc_vs_hidden.Test_Accuracy,  color='c')
# plt.plot(acc_vs_hidden.hidden_units, acc_vs_hidden.Validation_Accuracy,  color='r')
# plt.legend(('Training Accuracy','Testing Accuracy','Validation Accuracy'))
# plt.show()
# plt.figure(figsize=(10,5))
# plt.title('Accuracy vs Lambda for Optimal hidden_units: '+str(hidden_optimal))
# plt.xlabel('Lambda')
# plt.ylabel('Accuracy')
#
# time_vs_lambda=acc_comparisons[acc_comparisons.hidden_units==hidden_optimal]
# print(time_vs_lambda.head())
# plt.plot(sorted(time_vs_lambda.λ), time_vs_lambda.Train_Accuracy,  color='g')
# plt.plot(sorted(time_vs_lambda.λ), time_vs_lambda.Test_Accuracy,  color='c')
# plt.plot(sorted(time_vs_lambda.λ), time_vs_lambda.Validation_Accuracy,  color='r')
# plt.legend(('Training Accuracy','Testing Accuracy','Validation Accuracy'))
#
#
# plt.show()
#
# fig = plt.figure(figsize=(10,7))
# ax = fig.add_subplot(projection='3d')
# ax.scatter3D(data.iloc[:,0],data.iloc[:,1], data.iloc[:,4],c=data.iloc[:,4], cmap='Greens')
# ax.set_xlabel('Lambda')
# ax.set_ylabel('hidden units')
# ax.set_zlabel('Test Accuracy')
