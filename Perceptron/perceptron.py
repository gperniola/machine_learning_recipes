import sys
from matplotlib import pyplot as plt
import numpy as np


def predict(inputs, weights):
    sum = 0.0
    threshold = 0.0

    #print("SUM = ")
    for i,w in zip(inputs,weights):
        sum = sum + i*w
        #print("(%f * %f) + ", i, w)

    if sum >= threshold:
        #print(">= %f - R:1", threshold)
        return 1
    else:
        #print("< %f - R:0", threshold)
        return 0

def accuracy(matrix, weights):
    num_correct = 0.0
    predictions = []
    for i in range(len(matrix)):
        pred = predict(matrix[i][:-1], weights)
        predictions.append(pred)
        if pred == matrix[i][-1]:
            num_correct = num_correct + 1
    print("Predictions: ", predictions)
    return num_correct/float(len(matrix))


def plot_matrix(matrix, weights, title="Prediction Matrix"):

    if weights != None:
        map_min = 0.0
        map_max = 1.1
        x_res = 0.001
        y_res = 0.001
        xs = np.arange(map_min, map_max, x_res)
        ys = np.arange(map_min, map_max, y_res)
        zs=[]
        for cur_y in np.arange(map_min,map_max,y_res):
            for cur_x in np.arange(map_min,map_max,x_res):
                #print("x: ", cur_x, " y: ", cur_y)
                zs.append(predict([1.0,cur_x,cur_y],weights))

        xs,ys = np.meshgrid(xs,ys)
        zs = np.array(zs)
        zs = zs.reshape(xs.shape)
        cp = plt.contourf(xs,ys,zs,levels=[-1,-0.0001,0,1],colors=('b','r'), alpha=0.1)

    c0_data=[[],[]]
    c1_data=[[],[]]

    for i in range(len(matrix)):
        cur_i1 = matrix[i][1]
        cur_i2 = matrix[i][2]
        cur_y  = matrix[i][3]

        if cur_y ==1:
            c1_data[0].append(cur_i1)
            c1_data[1].append(cur_i2)
        else:
            c0_data[0].append(cur_i1)
            c0_data[1].append(cur_i2)

    plt.xticks(np.arange(0.0,1.1,0.1))
    plt.yticks(np.arange(0.0,1.1,0.1))
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)

    c0s = plt.scatter(c0_data[0], c0_data[1], s=40.0, c='r', label = "Class -1")
    c1s = plt.scatter(c1_data[0], c1_data[1], s=40.0, c='b', label = "Class 1")

    plt.legend(fontsize=10,loc=1)
    plt.show()

    return


def train_weights(matrix, weights, nb_epoch=10, l_rate=1.00, do_plot=False, stop_early=True, verbose=True):
    for epoch in range(nb_epoch):
        cur_acc = accuracy(matrix,weights)
        print("Epoch ", epoch, " Weights: ", weights)
        print("Accuracy: ", cur_acc)

        if cur_acc == 1.0 and stop_early: break
        title_string = "Epoch " + str(epoch)
        if do_plot: plot_matrix(matrix, weights, title=title_string)

        for i in range(len(matrix)):
            prediction = predict(matrix[i][:-1], weights)
            error = matrix[i][-1] - prediction
            if verbose: sys.stdout.write("Training on data at index %d...\n"%(i))
            for j in range(len(weights)):
                if verbose: sys.stdout.write("\tWeight[%d]: %0.5f --> "%(j,weights[j]))
                weights[j] = weights[j] + (l_rate*error*matrix[i][j])
                if verbose: sys.stdout.write("%0.5f\n"%(weights[j]))

    plot_matrix(matrix,weights,title="Final Epoch")
    return weights

def main():


    nb_epoch = 1000
    l_rate = 1.0
    plot_each_epoch = False
    stop_early = True
    verbose = False
    bias = 1.00
    #           bias    x1   x2    y
    '''data = [    [bias, 0.08, 0.72, 1.0],
                [bias, 0.10, 1.00, 0.0],
                [bias, 0.26, 0.58, 1.0],
                [bias, 0.35, 0.95, 0.0],
                [bias, 0.45, 0.15, 1.0],
                [bias, 0.60, 0.30, 1.0],
                [bias, 0.70, 0.65, 0.0],
                [bias, 0.92, 0.45, 0.0]
            ]'''
    data =  [   [bias, 0.20, 0.10, 0.0],
                [bias, 0.10, 0.50, 0.0],
                [bias, 0.40, 0.60, 0.0],
                [bias, 0.40, 0.90, 0.0],
                [bias, 0.50, 0.10, 1.0],
                [bias, 0.70, 0.30, 1.0],
                [bias, 0.60, 0.70, 1.0],
                [bias, 0.80, 0.80, 1.0]
            ]

    #weights = [0.20, 1, -1]
    weights = [0,0,0]

    train_weights(data,weights=weights,nb_epoch=nb_epoch,l_rate=l_rate,do_plot=plot_each_epoch,stop_early=stop_early, verbose=verbose)
    x_int = (bias *-1)/weights[2]
    y_int = (bias *-1)/weights[1]
    print("x intersect point: 0, ", x_int)
    print("y intersect point: ", y_int, ", 0")


if __name__== '__main__':
    main()
