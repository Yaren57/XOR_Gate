# z = (wT)x + b
# z = b + px1w1 + px2w2 + ... + pxnwn

import math as m

w1 = 0.1
w2 = w3 = w4 = w5 = w6 = w1
Learning_rate = 0.2

exp = [0, 1, 1, 0]



def sigmoid(z, derv=False):
    if derv:
        return z * (1 - z)
    return 1 / (1 + np.exp(-z))



def cost_calculator(actual, value):
    error = value * ((1 - value) * (actual - value))
    return error



def new_weight(wi, xi, error):
    new_w = wi + (error * xi)
    return new_w


for epoch in range(20):
    print("")
    print("EPOCH {}".format(epoch + 1))
    iteration = 0

    for a in range(2):
        for b in range(2):
            print("")
            print("for x1={} and x2={}".format(a, b))
            z1 = (w1 * a) + (b * w3)
            z1 = sigmoid(z1)
            print("value of first neuron = {}".format(z1))
            z2 = (b * w4) + (a * w2)
            z2 = sigmoid(z2)
            print("Value of second neuron = {}".format(z2))
            f = (z1 * w5) + (z2 * w6)
            f = sigmoid(f)
            print("Output is {}".format(f))
            error = cost_calculator(exp[iteration], f)
            print("error is {}".format(error))
            w5 = new_weight(w5, z1, error)
            print("Updated First Weight for output is {}".format(w5))
            w6 = new_weight(w6, z2, error)
            print("Updated Second Weight for output is {}".format(w6))
            error1 = (error * w5) * ((1 - f) * f)
            print("Error for Hidden layer = {}".format(error1))
            error2 = (error * w6) * ((1 - f) * f)
            print("Error for hidden layer = {}".format(error2))
            w1 = w1 + (error1 * a)
            w3 = w3 + (error2 * b)
            w2 = w2 + (error1 * a)
            w4 = w4 + (error2 * b)
            print("New Weights for w1 {}".format(w1))
            print("New Weights for w3 {}".format(w3))
            print("New Weights for w2 {}".format(w2))
            print("New Weights for w4 {}".format(w4))