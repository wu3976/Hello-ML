import torch
import numpy as np

# a demo practice of machine learning using linear regression model.
# @author Chenjie Wu

STEP = 1e-5
EPOCH = 10000


# generate the linear modeling of data
def linear_model(data, weight, shift):
    return data @ weight.t() + shift


# perform Mean Square Error op.
def MSE(pred, targ):
    diff = pred - targ
    return torch.sum(diff * diff) / diff.numel()


def linear_train(data, target):
    weight = torch.randn(target.size(1), data.size(1),
                         requires_grad=True)  # generate data according to normal dist: center = 0 and std = 1
    shift = torch.randn(target.size(1), requires_grad=True)

    for i in range(EPOCH):
        prediction = linear_model(data, weight, shift)

        # calculate MSE
        loss = MSE(prediction, target)
        if i % 20 == 0:
            print("loss in " + str(i) + "th epoch: " + str(loss))

        loss.backward()
        with torch.no_grad():
            # make adjustment to model
            weight -= weight.grad * STEP
            shift -= shift.grad * STEP

            # reset grad property
            weight.grad.zero_()
            shift.grad.zero_()

    print("weights: ", end="")
    print(weight)
    print("shifts: ", end="")
    print(shift)
    return weight, shift


data = np.array([[73, 67, 43],
                 [91, 88, 64],
                 [87, 134, 58],
                 [102, 43, 37],
                 [69, 96, 70]], dtype='float32')
data = torch.from_numpy(data)
print(data.dtype)
target = np.array([[56, 70],
                   [81, 101],
                   [119, 133],
                   [22, 37],
                   [103, 119]], dtype='float32')
target = torch.from_numpy(target)

# train a linear model
(weight, shift) = linear_train(data, target)

# make prediction of apple production using trained model
condition = torch.tensor([90, 86, 60], dtype=torch.float32)
print("predicted value: ", end="")
print(str(condition @ weight[0].t() + shift[0]))
