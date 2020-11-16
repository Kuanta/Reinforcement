from DeepTorch import Trainer as trn
from DeepTorch.Datasets.NumericalDataset import NumericalDataset
import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

class FilterNetwork(nn.Module):
    def __init__(self):
        super(FilterNetwork, self).__init__()
        self.fc1 = nn.Linear(7, 1)

    def forward(self, x):
        return self.fc1(x)

# Load data
dataset = loadmat("dataset.mat")
train = dataset["tr_out"][0][0]
val = dataset["val_out"][0][0]
test = dataset["test_out"][0][0]

tr_reg = np.array([train["y2"], train["y2_1"], train["y2_2"], train["y2_3"], train["y1_1"], train["y1_2"], train["y1_3"]])
tr_reg = np.transpose(tr_reg).squeeze(0)
tr_labels = np.array(train["y1"])

val_reg = np.array([val["y2"], val["y2_1"], val["y2_2"], val["y2_3"], val["y1_1"], val["y1_2"], val["y1_3"]])
val_reg = np.transpose(val_reg).squeeze(0)
val_labels = np.array(val["y1"])

test_reg = np.array([test["y2"], test["y2_1"], test["y2_2"], test["y2_3"], test["y1_1"], test["y1_2"], test["y1_3"]])
test_reg = np.transpose(test_reg).squeeze(0)
test_labels = np.array(test["y1"])

train_dataset = NumericalDataset(tr_reg, tr_labels)
validation_dataset = NumericalDataset(val_reg, val_labels)

# Load network
net = FilterNetwork()

net.load_state_dict(torch.load("filter_network"))
net.to("cuda:0")
trn_opts = trn.TrainingOptions()
trn_opts.batch_size = 1000
trn_opts.learning_rate = 0.01
trn_opts.n_epochs = 400
trn_opts.l2_reg_constant = 0.0001
trn_opts.saved_model_name = "filter_network"
trn_opts.save_model = True
trainer = trn.Trainer(net, trn_opts)
#trainer.train(torch.nn.MSELoss(), train_dataset, validation_set=validation_dataset)
print("Done")

# Regular Test
test_preds = net.forward(torch.tensor(test_reg).to("cuda:0").float()).detach().cpu().numpy()
plt.figure()
plt.plot(np.array(test["y1"]))
plt.legend("Y1")
plt.plot(np.array(test_preds))
plt.legend("Prediction")
plt.title("Regular Test")
plt.show()



# In place Test 1
'''
In this test, y2 inputs are fed into F(s) and the results are compared with y1
'''
test_inputs = np.transpose([test["y2"], test["y2_1"], test["y2_2"], test["y2_3"]]).squeeze(0)
past_preds = [0, 0, 0]
preds = []
for i in range(len(test["tout"])):
    curr_input = np.concatenate((test_inputs[i,:], np.array(past_preds)), axis=0)
    pred = net.forward(torch.tensor(curr_input).to("cuda:0").float()).item()
    preds.append(pred)
    past_preds.insert(0, pred)
    past_preds.pop(-1)

plt.figure()
plt.plot(np.array(test["y1"]))
plt.legend("Y1")
plt.plot(np.array(preds))
plt.legend("Prediction")
plt.show()

# In place Test 2
'''
In this test, a step input is fed into the F(s)*T2(s) system and the results are compared to T1(s)
'''
diff_eq_coeffs = [0,    0.1113,   -0.1775,    0.0671,    2.7563,   -2.5360,    0.7788];
test_inputs = np.transpose([test["u"], test["u_1"], test["u_2"], test["u_3"]]).squeeze(0)
past_preds = [0, 0, 0]
past_y2 = [0, 0, 0]
control_signals = []
preds = []
for i in range(len(test["tout"])):
    # F(s)
    curr_input = np.concatenate((test_inputs[i,:], np.array(past_preds)), axis=0)
    pred = net.forward(torch.tensor(curr_input).to("cuda:0").float()).item()
    control_signals.append(pred)
    past_preds.insert(0, pred)
    past_preds.pop(-1),

    # T2(s)
    curr_r2 = np.array([pred, past_preds[0], past_preds[1], past_preds[2]])
    curr_r2 = np.concatenate((curr_r2, np.array(past_y2)), axis=0)
    _y2 = np.dot(curr_r2, diff_eq_coeffs)
    past_y2.insert(0, _y2)
    past_y2.pop(-1)
    preds.append(_y2)
plt.figure()
plt.plot(np.array(test["y1"]))
plt.legend("Y1")
plt.plot(np.array(preds))
plt.legend("Prediction")
plt.title("In Place Test 2")
plt.show()


