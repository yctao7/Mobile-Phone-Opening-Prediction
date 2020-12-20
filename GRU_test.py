import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from new_model import GruPredictor

NUM_EPOCHS = 300
BATCH_SIZE = 128
HIDDEN_SIZE = 64
LR = 1e-2

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def dataloader(x_dict):
    x, y = [], []
    for i in range(len(x_dict)):
        dic = x_dict[i]
        y_all = dic['time before bright'].values.tolist()
        dic1 = dic.drop(columns='time before bright')
        x_all = dic1.values.tolist()
        for j in range(len(dic) - 5):
            x.append(x_all[j:(j+5)])
            y.append(y_all[j+4])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    x_train, y_train = torch.tensor(x_train), torch.tensor(y_train)
    x_test, y_test = torch.tensor(x_test), torch.tensor(y_test)

    # normalize x_train
    s1 = x_train.shape
    x_train = x_train.reshape(-1, x_train.shape[-1])
    ma, _ = x_train[:, 5:13].max(dim=0)
    mi, _ = x_train[:, 5:13].min(dim=0)
    x_train[:, 5:13] = (x_train[:, 5:13] - mi) / (ma - mi + 1e-8)
    x_train[:, 13] /= 24 # hour
    x_train[:, 14] /= 60 # minute
    x_train[:, 15] /= 60 # second
    x_train = x_train.reshape(s1)

    # normalize x_test
    s2 = x_test.shape
    x_test = x_test.reshape(-1, x_test.shape[-1])
    x_test[:, 5:13] = (x_test[:, 5:13] - mi) / (ma - mi + 1e-8)
    x_test[:, 13] /= 24 # hour
    x_test[:, 14] /= 60 # minute
    x_test[:, 15] /= 60 # second
    x_test = x_test.reshape(s2)

    # normalize y
    y_train //= 60
    y_test //= 60

    train_set = MyDataset(x_train, y_train)
    test_set = MyDataset(x_test, y_test)
    loader_train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    loader_test = DataLoader(test_set, batch_size=test_set.__len__())
    return loader_train, loader_test, ma, mi

x_dict = np.load('x_dict_new.npy', allow_pickle=True).item()
loader_train, loader_test, ma, mi = dataloader(x_dict)

# import xgboost as xgb
# xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
#                 max_depth = 5, alpha = 10, n_estimators = 10)
# xg_reg.fit(x_train.reshape(-1, x_train.shape[-1]).numpy(), y_train.reshape(-1).numpy())
# preds = xg_reg.predict(x_test.reshape(-1, x_test.shape[-1]).numpy())

input_size = len(x_dict[0].columns) - 1
model = GruPredictor(input_size, HIDDEN_SIZE)
optimizer = Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.StepLR(optimizer, 150, gamma=0.5)
#scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 150], gamma=0.1)
criterion = nn.MSELoss()

for i in range(NUM_EPOCHS):
    for t, (x, y) in enumerate(loader_train):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        print('[%d, %d] loss: %.3f' % (i + 1, t + 1, loss))
    scheduler.step()

    model.eval()
    with torch.no_grad():
        for (x_, y_) in loader_test:
            loss_ = criterion(model(x_), y_)
            print('[%d] loss: %.3f' % (i + 1, loss_))
    model.train()



# test part

#absolute error
label_distribution = {'<10':0,'<20':0,'<50':0,'>50':0}
for t, (x, y) in enumerate(loader_test):
    y_predict = model(x)
    #print(y)
    diff = abs((y_predict - y) / y * 100).tolist()
    cnt = 0
    for value in enumerate(diff):
        if value[1] > 50:
        # The difference exceeds 2 min
            print('[%d, %d] difference: %.3f' % (t, value[0], value[1]))
            print('predicted: %.3f' %(y_predict.tolist()[value[0]]))
            print(y.tolist()[value[0]])
            print(x.tolist()[value[0]])
            print('\n')
            cnt = cnt + 1
            if y.tolist()[value[0]] < 1:
                label_distribution['<10'] = label_distribution['<10'] + 1
            if y.tolist()[value[0]] >= 1 and y.tolist()[value[0]] < 2:
                label_distribution['<20'] = label_distribution['<20'] + 1
            if y.tolist()[value[0]] >= 2 and y.tolist()[value[0]] < 5:
                label_distribution['<50'] = label_distribution['<50'] + 1
            if y.tolist()[value[0]] >= 5:
                label_distribution['>50'] = label_distribution['>50'] + 1

label_distribution

#relative error
label_distribution_absolute = {'<2':0, '<5':0, '<10':0, '<20':0, '>20':0}
for t, (x, y) in enumerate(loader_test):
    y_predict = model(x)
    #print(y)
    diff = abs(y_predict - y).tolist()
    for value in enumerate(diff):

        if value[1] < 2:
            label_distribution_absolute['<2'] = label_distribution_absolute['<2'] + 1
        if value[1] >= 2 and value[1] < 5:
            label_distribution_absolute['<5'] = label_distribution_absolute['<5'] + 1
        if value[1] >= 5 and value[1] < 10:
            label_distribution_absolute['<10'] = label_distribution_absolute['<10'] + 1
        if value[1] >= 10 and value[1] < 20:
            label_distribution_absolute['<20'] = label_distribution_absolute['<20'] + 1
        if value[1] > 20:
        # The difference exceeds 2 min
            label_distribution_absolute['>20'] = label_distribution_absolute['>20'] + 1
            print('[%d, %d] difference: %.3f' % (t, value[0], value[1]))
            print('predicted: %.3f' %(y_predict.tolist()[value[0]]))
            print(y.tolist()[value[0]])
            # see the original x data feature
            shape = x.shape
            x_original = x.reshape(-1, x.shape[-1])
            #x_original = x_original
            x_original[:,0:4] = x_original[:,0:4]
            x_original[:,5:13] = x_original[:, 5:13] * (ma - mi + 1e-8) + mi
            x_original[:,13] = x_original[:,13] * 24
            x_original[:,14] = x_original[:,14] * 60
            x_original[:,15] = x_original[:,15] * 60
            x_original = x_original.reshape(shape)
            print(x_original.tolist()[value[0]])
            print('\n')

label_distribution_absolute

        
