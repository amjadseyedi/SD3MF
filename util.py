import torch
import numpy as np
from scipy.io import loadmat

def loadDataset(dataset):
    A_list = []
    if dataset=='PPMI':

        PPMI = loadmat('smallDatasets/PPMI.mat')
        y = torch.tensor(PPMI['label_selected'].flatten())

        A1 = torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 0, :], (2, 0, 1))).type(torch.float32)
        A2 = torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 1, :], (2, 0, 1))).type(torch.float32)
        A3 = torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 2, :], (2, 0, 1))).type(torch.float32)

        A_list.append(A1)
        A_list.append(A2)
        A_list.append(A3)

    elif(dataset=="BP"):

        BP = loadmat('smallDatasets/BP.mat')
        y = torch.tensor(BP['label'].flatten())
        y[y == -1] = 0

        A1 = torch.tensor(np.transpose(BP['dti'], (2, 0, 1))).type(torch.float32)
        A2 = torch.tensor(np.transpose(BP['fmri'], (2, 0, 1))).type(torch.float32)

        A_list.append(A1)
        A_list.append(A2)

    elif (dataset == "HIV"):

        HIV = loadmat('smallDatasets/HIV.mat')
        y = torch.tensor(HIV['label'].flatten())
        y[y == -1] = 0

        A1 = torch.tensor(np.transpose(HIV['dti'], (2, 0, 1))).type(torch.float32)
        A2 = torch.tensor(np.transpose(HIV['fmri'], (2, 0, 1))).type(torch.float32)

        A_list.append(A1)
        A_list.append(A2)

    return A_list, y


