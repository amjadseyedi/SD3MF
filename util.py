import torch
import numpy as np
from scipy.io import loadmat

def topk(A, k):
    vals, idx = torch.topk(A, k, dim=2)
    mask = torch.zeros_like(A, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    return A * mask
    # return mask.to(torch.float32)

def loadDataset(dataset):
    A_list = []
    if dataset=='PPMI':

        PPMI = loadmat('smallDatasets/PPMI.mat')
        y = torch.tensor(PPMI['label_selected'].flatten())

        A1 = torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 0, :], (2, 0, 1))).type(torch.float32)
        A2 = torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 1, :], (2, 0, 1))).type(torch.float32)
        A3 = torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 2, :], (2, 0, 1))).type(torch.float32)

        # k = 10
        # A1 = topk(A1, k)
        # A2 = topk(A2, k)
        # A3 = topk(A3, k)


        A_list.append(A1)
        A_list.append(A2)
        A_list.append(A3)


        # A_list.append(torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 0, :], (2, 0, 1))).type(torch.float32))
        # A_list.append(torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 1, :], (2, 0, 1))).type(torch.float32))
        # A_list.append(torch.tensor(np.transpose(PPMI['X_normalize'][:, :, 2, :], (2, 0, 1))).type(torch.float32))

        # A = torch.from_numpy(np.stack([PPMI['X'][i][0] for i in range(len(PPMI['X']))], axis=-1))
        # #
        # A_list.append(torch.tensor(np.transpose(A[:, :, 0, :], (2, 0, 1))).type(torch.float32))
        # A_list.append(torch.tensor(np.transpose(A[:, :, 1, :], (2, 0, 1))).type(torch.float32))
        # A_list.append(torch.tensor(np.transpose(A[:, :, 2, :], (2, 0, 1))).type(torch.float32))


    elif(dataset=="BP"):

        BP = loadmat('smallDatasets/BP.mat')
        y = torch.tensor(BP['label'].flatten())
        y[y == -1] = 0

        A1 = torch.tensor(np.transpose(BP['dti'], (2, 0, 1))).type(torch.float32)
        A2 = torch.tensor(np.transpose(BP['fmri'], (2, 0, 1))).type(torch.float32)

        # k = 5
        # A1 = topk(A1, k)
        # A2 = topk(A2, k)

        A_list.append(A1)
        A_list.append(A2)

    elif (dataset == "HIV"):

        HIV = loadmat('smallDatasets/HIV.mat')
        y = torch.tensor(HIV['label'].flatten())
        y[y == -1] = 0

        A1 = torch.tensor(np.transpose(HIV['dti'], (2, 0, 1))).type(torch.float32)
        A2 = torch.tensor(np.transpose(HIV['fmri'], (2, 0, 1))).type(torch.float32)

        # k = 10
        # A1 = topk(A1, k)
        # A2 = topk(A2, k)

        A_list.append(A1)
        A_list.append(A2)

    return A_list, y

