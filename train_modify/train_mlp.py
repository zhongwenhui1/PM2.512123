import os
import sys
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(proj_dir)

from util import config, file_dir
from dataset_modify.dataset_mlp import HazeData
from graph import Graph
from model.MLP import MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import arrow
import numpy as np
from tqdm import tqdm

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
results_dir = file_dir['results_dir']
dataset_num = config['experiments']['dataset_num']
exp_model = config['experiments']['model']
exp_repeat = config['train']['exp_repeat']
save_npy = config['experiments']['save_npy']
criterion = nn.MSELoss()

# Training parameters
train_config = config['train']
batch_size = train_config['batch_size']
epochs = train_config['epochs']
hist_len = train_config['hist_len']
pred_len = train_config['pred_len']
weight_decay = train_config['weight_decay']
early_stop = train_config['early_stop']
lr = train_config['lr']
optimizer_type = train_config['optimizer']
momentum = train_config.get('momentum', 0.9)

# Load data (MLP doesn't need graph structure)
class SimpleGraph:
    def __init__(self):
        self.node_num = 206  # Fixed number of stations

graph = SimpleGraph()
train_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Train')
val_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Val')
test_data = HazeData(graph, hist_len, pred_len, dataset_num, flag='Test')

# Calculate dimensions
in_dim = train_data.feature.shape[-1]
city_num = train_data.graph.node_num
wind_mean, wind_std = train_data.wind_mean, train_data.wind_std
pm25_mean, pm25_std = test_data.pm25_mean, test_data.pm25_std


def get_metric(predict_epoch, label_epoch):
    haze_threshold = 75
    predict_haze = predict_epoch >= haze_threshold
    predict_clear = predict_epoch < haze_threshold
    label_haze = label_epoch >= haze_threshold
    label_clear = label_epoch < haze_threshold
    hit = np.sum(np.logical_and(predict_haze, label_haze))
    miss = np.sum(np.logical_and(label_haze, predict_clear))
    falsealarm = np.sum(np.logical_and(predict_haze, label_clear))
    csi = hit / (hit + falsealarm + miss)
    pod = hit / (hit + miss)
    far = falsealarm / (hit + falsealarm)
    predict = predict_epoch[:,:,:,0].transpose((0,2,1))
    label = label_epoch[:,:,:,0].transpose((0,2,1))
    predict = predict.reshape((-1, predict.shape[-1]))
    label = label.reshape((-1, label.shape[-1]))
    mae = np.mean(np.mean(np.abs(predict - label), axis=1))
    rmse = np.mean(np.sqrt(np.mean(np.square(predict - label), axis=1)))
    return rmse, mae, csi, pod, far


def get_exp_info():
    exp_info =  '============== Train Info ==============\n' + \
                'Dataset number: %s\n' % dataset_num + \
                'Model: %s\n' % exp_model + \
                'Train: %s --> %s\n' % (train_data.start_time, train_data.end_time) + \
                'Val: %s --> %s\n' % (val_data.start_time, val_data.end_time) + \
                'Test: %s --> %s\n' % (test_data.start_time, test_data.end_time) + \
                'City number: %s\n' % city_num + \
                'Use metero: %s\n' % config['experiments']['metero_use'] + \
                'batch_size: %s\n' % batch_size + \
                'epochs: %s\n' % epochs + \
                'hist_len: %s\n' % hist_len + \
                'pred_len: %s\n' % pred_len + \
                'weight_decay: %s\n' % weight_decay + \
                'lr: %s\n' % lr + \
                'optimizer: %s\n' % optimizer_type + \
                'early_stop: %s\n' % early_stop
    return exp_info


def get_model():
    if exp_model == 'MLP':
        return MLP(hist_len, pred_len, in_dim)
    else:
        raise Exception('Only MLP model supported in train_mlp.py!')


def train(train_loader, model, optimizer):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def val(val_loader, model):
    model.eval()
    val_loss = 0
    for batch_idx, data in enumerate(val_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        val_loss += loss.item()

    val_loss /= batch_idx + 1
    return val_loss


def test(test_loader, model):
    model.eval()
    predict_list = []
    label_list = []
    time_list = []
    test_loss = 0
    for batch_idx, data in enumerate(test_loader):
        pm25, feature, time_arr = data
        pm25 = pm25.to(device)
        feature = feature.to(device)
        pm25_label = pm25[:, hist_len:]
        pm25_hist = pm25[:, :hist_len]
        pm25_pred = model(pm25_hist, feature)
        loss = criterion(pm25_pred, pm25_label)
        test_loss += loss.item()

        pm25_pred_val = np.concatenate([pm25_hist.cpu().detach().numpy(), pm25_pred.cpu().detach().numpy()], axis=1) * pm25_std + pm25_mean
        pm25_label_val = pm25.cpu().detach().numpy() * pm25_std + pm25_mean
        predict_list.append(pm25_pred_val)
        label_list.append(pm25_label_val)
        time_list.append(time_arr.cpu().detach().numpy())

    test_loss /= batch_idx + 1

    predict_epoch = np.concatenate(predict_list, axis=0)
    label_epoch = np.concatenate(label_list, axis=0)
    time_epoch = np.concatenate(time_list, axis=0)
    predict_epoch[predict_epoch < 0] = 0

    return test_loss, predict_epoch, label_epoch, time_epoch


def get_mean_std(data_list):
    data = np.asarray(data_list)
    return data.mean(), data.std()


def main():
    exp_info = get_exp_info()
    print(exp_info)

    exp_time = arrow.now().format('YYYYMMDDHHmmss')

    train_loss_list, val_loss_list, test_loss_list, rmse_list, mae_list, csi_list, pod_list, far_list = [], [], [], [], [], [], [], []

    for exp_idx in range(exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

        model = get_model()
        model = model.to(device)
        model_name = type(model).__name__

        print(str(model))

        # Set optimizer based on config
        if optimizer_type == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer_type}')

        exp_model_dir = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), '%02d' % exp_idx)
        if not os.path.exists(exp_model_dir):
            os.makedirs(exp_model_dir)
        model_fp = os.path.join(exp_model_dir, 'model.pth')

        val_loss_min = 100000
        best_epoch = 0

        train_loss_, val_loss_ = 0, 0

        # Early stopping counter
        early_stop_counter = 0

        for epoch in tqdm(range(epochs)):
            train(train_loader, model, optimizer)
            val_loss_ = val(val_loader, model)
            train_loss_, val_loss_ = 0, 0

            if val_loss_ < val_loss_min:
                val_loss_min = val_loss_
                best_epoch = epoch
                early_stop_counter = 0
                torch.save(model.state_dict(), model_fp)
                print('val_loss improved! epoch: %d, val_loss: %.5f' % (epoch, val_loss_))
            else:
                early_stop_counter += 1
                print('val_loss not improved! epoch: %d, val_loss: %.5f, counter: %d' % (epoch, val_loss_, early_stop_counter))

            if early_stop_counter >= early_stop:
                print('Early stopping triggered!')
                break

        print('Best epoch: %d, val_loss_min: %.5f' % (best_epoch, val_loss_min))
        model.load_state_dict(torch.load(model_fp))
        test_loss_, predict_epoch, label_epoch, time_epoch = test(test_loader, model)
        rmse_, mae_, csi_, pod_, far_ = get_metric(predict_epoch, label_epoch)
        print('test_loss: %.5f, rmse: %.5f, mae: %.5f, csi: %.5f, pod: %.5f, far: %.5f' % (test_loss_, rmse_, mae_, csi_, pod_, far_))

        train_loss_list.append(train_loss_)
        val_loss_list.append(val_loss_)
        test_loss_list.append(test_loss_)
        rmse_list.append(rmse_)
        mae_list.append(mae_)
        csi_list.append(csi_)
        pod_list.append(pod_)
        far_list.append(far_)

    train_loss_mean, train_loss_std = get_mean_std(train_loss_list)
    val_loss_mean, val_loss_std = get_mean_std(val_loss_list)
    test_loss_mean, test_loss_std = get_mean_std(test_loss_list)
    rmse_mean, rmse_std = get_mean_std(rmse_list)
    mae_mean, mae_std = get_mean_std(mae_list)
    csi_mean, csi_std = get_mean_std(csi_list)
    pod_mean, pod_std = get_mean_std(pod_list)
    far_mean, far_std = get_mean_std(far_list)

    results_path = os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), 'results.txt')
    with open(results_path, 'w') as f:
        f.write(exp_info)
        f.write('\n')
        f.write('train_loss_mean: %.5f, train_loss_std: %.5f\n' % (train_loss_mean, train_loss_std))
        f.write('val_loss_mean: %.5f, val_loss_std: %.5f\n' % (val_loss_mean, val_loss_std))
        f.write('test_loss_mean: %.5f, test_loss_std: %.5f\n' % (test_loss_mean, test_loss_std))
        f.write('rmse_mean: %.5f, rmse_std: %.5f\n' % (rmse_mean, rmse_std))
        f.write('mae_mean: %.5f, mae_std: %.5f\n' % (mae_mean, mae_std))
        f.write('csi_mean: %.5f, csi_std: %.5f\n' % (csi_mean, csi_std))
        f.write('pod_mean: %.5f, pod_std: %.5f\n' % (pod_mean, pod_std))
        f.write('far_mean: %.5f, far_std: %.5f\n' % (far_mean, far_std))

    if save_npy:
        np.save(os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), 'predict.npy'), predict_epoch)
        np.save(os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), 'label.npy'), label_epoch)
        np.save(os.path.join(results_dir, '%s_%s' % (hist_len, pred_len), str(dataset_num), model_name, str(exp_time), 'time.npy'), time_epoch)


if __name__ == '__main__':
    main()