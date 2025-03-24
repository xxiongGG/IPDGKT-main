import torch
import tqdm
from torch import nn
from sklearn.metrics import roc_auc_score, root_mean_squared_error
from os.path import join as opj
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from utils import get_acc, get_log, get_normalized_adj, get_e2s_map, get_adj, get_features
from my_dataset import get_dataloader
from my_model import IPDGKT


def train_model(model, train_data, vail_data, epoch, criterion_e, criterion_p,
                optimizer, device, out_path, log_path):
    best_auc, best_acc = 0.0, 0.0
    train_log = []
    print('Beginning train model.')
    for e in range(epoch):
        model.train()
        train_loss, train_loss_p = [], []
        all_pred, all_target = torch.tensor([]).to(device), torch.tensor([]).to(device)
        for [batch_q, batch_y] in tqdm.tqdm(train_data, desc='Epoch {}'.format(e)):
            batch_estate, batch_pred, batch_target = (torch.tensor([]).to(device),
                                                      torch.tensor([]).to(device),
                                                      torch.tensor([]).to(device))
            batch_q, batch_y = batch_q.to(device), batch_y.to(device)
            e_state, preds = model(batch_q, batch_y)

            for student in range(len(batch_y)):
                mask = batch_y[student][1:] != -1
                truth = batch_y[student][1:][mask]
                estate = e_state[student].squeeze(-1)[:-1][mask]
                pred = preds[student].squeeze(-1)[:-1][mask]
                batch_estate = torch.cat([batch_estate, estate])
                batch_pred = torch.cat([batch_pred, pred])
                batch_target = torch.cat([batch_target, truth.float()])
            batch_loss_e = criterion_e(batch_estate, batch_target)
            batch_loss_p = criterion_p(batch_pred, batch_target)
            batch_loss = batch_loss_e + batch_loss_p
            train_loss.append(batch_loss.item())
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            all_pred = torch.cat([all_pred, batch_pred])
            all_target = torch.cat([all_target, batch_target])
        all_target = all_target.to(torch.int32).detach().cpu()

        all_pred = all_pred.detach().cpu()
        auc = roc_auc_score(all_target, all_pred)
        acc = get_acc(all_target, all_pred)
        rmse = root_mean_squared_error(all_target, all_pred)
        mean_loss = round(sum(train_loss) / len(train_data), 6)

        print("【Train】epoch {} ==> loss: {:.6f}, auc: {:.3f}, acc: {:.3f}, rmse: {:.3f}.".format(
            e, mean_loss, auc, acc, rmse))

        train_log.append([e, round(sum(train_loss) / len(train_data), 3), auc, acc, rmse])

        if vail_data is not None:
            vail_auc, vail_acc, vail_rmse = test_model(model, vail_data, device)
            print("【Vail】 auc: {:.3f}, acc: {:.3f}, rmse: {:.3f}.".format(vail_auc, vail_acc, vail_rmse))
            if vail_auc > best_auc:
                best_auc, best_acc = vail_auc, vail_acc
                torch.save(model.state_dict(), out_path)
                print("<<<===MODEL SAVED===>>>")
    get_log(pd.DataFrame(train_log, columns=['Epoch', 'Train Loss', 'Train AUC', 'Train ACC', 'Train RMSE']), log_path)
    return model


def test_model(model, test_data, device):
    model.eval()
    y_pred, y_truth = torch.tensor([]).to(device), torch.tensor([]).to(device)
    with torch.no_grad():
        for index, (batch_q, batch_y) in enumerate(test_data):
            batch_q, batch_y = batch_q.to(device), batch_y.to(device)
            _, preds = model(batch_q, batch_y)

            for student in range(len(batch_y)):
                mask = batch_y[student][1:] != -1
                truth = batch_y[student][1:][mask]
                pred = preds[student].squeeze(-1)[:-1][mask]
                y_pred = torch.cat([y_pred, pred])
                y_truth = torch.cat([y_truth, truth.float().to(device)])

        auc = roc_auc_score(y_truth.detach().cpu().numpy(), y_pred.detach().cpu().numpy())
        acc = get_acc(y_truth.detach().cpu(), y_pred.detach().cpu())
        rmse = root_mean_squared_error(y_truth.detach().cpu(), y_pred.detach().cpu())

    return auc, acc, rmse


if __name__ == '__main__':
    data_name = 'assist09_processed.csv'
    data_path = 'data/assistment_09'
    out_path = 'results/assistment_09'
    problem_id, skill_id = 'problem_id', 'skill_id'

    print('Loading data..., current data is: {}.'.format(data_name))
    data = pd.read_csv(opj(data_path, data_name))
    num_exercise = len(data[problem_id].unique().tolist())
    num_skill = len(data[skill_id].unique().tolist())
    print('Exercise number is: {}, skill number is: {}.'.format(num_exercise, num_skill))

    e2s_map, skill_index = get_e2s_map(data[[problem_id, skill_id]], problem_id, skill_id)
    pro_skill_adj = get_adj(e2s_map, num_exercise, num_skill)
    adj_hat = get_normalized_adj(pro_skill_adj)

    # init model parameters
    feature_index = [0, 1, 2]
    e_hidden_size = 256
    s_hidden_size = 256
    f_hidden_size = 256

    # train model parameters
    LEARNING_RATE = 0.001
    EPOCHS = 30
    BATCH_SIZE = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DEVICE = torch.device('cpu')

    problem_features = get_features(opj(data_path, 'problem_features.txt'), feature_index)
    train_dataset = torch.load(opj(data_path, 'train_dataset.pth'), map_location=DEVICE)
    test_dataset = torch.load(opj(data_path, 'test_dataset.pth'), map_location=DEVICE)
    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    test_loader = get_dataloader(test_dataset, batch_size=1, shuffle=True, drop_last=False)

    print('Train dataset len: {}, test dataset len: {}'.format(len(train_loader), len(test_loader)))
    print('Current device: {}'.format(DEVICE))

    model = IPDGKT(e_hidden_size, s_hidden_size, f_hidden_size, num_exercise, num_skill,
                   problem_features.to(DEVICE), adj_hat, skill_index, DEVICE).to(DEVICE)
    criterion_e = nn.BCELoss()
    criterion_p = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model = train_model(model, train_loader, test_loader, EPOCHS, criterion_e, criterion_p,
                        optimizer, DEVICE, opj(out_path, 'checkpoint.pth'),
                        opj(out_path, 'train_log.xlsx'))

    model.load_state_dict(torch.load(opj(out_path, 'checkpoint.pth')))
    auc, acc, rmse = test_model(model, test_loader, DEVICE)
    get_log(pd.DataFrame([[auc, acc, rmse]], columns=['Test AUC', 'Test ACC', 'Test RMSE']),
            opj(out_path, 'test_log.xlsx'))
    print('【Test】 auc: {:.3f}, acc: {:.3f}, rmse: {:.3f}.'.format(auc, acc, rmse))
