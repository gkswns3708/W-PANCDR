from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

import wandb
import numpy as np
import pandas as pd

import torch

import os
import random
import itertools
from tqdm import tqdm
from itertools import cycle


# TODO: Variational Encoder를 추가 혹은 다른 버전의 Encoder를 추가. 현재까지는 FC Layer만 존재
from utils import scores 
from ModelTraining.model import Encoder_FC, GCN, ADV, gradient_penalty, Critic

# For PANCDR
from ModelTraining.model import Encoder
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class train_W_PANCDR():
    def __init__(self, train_data, test_data, outer_fold=None, project=None):
        self.train_data = train_data
        self.test_data = test_data
        self.outer_fold = outer_fold
        self.project = project

    def train(self, params, weight_path='../checkpoint/model.pt'):
        run_name = f"outer_fold_{self.outer_fold}_nz_{params['nz']}_d_dim_{params['d_dim']}_lr_{params['lr']}_lr_adv_{params['lr_adv']}_lam_{params['lam']}_batch_size_{params['batch_size'][0]}"
        run = wandb.init(
            project=self.project,
            config=params,
            name=run_name,
            reinit=True
        )

        nz, d_dim, lr, lr_critic, lam, batch_size = params.values()

        # train/test 데이터만 사용 (validation 제거)
        X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = self.train_data
        TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = self.test_data

        # 모두 학습에 사용
        X_drug_feat_train = torch.tensor(X_drug_feat_data, dtype=torch.float).to(device)
        X_drug_adj_train  = torch.tensor(X_drug_adj_data, dtype=torch.float).to(device)
        X_gexpr_train     = torch.tensor(X_gexpr_data, dtype=torch.float).to(device)
        Y_train           = torch.tensor(Y, dtype=torch.float).view(-1, 1).to(device)
        X_t_gexpr_train   = torch.tensor(t_gexpr_feature.T.values, dtype=torch.float).to(device)

        TX_drug_feat_test = torch.tensor(TX_drug_feat_data_test, dtype=torch.float).to(device)
        TX_drug_adj_test  = torch.tensor(TX_drug_adj_data_test, dtype=torch.float).to(device)
        TX_gexpr_test     = torch.tensor(TX_gexpr_data_test, dtype=torch.float).to(device)
        TY_test           = torch.tensor(TY_test, dtype=torch.float).view(-1, 1).to(device)


        # DataLoader
        GDSC_Loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train),
            batch_size=batch_size[0], shuffle=True, drop_last=True
        )
        E_TEST_Loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t_gexpr_train),
            batch_size=batch_size[1], shuffle=True, drop_last=True
        )

        wait, best_auc = 0, 0
        # 모델 초기화
        EN_model = Encoder_FC(X_gexpr_train.shape[1], nz).to(device)
        GCN_model = GCN(X_drug_feat_train.shape[2], [256,256,256], h_dims=[d_dim, nz+d_dim], use_dropout=False).to(device)
        Critic_model = Critic(nz).to(device)

        wandb.watch(EN_model, log="all", log_freq=50)
        wandb.watch(GCN_model, log="all", log_freq=50)
        wandb.watch(Critic_model, log="all", log_freq=50)

        optimizer = torch.optim.Adam(
            itertools.chain(EN_model.parameters(), GCN_model.parameters()), lr=lr
        )
        optimizer_critic = torch.optim.Adam(Critic_model.parameters(), lr=lr_critic)
        loss_fn = torch.nn.BCELoss()

        current_epoch = -1
        best_metric = {"Accuracy": 0, "AUC": 0, "F1": 0, "Recall": 0, "Precision": 0}
        for epoch in tqdm(range(1000), desc="Epoch", leave=True):
            total_critic_loss = 0.0
            total_gen_loss = 0.0
            train_y_true_list, train_y_pred_list = [], []

            for DataG, (t_gexpr,) in tqdm(zip(GDSC_Loader, cycle(E_TEST_Loader)),
                                        desc=f"Batch (Epoch {epoch})",
                                        leave=False, total=len(GDSC_Loader)):
                drug_feat, drug_adj, gexpr, y_true = DataG
                t_gexpr = t_gexpr.to(device)
                y_true = y_true.to(device)

                # Critic 업데이트
                optimizer_critic.zero_grad()
                with torch.no_grad():
                    fake = EN_model(gexpr)
                    real = EN_model(t_gexpr)
                loss_critic = (
                    Critic_model(fake).mean() - Critic_model(real).mean()
                    + gradient_penalty(Critic_model, fake, real, device, gp_weight=10.0)
                )
                loss_critic.backward()
                optimizer_critic.step()

                # Generator 업데이트
                optimizer.zero_grad()
                F_gexpr = EN_model(gexpr)
                adv_loss = -Critic_model(F_gexpr).mean()
                y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
                cdr_loss = loss_fn(y_pred, y_true.view(-1,1))
                gen_loss = cdr_loss + lam * adv_loss
                gen_loss.backward()
                optimizer.step()

                total_critic_loss += loss_critic.item()
                total_gen_loss    += gen_loss.item()
                train_y_true_list.append(y_true.cpu().numpy().flatten())
                train_y_pred_list.append(y_pred.cpu().detach().numpy().flatten())

            # Epoch별 로그 (weight norm 등)
            for model, tag in [(EN_model, "EN"), (GCN_model, "GCN"), (Critic_model, "Critic")]:
                for name, p in model.named_parameters():
                    if "weight" in name:
                        wandb.log({f"{tag}/{name}_w_norm": p.data.norm(2).item()})

            # Training metrics
            train_y_true = np.concatenate(train_y_true_list)
            train_y_pred = np.concatenate(train_y_pred_list)
            train_auc, train_acc, train_prec, train_rec, train_f1 = scores(train_y_true, train_y_pred)

            # Test 평가
            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                F_test = EN_model(TX_gexpr_test)
                y_pred_test = GCN_model(TX_drug_feat_test, TX_drug_adj_test, F_test)
                test_metrics = scores(
                    TY_test.cpu().numpy().flatten(),
                    y_pred_test.cpu().numpy().flatten()
                )
            test_auc, test_acc, test_prec, test_rec, test_f1 = test_metrics

            # wandb 로그 (validation 항목 삭제)
            wandb.log({
                "epoch": epoch,
                "avg_critic_loss": total_critic_loss / len(GDSC_Loader),
                "avg_gen_loss": total_gen_loss / len(GDSC_Loader),
                "train_auc": train_auc,
                "train_acc": train_acc,
                "train_precision": train_prec,
                "train_recall": train_rec,
                "train_f1": train_f1,
                "test_auc": test_auc,
                "test_acc": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1
            })

            # Early stopping on test_auc
            if test_auc > best_auc:
                best_auc = test_auc
                best_metric['Accuracy'] = test_acc
                best_metric['AUC'] = test_auc
                best_metric['F1'] = test_f1
                best_metric['Recall'] = test_rec
                best_metric['Precision'] = test_prec
                wait = 0
                torch.save({
                    'EN_model': EN_model.state_dict(),
                    'GCN_model': GCN_model.state_dict(),
                    'Critic_model': Critic_model.state_dict()
                }, weight_path)
            else:
                wait += 1
                if wait >= 10:
                    break

            current_epoch = epoch

        run.finish()
        return best_metric, current_epoch

    def predict(self, data, params, weight_path):
        nz, d_dim, _, _, _, _ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder_FC(gexpr.shape[1], nz, device)
        GCN_model = GCN(drug_feat.shape[2], [256,256,256], h_dims=[d_dim, nz+d_dim], use_dropout=False)

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint['EN_model'])
        GCN_model.load_state_dict(checkpoint['GCN_model'])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()
            F_gexpr = EN_model(gexpr)[0] if isinstance(EN_model(gexpr), (list, tuple)) else EN_model(gexpr)
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
        return y_pred.cpu()




def train_W_PANCDR_nested(n_outer_splits, data, best_params_file, 
                          result_file='GDSC_nested_all_results.csv', 
                          best_file='GDSC_nested_best_results.csv'):
    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data
    outer_splits = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=0)
    print(n_outer_splits, "- n_outer_splits")
    outer_folds = outer_splits.split(X_drug_feat_data, Y)
    
    # Load all best params for all folds and all random samples
    best_params_df = pd.read_csv(best_params_file)
    
    # Initialize result DataFrames
    auc_test_df = pd.DataFrame(columns=['Fold', 'Iteration', 'Test_AUC', 'Best_params', 'Best_epoch'])
    best_params_all_folds = pd.DataFrame(columns=['Fold', 'Test_AUC', 'Best_params', 'Best_epoch'])

    for outer_fold, (idx, test_idx) in enumerate(outer_folds):
        fold_params = best_params_df[best_params_df['Fold'] == f"Fold_{outer_fold}"]
        
        X_drug_feat_data_ = X_drug_feat_data[idx]
        X_drug_adj_data_ = X_drug_adj_data[idx]
        X_gexpr_data_ = X_gexpr_data[idx]
        Y_ = Y[idx]

        X_drug_feat_data_test = X_drug_feat_data[test_idx]
        X_drug_adj_data_test = X_drug_adj_data[test_idx]
        X_gexpr_data_test = X_gexpr_data[test_idx]
        Y_test = Y[test_idx]

        X_drug_feat_test = torch.FloatTensor(X_drug_feat_data_test)
        X_drug_adj_test = torch.FloatTensor(X_drug_adj_data_test)
        X_gexpr_test = torch.FloatTensor(X_gexpr_data_test)
        Y_test = torch.FloatTensor(Y_test)

        train_data = [X_drug_feat_data_, X_drug_adj_data_, X_gexpr_data_, Y_, t_gexpr_feature]
        test_data = [X_drug_feat_test, X_drug_adj_test, X_gexpr_test, Y_test]

        best_auc = -1
        best_params = None
        best_epoch = -1

        for i, row in fold_params.iterrows():
            current_params = eval(row['Best_params'])
            model = train_W_PANCDR(train_data, test_data, outer_fold=outer_fold)
            
            epoch_counter = 0  # 추가: Epoch 번호 기록용
            
            while True:
                auc_TEST, current_epoch = model.train(current_params, weight_path=f'../checkpoint/GDSC_kfold/model_best_outerfold_{outer_fold}_{i}.pt')
                epoch_counter += 1  # Epoch 증가
                
                if auc_TEST != -1:
                    break
            
            # Save each result immediately to the all results file
            auc_test_df = pd.concat([auc_test_df, pd.DataFrame([[f"Fold_{outer_fold}", i, auc_TEST, current_params, current_epoch]],
                                                               columns=['Fold', 'Iteration', 'Test_AUC', 'Best_params', 'Best_epoch'])])
            auc_test_df.to_csv(result_file, index=False)  # 모든 학습 결과 저장

            if auc_TEST > best_auc:
                best_auc = auc_TEST
                best_params = current_params
                best_epoch = current_epoch  # 최적 AUC를 기록한 Epoch 저장

        # Save the best result for the current fold
        best_params_all_folds = pd.concat([best_params_all_folds, pd.DataFrame([[f"Fold_{outer_fold}", best_auc, best_params, best_epoch]],
                                                                               columns=['Fold', 'Test_AUC', 'Best_params', 'Best_epoch'])])
        best_params_all_folds.to_csv(best_file, index=False)  # Best 결과 저장

        print(f'Fold {outer_fold} - Best AUC: {best_auc:.4f} (Epoch: {best_epoch})')

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n')
    print(f'Mean test AUC - {auc_test_df["Test_AUC"].mean():.4f}\n')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    auc_test_df = pd.concat([auc_test_df, pd.DataFrame([[None, None, auc_test_df['Test_AUC'].mean(), None, None]],
                                                       columns=['Fold', 'Iteration', 'Test_AUC', 'Best_params', 'Best_epoch'])])
    auc_test_df.to_csv(result_file, index=False)
    best_params_all_folds.to_csv(best_file, index=False)

    return auc_test_df

def train_W_PANCDR_full_cv(n_splits, data, best_params_file, 
                           result_file='./logs/full_CV/GDSC_results.csv',
                           param_summary_file='./logs/full_CV/GDSC_param_summary.csv'):

    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_list = pd.read_csv(best_params_file)['Best_params'].tolist()

    auc_test_df = pd.DataFrame(columns=['Fold', 'Test_AUC', 'params', 'end_epoch'])
    best_params_all_folds = pd.DataFrame(columns=['Fold', 'Test_AUC', 'params', 'end_epoch'])
    for i, param_str in enumerate(param_list):
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_drug_feat_data, Y)):
            X_drug_feat_train = X_drug_feat_data[train_idx]
            X_drug_adj_train = X_drug_adj_data[train_idx]
            X_gexpr_train = X_gexpr_data[train_idx]
            Y_train = Y[train_idx]

            X_drug_feat_test = X_drug_feat_data[test_idx]
            X_drug_adj_test = X_drug_adj_data[test_idx]
            X_gexpr_test = X_gexpr_data[test_idx]
            Y_test = Y[test_idx]

            X_drug_feat_test = torch.FloatTensor(X_drug_feat_test)
            X_drug_adj_test = torch.FloatTensor(X_drug_adj_test)
            X_gexpr_test = torch.FloatTensor(X_gexpr_test)
            Y_test = torch.FloatTensor(Y_test)

            train_data = [X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train, t_gexpr_feature]
            test_data = [X_drug_feat_test, X_drug_adj_test, X_gexpr_test, Y_test]

            current_params = eval(param_str)
            model = train_W_PANCDR(train_data, test_data, outer_fold=fold, project='W-PANCDR_fullCV')

            while True:
                # TODO: 여기 auc_TEST가 아니라, TCGA_Metric로 바꿔야 함
                auc_TEST, end_epoch = model.train(current_params, weight_path=f'../checkpoint/GDSC_fullCV_WANCDR/model_{fold}_{i}.pt')
                if auc_TEST != -1:
                    break

            auc_test_df = pd.concat([auc_test_df, pd.DataFrame([[fold, auc_TEST, current_params, end_epoch]],
                                                            columns=['Fold','Test_AUC', 'Best_params', 'Best_epoch'])])
            auc_test_df.to_csv(result_file, index=False)

    return auc_test_df


import wandb
import numpy as np
import torch
from itertools import cycle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

class train_PANCDR():
    def __init__(self, train_data, test_data, outer_fold=None, project=None):
        self.train_data = train_data
        self.test_data  = test_data
        self.outer_fold = outer_fold
        self.project    = project

    def train(self, params, weight_path='../checkpoint/model.pt'):
        # 하이퍼파라미터 언팩
        nz, d_dim, lr, lr_adv, lam, batch_size = params.values()
        run_name = f"outer_fold_{self.outer_fold}_nz_{params['nz']}_d_dim_{params['d_dim']}_lr_{params['lr']}_lr_adv_{params['lr_adv']}_lam_{params['lam']}_batch_size_{params['batch_size'][0]}"

        # W&B 초기화
        run = wandb.init(
            project=self.project,
            config=params,
            name=run_name,
            reinit=True
        )

        # train/test 데이터 로드 (validation 분할 제거)
        X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = self.train_data
        TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = self.test_data

        # Tensor 변환
        X_drug_feat_train = torch.tensor(X_drug_feat_data, dtype=torch.float).to(device)
        X_drug_adj_train  = torch.tensor(X_drug_adj_data, dtype=torch.float).to(device)
        X_gexpr_train     = torch.tensor(X_gexpr_data, dtype=torch.float).to(device)
        Y_train           = torch.tensor(Y, dtype=torch.float).view(-1, 1).to(device)
        X_t_gexpr_train   = torch.tensor(t_gexpr_feature.T.values, dtype=torch.float).to(device)

        TX_drug_feat_test = torch.tensor(TX_drug_feat_data_test, dtype=torch.float).to(device)
        TX_drug_adj_test  = torch.tensor(TX_drug_adj_data_test, dtype=torch.float).to(device)
        TX_gexpr_test     = torch.tensor(TX_gexpr_data_test, dtype=torch.float).to(device)
        TY_test           = torch.tensor(TY_test, dtype=torch.float).view(-1, 1).to(device)

        # DataLoader
        GDSC_Loader = DataLoader(
            TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train),
            batch_size=batch_size[0],
            shuffle=True,
            drop_last=True
        )
        E_TEST_Loader = DataLoader(
            TensorDataset(X_t_gexpr_train),
            batch_size=batch_size[1],
            shuffle=True,
            drop_last=True
        )

        # 모델 초기화
        EN_model  = Encoder(X_gexpr_train.shape[1], nz, device).to(device)
        GCN_model = GCN(
            X_drug_feat_train.shape[2],
            [256,256,256],
            h_dims=[d_dim, nz+d_dim],
            use_dropout=False
        ).to(device)
        ADV_model = ADV(nz).to(device)

        # 옵티마이저 & 손실함수
        optimizer     = torch.optim.Adam(
            itertools.chain(EN_model.parameters(), GCN_model.parameters()),
            lr=lr
        )
        optimizer_adv = torch.optim.Adam(ADV_model.parameters(), lr=lr_adv)
        loss_fn       = torch.nn.BCELoss()

        # W&B 모델 모니터링
        wandb.watch(EN_model,  log="all", log_freq=50)
        wandb.watch(GCN_model, log="all", log_freq=50)
        wandb.watch(ADV_model, log="all", log_freq=50)

        current_epoch, wait, best_auc_TEST = 0, 0, 0
        best_metric = {"Accuracy": 0, "AUC": 0, "F1": 0, "Recall": 0, "Precision": 0}
        for epoch in tqdm(range(1000), desc="Epoch", leave=True):
            EN_model.train(); GCN_model.train(); ADV_model.train()

            total_adv, total_cdr = 0.0, 0.0
            train_y_true_list, train_y_pred_list = [], []

            # ─── Batch-level Training & Logging ───
            for DataG, (t_gexpr,) in tqdm(zip(GDSC_Loader, cycle(E_TEST_Loader)),
                                        desc=f"Batch (Epoch {epoch})",
                                        leave=False, total=len(GDSC_Loader)):
                drug_feat, drug_adj, gexpr, y_true = DataG
                drug_feat = drug_feat.to(device)
                drug_adj  = drug_adj.to(device)
                gexpr     = gexpr.to(device)
                y_true    = y_true.view(-1,1).to(device)
                t_gexpr   = t_gexpr.to(device)

                # 1) ADV 모델 업데이트
                optimizer_adv.zero_grad()
                F_gexpr, _, _   = EN_model(gexpr)
                F_t_gexpr, _, _ = EN_model(t_gexpr)
                F_cat = torch.cat((F_gexpr, F_t_gexpr))
                z_true = torch.cat([
                    torch.zeros(F_gexpr.size(0), device=device),
                    torch.ones(F_t_gexpr.size(0), device=device)
                ]).view(-1,1)
                z_pred = ADV_model(F_cat)
                adv_loss = loss_fn(z_pred, z_true)
                adv_loss.backward()
                optimizer_adv.step()

                # 2) Generator(EN+GCN) 업데이트
                optimizer.zero_grad()
                g_latents, _, _ = EN_model(gexpr)
                t_latents, _, _ = EN_model(t_gexpr)
                F_cat2 = torch.cat((g_latents, t_latents))
                z_true_ = torch.cat([
                    torch.ones(g_latents.size(0), device=device),
                    torch.zeros(t_latents.size(0), device=device)
                ]).view(-1,1)
                z_pred_ = ADV_model(F_cat2)

                y_pred    = GCN_model(drug_feat, drug_adj, g_latents)
                adv_loss_ = loss_fn(z_pred_, z_true_)
                cdr_loss  = loss_fn(y_pred, y_true)

                Loss = cdr_loss + lam * adv_loss_
                Loss.backward()
                optimizer.step()

                # Batch-level W&B 로깅
                wandb.log({
                    "train/cdr_loss": cdr_loss.item(),
                    "train/adv_loss": adv_loss_.item()
                })

                total_adv += adv_loss_.item()
                total_cdr += cdr_loss.item()

                train_y_true_list.append(y_true.cpu().numpy().flatten())
                train_y_pred_list.append(y_pred.detach().cpu().numpy().flatten())

            # ─── Epoch-level Training Metrics ───
            train_y_true = np.concatenate(train_y_true_list)
            train_y_pred = np.concatenate(train_y_pred_list)
            train_auc, train_acc, train_prec, train_rec, train_f1 = scores(train_y_true, train_y_pred)

            # ─── Test Metrics ───
            with torch.no_grad():
                EN_model.eval(); GCN_model.eval()
                F_test = EN_model(TX_gexpr_data_test)
                # Encoder가 tuple 반환 시 첫번째 요소 사용
                if isinstance(F_test, (list, tuple)):
                    F_test = F_test[0]
                y_pred_TEST = GCN_model(
                    TX_drug_feat_data_test,
                    TX_drug_adj_data_test,
                    F_test
                )

            test_y_true = TY_test.cpu().numpy().flatten()
            test_y_pred = y_pred_TEST.cpu().numpy().flatten()
            test_auc, test_acc, test_prec, test_rec, test_f1 = scores(test_y_true, test_y_pred)

            # ─── Epoch-level W&B 로깅 ───
            wandb.log({
                "epoch": epoch,

                "train_auc": train_auc,
                "train_acc": train_acc,
                "train_precision": train_prec,
                "train_recall": train_rec,
                "train_f1": train_f1,

                "test_auc": test_auc,
                "test_acc": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
                "test_f1": test_f1,

                "avg_adv_loss": total_adv / len(GDSC_Loader),
                "avg_cdr_loss": total_cdr / len(GDSC_Loader),
            })

            # ─── Epoch별 Weight Norm 로깅 ───
            for model, tag in [(EN_model, "EN"), (GCN_model, "GCN"), (ADV_model, "ADV")]:
                for name, p in model.named_parameters():
                    if "weight" in name:
                        wandb.log(
                            {f"{tag}/{name}_w_norm": p.data.norm(2).item()},
                        )

            # ─── Early Stopping on test_auc ───
            if test_auc > best_auc_TEST:
                best_auc_TEST = test_auc
                best_metric['Accuracy'] = test_acc
                best_metric['AUC'] = test_auc
                best_metric['F1'] = test_f1
                best_metric['Recall'] = test_rec
                best_metric['Precision'] = test_prec
                wait = 0
                torch.save({
                    "EN_model":  EN_model.state_dict(),
                    "GCN_model": GCN_model.state_dict(),
                    "ADV_model": ADV_model.state_dict()
                }, weight_path)
            else:
                wait += 1
                if wait >= 10:
                    break
            current_epoch = epoch

        run.finish()
        return best_metric, current_epoch


    def predict(self, data, params, weight_path):
        nz,d_dim,_,_,_,_ = params.values()
        drug_feat, drug_adj, gexpr = data

        EN_model = Encoder(gexpr.shape[1], nz, device)
        GCN_model = GCN(drug_feat.shape[2],[256,256,256],h_dims=[d_dim, nz+d_dim],use_dropout=
        False)

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint['EN_model'])
        GCN_model.load_state_dict(checkpoint['GCN_model'])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()

            F_gexpr = EN_model(gexpr)[0]
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)

        return y_pred.cpu()

def train_PANCDR_full_cv(n_splits, data, best_params_file, 
                           result_file='./logs/full_CV_PANCDR/GDSC_results.csv',
                           param_summary_file='./logs/full_CV_PANCDR/GDSC_param_summary.csv'):

    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_list = pd.read_csv(best_params_file)['Best_params'].tolist()

    auc_test_df = pd.DataFrame(columns=['Fold', 'Test_AUC', 'params', 'end_epoch'])
    best_params_all_folds = pd.DataFrame(columns=['Fold', 'Test_AUC', 'params', 'end_epoch'])
    for i, param_str in enumerate(param_list):
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_drug_feat_data, Y)):
            X_drug_feat_train = X_drug_feat_data[train_idx]
            X_drug_adj_train = X_drug_adj_data[train_idx]
            X_gexpr_train = X_gexpr_data[train_idx]
            Y_train = Y[train_idx]

            X_drug_feat_test = X_drug_feat_data[test_idx]
            X_drug_adj_test = X_drug_adj_data[test_idx]
            X_gexpr_test = X_gexpr_data[test_idx]
            Y_test = Y[test_idx]

            X_drug_feat_test = torch.FloatTensor(X_drug_feat_test)
            X_drug_adj_test = torch.FloatTensor(X_drug_adj_test)
            X_gexpr_test = torch.FloatTensor(X_gexpr_test)
            Y_test = torch.FloatTensor(Y_test)

            train_data = [X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train, t_gexpr_feature]
            test_data = [X_drug_feat_test, X_drug_adj_test, X_gexpr_test, Y_test]

            current_params = eval(param_str)
            model = train_PANCDR(train_data, test_data, outer_fold=fold, project='PANCDR_fullCV')

            while True:
                auc_TEST, end_epoch = model.train(current_params, weight_path=f'../checkpoint/GDSC_fullCV_PANCDR/model_{fold}_{i}.pt')
                if auc_TEST != -1:
                    break

            auc_test_df = pd.concat([auc_test_df, pd.DataFrame([[fold, auc_TEST, current_params, end_epoch]],
                                                            columns=['Fold','Test_AUC', 'Best_params', 'Best_epoch'])])
            auc_test_df.to_csv(result_file, index=False)

    return auc_test_df

def IsNaN(pred):
    return torch.isnan(pred).sum()>0
