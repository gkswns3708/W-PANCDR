from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import wandb
import numpy as np
import pandas as pd

import torch

import os
import random
import itertools
from tqdm import tqdm
from copy import deepcopy
from itertools import cycle

# TODO: Variational Encoder를 추가 혹은 다른 버전의 Encoder를 추가. 현재까지는 FC Layer만 존재
from utils import scores, create_one_random_search_params_df
from ModelTraining.model import Encoder_FC, GCN, ADV, gradient_penalty, Critic

# For PANCDR
from ModelTraining.model import Encoder

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class train_WANCDR:
    def __init__(self, train_data, test_data, outer_fold=None, config=None):
        self.train_data = train_data
        self.test_data = test_data
        self.outer_fold = outer_fold
        self.config = config

    def train(self, params, weight_path=f"../checkpoint/model.pt"):
        run_name = f"outer_fold_{self.outer_fold}_nz_{params['nz']}_d_dim_{params['d_dim']}_lr_{params['lr']}_lr_adv_{params['lr_adv']}_lam_{params['lam']}_batch_size_{params['batch_size'][0]}"
        run = wandb.init(
            project=self.config["wandb"]["project_name"],
            config=params,  # ?
            name=run_name,
            reinit=True,
        )

        nz, d_dim, lr, lr_critic, lam, batch_size = params.values()

        # unpack train/test
        X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = (
            self.train_data
        )
        TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = (
            self.test_data
        )

        # split Unlabeled TCGA into train vs holdout for W-distance
        TCGA_vals = t_gexpr_feature.T.values
        X_t_train, X_t_holdout = train_test_split(
            TCGA_vals, test_size=0.05, random_state=0
        )

        # split labeled GDSC + small labeled TCGA for standard validation (optional)
        (
            X_drug_feat_data_train,
            X_drug_feat_data_val,
            X_drug_adj_data_train,
            X_drug_adj_data_val,
            X_gexpr_data_train,
            X_gexpr_data_val,
            Y_train,
            Y_val,
        ) = train_test_split(
            X_drug_feat_data,
            X_drug_adj_data,
            X_gexpr_data,
            Y,
            test_size=0.05,
            random_state=0,
        )

        # Labeled TCGA 데이터
        TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = (
            self.test_data
        )

        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train).to(device)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train).to(device)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train).to(device)
        X_t_train_tensor = torch.FloatTensor(X_t_train).to(device)
        Y_train_tensor = torch.FloatTensor(Y_train).to(device)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)

        X_t_holdout = torch.FloatTensor(X_t_holdout).to(device)

        TX_drug_feat_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
        TX_drug_adj_test = torch.FloatTensor(TX_drug_adj_data_test).to(device)
        TX_gexpr_test = torch.FloatTensor(TX_gexpr_data_test).to(device)
        TY_test = torch.FloatTensor(TY_test).to(device)

        # DataLoaders
        GDSC_Dataset = torch.utils.data.TensorDataset(
            X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train_tensor
        )
        GDSC_Loader = torch.utils.data.DataLoader(
            dataset=GDSC_Dataset, batch_size=batch_size[0], shuffle=True, drop_last=True
        )

        TCGA_Dataset = torch.utils.data.TensorDataset(X_t_train_tensor)
        TCGA_Loader = torch.utils.data.DataLoader(
            dataset=TCGA_Dataset, batch_size=batch_size[1], shuffle=True, drop_last=True
        )

        # model init
        EN_model = Encoder_FC(X_gexpr_train.shape[1], nz).to(device)
        GCN_model = GCN(
            X_drug_feat_train.shape[2],
            [256, 256, 256],
            h_dims=[d_dim, nz + d_dim],
            use_dropout=False,
        ).to(device)
        Critic_model = Critic(nz).to(device)

        wandb.watch(EN_model, log="all", log_freq=50)
        wandb.watch(GCN_model, log="all", log_freq=50)
        wandb.watch(Critic_model, log="all", log_freq=50)

        optimizer = torch.optim.Adam(
            itertools.chain(EN_model.parameters(), GCN_model.parameters()), lr=lr
        )
        optimizer_critic = torch.optim.Adam(Critic_model.parameters(), lr=lr_critic)
        loss_fn = torch.nn.BCELoss()

        # Initialize log_metric with default values before training loop
        log_metric = {
            "Epoch": 0,
            "Train AUC": 0.0,
            "Train Accuracy": 0.0,
            "Train F1": 0.0,
            "Train Recall": 0.0,
            "Train Precision": 0.0,
            "Train W_distance": float('inf'),
            "Train critic_loss": float('inf'),
            "Train gen_loss": float('inf'),
            "Train total_loss": float('inf'),
            "Val AUC": 0.0,
            "Val Accuracy": 0.0,
            "Val F1": 0.0,
            "Val Recall": 0.0,
            "Val Precision": 0.0,
            "Val W_distance": float('inf'),
            "Val critic_loss": 0.0,
            "Val gen_loss": 0.0,
            "Val total_loss": 0.0,
            "Test AUC": 0.0,
            "Test Accuracy": 0.0,
            "Test F1": 0.0,
            "Test Recall": 0.0,
            "Test Precision": 0.0,
            "Test W_distance": float('inf'),
            "Test critic_loss": 0.0,
            "Test gen_loss": 0.0,
            "Test total_loss": 0.0,
        }
        best_metric = deepcopy(log_metric)
        wait = 0
        current_epoch = -1

        # training loop
        for epoch in tqdm(
            range(self.config["train"]["max_epochs"]), desc="Epoch", leave=True
        ):
            EN_model.train()
            GCN_model.train()
            Critic_model.train()
            train_total_critic_loss = 0.0
            train_total_gen_loss = 0.0
            train_total_loss = 0.0
            num_batches = 0
            # per-batch adversarial + CDR training
            train_y_true_list = []
            train_y_pred_list = []
            train_w_distance_list = []
            for (drug_feat, drug_adj, gexpr, y_true), (t_gexpr,) in tqdm(
                zip(GDSC_Loader, cycle(TCGA_Loader)),
                desc=f"Batch (Epoch {epoch})",
                leave=False,
                total=len(GDSC_Loader),
            ):

                # critic update
                optimizer_critic.zero_grad()
                with torch.no_grad():
                    F_fake = EN_model(gexpr)
                    F_real = EN_model(t_gexpr.to(device))
                D_real = Critic_model(F_real)
                D_fake = Critic_model(F_fake)
                loss_critic = (D_fake.mean() - D_real.mean()) + gradient_penalty(
                    Critic_model, F_fake, F_real, device, gp_weight=10.0
                )
                loss_critic.backward()
                optimizer_critic.step()

                # generator + CDR update
                optimizer.zero_grad()
                F_fake = EN_model(gexpr)
                adv_loss = -Critic_model(F_fake).mean()
                y_pred = GCN_model(drug_feat, drug_adj, F_fake)
                cdr_loss = loss_fn(y_pred, y_true.view(-1, 1).to(device))
                total_loss = cdr_loss + lam * adv_loss
                total_loss.backward()
                optimizer.step()

                train_total_critic_loss += loss_critic.item()
                train_total_gen_loss += total_loss.item()
                train_total_loss += total_loss.item() + lam * loss_critic.item()
                num_batches += 1

                train_y_true_list.append(y_true.cpu().detach().numpy().flatten())
                train_y_pred_list.append(y_pred.cpu().detach().numpy().flatten())
                train_w_distance_list.append((D_real.mean() - D_fake.mean()).item())

            train_critic_loss = train_total_critic_loss / num_batches
            train_gen_loss = train_total_gen_loss / num_batches
            train_total_loss = train_total_loss / num_batches

            train_y_true = np.concatenate(train_y_true_list)
            train_y_pred = np.concatenate(train_y_pred_list)
            train_auc, train_acc, train_precision, train_recall, train_f1 = scores(
                train_y_true, train_y_pred
            )
            train_w_distance = np.mean(train_w_distance_list)

            # TODO: validation/Test metric 추가
            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                # Validation 평가
                GDSC_val_latent_vector = EN_model(X_gexpr_val)
                GDSC_val = Critic_model(GDSC_val_latent_vector)
                uTCGA_val_latent_vector = EN_model(X_t_holdout)
                uTCGA_val = Critic_model(uTCGA_val_latent_vector)
                val_w_distance = (uTCGA_val.mean() - GDSC_val.mean()).item()
                val_loss_critic = GDSC_val.mean() - uTCGA_val.mean()

                y_pred_val = GCN_model(
                    X_drug_feat_val, X_drug_adj_val, GDSC_val_latent_vector
                )
                val_gen_loss = loss_fn(y_pred_val, Y_val.view(-1, 1).to(device))
                val_total_loss = val_gen_loss + lam * val_loss_critic
                y_true_val = Y_val.cpu().detach().numpy().flatten()
                y_pred_val_np = y_pred_val.cpu().detach().numpy().flatten()
                val_auc, val_acc, val_precision, val_recall, val_f1 = scores(
                    y_true_val, y_pred_val_np
                )

                # Test 평가
                F_test = EN_model(TX_gexpr_test)
                y_pred_test = GCN_model(TX_drug_feat_test, TX_drug_adj_test, F_test)

                test_w_distance = (Critic_model(F_test).mean() - GDSC_val.mean()).item()
                test_loss_critic = GDSC_val.mean() - Critic_model(F_test).mean()
                test_gen_loss = loss_fn(y_pred_test, TY_test.view(-1, 1).to(device))
                test_total_loss = test_gen_loss + lam * test_loss_critic
                y_true_test = TY_test.cpu().detach().numpy().flatten()
                y_pred_test_np = y_pred_test.cpu().detach().numpy().flatten()
                test_auc, test_acc, test_precision, test_recall, test_f1 = scores(
                    y_true_test, y_pred_test_np
                )

                # wandb에 epoch 단위로 로깅 (평균 training loss 및 training metric 포함)

            log_metric.update(
                {
                    "Epoch": epoch,
                    "Train AUC": train_auc,
                    "Train Accuracy": train_acc,
                    "Train F1": train_f1,
                    "Train Recall": train_recall,
                    "Train Precision": train_precision,
                    "Train W_distance": train_w_distance,
                    "Train critic_loss": train_critic_loss,
                    "Train gen_loss": train_gen_loss,
                    "Train total_loss": train_total_loss,
                    "Val AUC": val_auc,
                    "Val Accuracy": val_acc,
                    "Val F1": val_f1,
                    "Val Recall": val_recall,
                    "Val Precision": val_precision,
                    "Val W_distance": val_w_distance,
                    "Val critic_loss": val_loss_critic.item(),
                    "Val gen_loss": val_gen_loss.item(),
                    "Val total_loss": val_total_loss.item(),
                    "Test AUC": test_auc,
                    "Test Accuracy": test_acc,
                    "Test F1": test_f1,
                    "Test Recall": test_recall,
                    "Test Precision": test_precision,
                    "Test W_distance": test_w_distance,
                    "Test critic_loss": test_loss_critic.item(),
                    "Test gen_loss": test_gen_loss.item(),
                    "Test total_loss": test_total_loss.item(),
                }
            )

            wandb.log({**log_metric})

            save = False
            if self.config["test_metric"] == "W_distance":
                if log_metric["Val W_distance"] < best_metric["Val W_distance"]:
                    save = True
            elif self.config["test_metric"] == "Loss":
                if log_metric["Val total_loss"] < best_metric["Val total_loss"]:
                    save = True
            elif self.config["test_metric"] == "AUC":
                if log_metric["Val AUC"] > best_metric["Val AUC"]:
                    save = True

            if save:
                wait = 0
                # save checkpoint
                torch.save(
                    {
                        "EN_model": EN_model.state_dict(),
                        "GCN_model": GCN_model.state_dict(),
                        "Critic_model": Critic_model.state_dict(),
                    },
                    weight_path,
                )
                current_epoch = epoch
                best_metric.update({**log_metric})
            else:
                wait += 1

            current_csv_path = self.config["csv"]["current_result_path"]
            df_current = pd.DataFrame(
                {
                    "Iteration": self.outer_fold,
                    **log_metric,
                },
                index=[0],
            )

            if os.path.exists(self.config["csv"]["current_result_path"]):
                df_history = pd.read_csv(current_csv_path)
                df_all = pd.concat([df_history, df_current], ignore_index=True)
            else:
                df_all = df_current

            # 3) 다시 덮어쓰기
            df_all.to_csv(current_csv_path, index=False)
            print(f"Saved metrics for iteration {iter} to {current_csv_path}")

            if wait >= 10:
                print(
                    f"Early stopping at epoch {epoch} due to W-distance not improving."
                )
                break

        run.finish()
        return best_metric, current_epoch

    def _train(self, params, weight_path="../checkpoint/model.pt"):
        run_name = f"outer_fold_{self.outer_fold}_nz_{params['nz']}_d_dim_{params['d_dim']}_lr_{params['lr']}_lr_adv_{params['lr_adv']}_lam_{params['lam']}_batch_size_{params['batch_size'][0]}"
        run = wandb.init(
            project=self.config["wandb"]["project_name"],
            config=params,  # ?
            name=run_name,
            reinit=True,
        )

        nz, d_dim, lr, lr_critic, lam, batch_size = params.values()

        # train/test 데이터만 사용 (validation 제거)
        X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = (
            self.train_data
        )
        TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = (
            self.test_data
        )
        X_t_train, X_t_val = train_test_split(
            t_gexpr_feature.T.values, test_size=0.05, random_state=0
        )
        (
            X_drug_feat_data_train,
            X_drug_feat_data_val,
            X_drug_adj_data_train,
            X_drug_adj_data_val,
            X_gexpr_data_train,
            X_gexpr_data_val,
            Y_train,
            Y_val,
        ) = train_test_split(
            X_drug_feat_data,
            X_drug_adj_data,
            X_gexpr_data,
            Y,
            test_size=0.05,
            random_state=0,
        )

        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train).to(device)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train).to(device)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train).to(device)
        X_t_gexpr_train = torch.FloatTensor(X_t_train).to(device)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)

        TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
        TX_drug_adj_data_test = torch.FloatTensor(TX_drug_adj_data_test).to(device)
        TX_gexpr_data_test = torch.FloatTensor(TX_gexpr_data_test).to(device)
        TY_test = torch.FloatTensor(TY_test).to(device)

        # DataLoader 생성
        GDSC_Dataset = torch.utils.data.TensorDataset(
            X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train
        )
        GDSC_Loader = torch.utils.data.DataLoader(
            dataset=GDSC_Dataset, batch_size=batch_size[0], shuffle=True, drop_last=True
        )
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(
            dataset=E_TEST_Dataset,
            batch_size=batch_size[1],
            shuffle=True,
            drop_last=True,
        )

        wait, best_auc = 0, 0
        # 모델 생성
        EN_model = Encoder_FC(X_gexpr_train.shape[1], nz)
        GCN_model = GCN(
            X_drug_feat_train.shape[2],
            [256, 256, 256],
            h_dims=[d_dim, nz + d_dim],
            use_dropout=False,
        )
        Critic_model = Critic(nz)
        EN_model.to(device)
        GCN_model.to(device)
        Critic_model.to(device)
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
            total_critic_loss, total_cdr_loss, total_total_loss = 0.0, 0.0, 0.0
            train_y_true_list, train_y_pred_list = [], []

            for DataG, (t_gexpr,) in tqdm(
                zip(GDSC_Loader, cycle(E_TEST_Loader)),
                desc=f"Batch (Epoch {epoch})",
                leave=False,
                total=len(GDSC_Loader),
            ):
                drug_feat, drug_adj, gexpr, y_true = DataG
                t_gexpr = t_gexpr.to(device)
                y_true = y_true.to(device)

                EN_model.train()
                GCN_model.train()
                Critic_model.train()
                # Critic 업데이트
                optimizer_critic.zero_grad()
                with torch.no_grad():
                    F_gexpr = EN_model(gexpr)  # fake (GDSC)
                    F_t_gexpr = EN_model(t_gexpr)  # real (TCGA)
                real_validity = Critic_model(F_t_gexpr)  # D(real)
                fake_validity = Critic_model(F_gexpr)  # D(fake)
                loss_critic = (
                    fake_validity.mean() - real_validity.mean()
                ) + gradient_penalty(
                    Critic_model, F_gexpr, F_t_gexpr, device, gp_weight=10.0
                )
                loss_critic.backward()
                optimizer_critic.step()

                # Generator 업데이트
                optimizer.zero_grad()
                F_gexpr = EN_model(gexpr)
                adv_loss = -Critic_model(F_gexpr).mean()
                y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
                cdr_loss = loss_fn(y_pred, y_true.view(-1, 1))
                total_loss = cdr_loss + lam * adv_loss
                total_loss.backward()
                optimizer.step()

                total_critic_loss += loss_critic.item()
                total_cdr_loss += cdr_loss.item()
                total_total_loss += total_loss.item()
                train_y_true_list.append(y_true.cpu().numpy().flatten())
                train_y_pred_list.append(y_pred.cpu().detach().numpy().flatten())

            # Epoch별 로그 (weight norm 등)
            for model, tag in [
                (EN_model, "EN"),
                (GCN_model, "GCN"),
                (Critic_model, "Critic"),
            ]:
                for name, p in model.named_parameters():
                    if "weight" in name:
                        wandb.log({f"{tag}/{name}_w_norm": p.data.norm(2).item()})

            # Training metrics
            train_y_true = np.concatenate(train_y_true_list)
            train_y_pred = np.concatenate(train_y_pred_list)
            train_auc, train_acc, train_prec, train_rec, train_f1 = scores(
                train_y_true, train_y_pred
            )

            # Test 평가
            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                # Validation 평가
                F_val = EN_model(X_gexpr_val)
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_val)
                y_true_val = Y_val.cpu().detach().numpy().flatten()
                y_pred_val_np = y_pred_val.cpu().detach().numpy().flatten()
                val_auc, val_acc, val_precision, val_recall, val_f1 = scores(
                    y_true_val, y_pred_val_np
                )

                # Test 평가
                F_test = EN_model(TX_gexpr_data_test)
                y_pred_test = GCN_model(
                    TX_drug_feat_data_test, TX_drug_adj_data_test, F_test
                )
                y_true_test = TY_test.cpu().detach().numpy().flatten()
                y_pred_test_np = y_pred_test.cpu().detach().numpy().flatten()
                test_auc, test_acc, test_precision, test_recall, test_f1 = scores(
                    y_true_test, y_pred_test_np
                )

            wandb.log(
                {
                    "epoch": epoch,
                    "avg_critic_loss": total_critic_loss / len(GDSC_Loader),
                    "avg_cdr_loss": total_cdr_loss / len(GDSC_Loader),
                    "avg_total_loss": total_total_loss / len(GDSC_Loader),
                    "train_auc": train_auc,
                    "train_acc": train_acc,
                    "train_precision": train_prec,
                    "train_recall": train_rec,
                    "train_f1": train_f1,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "test_auc": test_auc,
                    "test_acc": test_acc,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                }
            )

            # Early stopping on test_auc
            if val_auc > best_auc:
                best_auc = val_auc
                best_metric["Accuracy"] = test_acc
                best_metric["AUC"] = test_auc
                best_metric["F1"] = test_f1
                best_metric["Recall"] = test_recall
                best_metric["Precision"] = test_precision
                wait = 0
                torch.save(
                    {
                        "EN_model": EN_model.state_dict(),
                        "GCN_model": GCN_model.state_dict(),
                        "Critic_model": Critic_model.state_dict(),
                    },
                    weight_path,
                )
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
        GCN_model = GCN(
            drug_feat.shape[2],
            [256, 256, 256],
            h_dims=[d_dim, nz + d_dim],
            use_dropout=False,
        )

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint["EN_model"])
        GCN_model.load_state_dict(checkpoint["GCN_model"])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()
            F_gexpr = (
                EN_model(gexpr)[0]
                if isinstance(EN_model(gexpr), (list, tuple))
                else EN_model(gexpr)
            )
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
        return y_pred.cpu()


def train_W_PANCDR_nested(
    n_outer_splits,
    data,
    best_params_file,
    result_file="GDSC_nested_all_results.csv",
    best_file="GDSC_nested_best_results.csv",
):
    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data
    outer_splits = StratifiedKFold(
        n_splits=n_outer_splits, shuffle=True, random_state=0
    )
    print(n_outer_splits, "- n_outer_splits")
    outer_folds = outer_splits.split(X_drug_feat_data, Y)

    # Load all best params for all folds and all random samples
    best_params_df = pd.read_csv(best_params_file)

    # Initialize result DataFrames
    auc_test_df = pd.DataFrame(
        columns=["Fold", "Iteration", "Test_AUC", "Best_params", "Best_epoch"]
    )
    best_params_all_folds = pd.DataFrame(
        columns=["Fold", "Test_AUC", "Best_params", "Best_epoch"]
    )

    for outer_fold, (idx, test_idx) in enumerate(outer_folds):
        fold_params = best_params_df[best_params_df["Fold"] == f"Fold_{outer_fold}"]

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

        train_data = [
            X_drug_feat_data_,
            X_drug_adj_data_,
            X_gexpr_data_,
            Y_,
            t_gexpr_feature,
        ]
        test_data = [X_drug_feat_test, X_drug_adj_test, X_gexpr_test, Y_test]

        best_auc = -1
        best_params = None
        best_epoch = -1

        for i, row in fold_params.iterrows():
            current_params = eval(row["Best_params"])
            model = train_W_PANCDR(train_data, test_data, outer_fold=outer_fold)

            epoch_counter = 0  # 추가: Epoch 번호 기록용

            while True:
                auc_TEST, current_epoch = model.train(
                    current_params,
                    weight_path=f"../checkpoint/GDSC_kfold/model_best_outerfold_{outer_fold}_{i}.pt",
                )
                epoch_counter += 1  # Epoch 증가

                if auc_TEST != -1:
                    break

            # Save each result immediately to the all results file
            auc_test_df = pd.concat(
                [
                    auc_test_df,
                    pd.DataFrame(
                        [
                            [
                                f"Fold_{outer_fold}",
                                i,
                                auc_TEST,
                                current_params,
                                current_epoch,
                            ]
                        ],
                        columns=[
                            "Fold",
                            "Iteration",
                            "Test_AUC",
                            "Best_params",
                            "Best_epoch",
                        ],
                    ),
                ]
            )
            auc_test_df.to_csv(result_file, index=False)  # 모든 학습 결과 저장

            if auc_TEST > best_auc:
                best_auc = auc_TEST
                best_params = current_params
                best_epoch = current_epoch  # 최적 AUC를 기록한 Epoch 저장

        # Save the best result for the current fold
        best_params_all_folds = pd.concat(
            [
                best_params_all_folds,
                pd.DataFrame(
                    [[f"Fold_{outer_fold}", best_auc, best_params, best_epoch]],
                    columns=["Fold", "Test_AUC", "Best_params", "Best_epoch"],
                ),
            ]
        )
        best_params_all_folds.to_csv(best_file, index=False)  # Best 결과 저장

        print(f"Fold {outer_fold} - Best AUC: {best_auc:.4f} (Epoch: {best_epoch})")

    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print(f'Mean test AUC - {auc_test_df["Test_AUC"].mean():.4f}\n')
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    auc_test_df = pd.concat(
        [
            auc_test_df,
            pd.DataFrame(
                [[None, None, auc_test_df["Test_AUC"].mean(), None, None]],
                columns=["Fold", "Iteration", "Test_AUC", "Best_params", "Best_epoch"],
            ),
        ]
    )
    auc_test_df.to_csv(result_file, index=False)
    best_params_all_folds.to_csv(best_file, index=False)

    return auc_test_df


def train_WANCDR_full_cv(
    train_data,
    result_file="./logs/full_CV/GDSC_results.csv",
    param_summary_file="./logs/full_CV/GDSC_param_summary.csv",
    config=None,
):

    # 랜덤 파라미터 생성 및 저장
    # param_list = pd.read_csv(config['csv']['hp_list_path'])['Best_params'].tolist()

    param_list = create_one_random_search_params_df(config=config)[
        "Hyperparameters"
    ].tolist()
    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = train_data
    cv = StratifiedKFold(
        n_splits=config["hp"]["n_outer_splits"], shuffle=True, random_state=42
    )
    auc_test_df = pd.DataFrame(
        columns=["Fold", f'metric_{config["test_metric"]}', "params", "end_epoch"]
    )
    for i, param_dict in enumerate(param_list):
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_drug_feat_data, Y)):
            X_drug_feat_train = X_drug_feat_data[train_idx]
            X_drug_adj_train = X_drug_adj_data[train_idx]
            X_gexpr_train = X_gexpr_data[train_idx]
            Y_train = Y[train_idx]

            X_drug_feat_val = torch.FloatTensor(X_drug_feat_data[val_idx])
            X_drug_adj_val = torch.FloatTensor(X_drug_adj_data[val_idx])
            X_gexpr_val = torch.FloatTensor(X_gexpr_data[val_idx])
            Y_val = torch.FloatTensor(Y[val_idx])

            # X_drug_feat_val = torch.FloatTensor(X_drug_feat_val)
            # X_drug_adj_val = torch.FloatTensor(X_drug_adj_val)
            # X_gexpr_val = torch.FloatTensor(X_gexpr_val)
            # Y_val = torch.FloatTensor(Y_val)

            train_data = [
                X_drug_feat_train,
                X_drug_adj_train,
                X_gexpr_train,
                Y_train,
                t_gexpr_feature,
            ]
            val_data = [X_drug_feat_val, X_drug_adj_val, X_gexpr_val, Y_val]
            model = train_WANCDR(train_data, val_data, outer_fold=fold, config=config)

            while True:
                # TODO: 여기 auc_TEST가 아니라, TCGA_Metric로 바꿔야 함
                auc_TEST, end_epoch = model.train(
                    config,
                    param_dict,
                    weight_path=os.path.join(
                        config["train"]["weight_path"], f"model_{fold}_{i}.pt"
                    ),
                )
                if auc_TEST != -1:
                    break

            param_df = pd.DataFrame([param_dict])
            row_df = pd.DataFrame(
                [[fold, auc_TEST, end_epoch]],
                columns=["Fold", f'metric_{config["test_metric"]}', "Best_epoch"],
            )
            combined_df = pd.concat([row_df, param_df], axis=1)
            auc_test_df = pd.concat([auc_test_df, combined_df], ignore_index=True)
            auc_test_df.to_csv(config["csv"]["result_file_path"], index=False)

    return auc_test_df


import wandb
import numpy as np
import torch
from itertools import cycle
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import TensorDataset, DataLoader


class train_PANCDR:
    def __init__(self, train_data, test_data, outer_fold=None, project=None):
        self.train_data = train_data
        self.test_data = test_data
        self.outer_fold = outer_fold
        self.project = project

    def train(self, params, weight_path="../checkpoint/model.pt"):
        nz, d_dim, lr, lr_adv, lam, batch_size = params.values()
        run_name = f"outer_fold_{self.outer_fold}_nz_{params['nz']}_d_dim_{params['d_dim']}_lr_{params['lr']}_lr_adv_{params['lr_adv']}_lam_{params['lam']}_batch_size_{params['batch_size'][0]}"

        # W&B 초기화
        run = wandb.init(
            project=self.project, config=params, name=run_name, reinit=True
        )

        X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = (
            self.train_data
        )
        TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = (
            self.test_data
        )
        X_t_train, X_t_val = train_test_split(
            t_gexpr_feature.T.values, test_size=0.05, random_state=0
        )
        (
            X_drug_feat_data_train,
            X_drug_feat_data_val,
            X_drug_adj_data_train,
            X_drug_adj_data_val,
            X_gexpr_data_train,
            X_gexpr_data_val,
            Y_train,
            Y_val,
        ) = train_test_split(
            X_drug_feat_data,
            X_drug_adj_data,
            X_gexpr_data,
            Y,
            test_size=0.05,
            random_state=0,
        )

        X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train)
        X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train)
        X_gexpr_train = torch.FloatTensor(X_gexpr_data_train)
        X_t_gexpr_train = torch.FloatTensor(X_t_train)
        Y_train = torch.FloatTensor(Y_train)

        X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
        X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
        X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
        X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
        Y_val = torch.FloatTensor(Y_val).to(device)

        TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
        TX_drug_adj_data_test = torch.FloatTensor(TX_drug_adj_data_test).to(device)
        TX_gexpr_data_test = torch.FloatTensor(TX_gexpr_data_test).to(device)
        TY_test = torch.FloatTensor(TY_test).to(device)

        GDSC_Dataset = torch.utils.data.TensorDataset(
            X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train
        )
        GDSC_Loader = torch.utils.data.DataLoader(
            dataset=GDSC_Dataset, batch_size=batch_size[0], shuffle=True, drop_last=True
        )
        E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
        E_TEST_Loader = torch.utils.data.DataLoader(
            dataset=E_TEST_Dataset,
            batch_size=batch_size[1],
            shuffle=True,
            drop_last=True,
        )

        wait, best_auc = 0, 0
        EN_model = Encoder(X_gexpr_train.shape[1], nz, device)
        GCN_model = GCN(
            X_drug_feat_train.shape[2],
            [256, 256, 256],
            h_dims=[d_dim, nz + d_dim],
            use_dropout=False,
        )
        ADV_model = ADV(nz)
        EN_model.to(device)
        GCN_model.to(device)
        ADV_model.to(device)

        optimizer = torch.optim.Adam(
            itertools.chain(EN_model.parameters(), GCN_model.parameters()), lr=lr
        )
        optimizer_adv = torch.optim.Adam(ADV_model.parameters(), lr=lr_adv)
        loss = torch.nn.BCELoss()

        # W&B 모델 모니터링
        wandb.watch(EN_model, log="all", log_freq=50)
        wandb.watch(GCN_model, log="all", log_freq=50)
        wandb.watch(ADV_model, log="all", log_freq=50)

        current_epoch, wait, best_auc_TEST = 0, 0, 0
        best_metric = {"Accuracy": 0, "AUC": 0, "F1": 0, "Recall": 0, "Precision": 0}
        for epoch in tqdm(range(1000), desc="Epoch", leave=True):
            EN_model.train()
            GCN_model.train()
            ADV_model.train()

            total_adv, total_cdr, total_total_loss = 0.0, 0.0, 0.0
            train_y_true_list, train_y_pred_list = [], []

            # ─── Batch-level Training & Logging ───
            for DataG, (t_gexpr,) in tqdm(
                zip(GDSC_Loader, cycle(E_TEST_Loader)),
                desc=f"Batch (Epoch {epoch})",
                leave=False,
                total=len(GDSC_Loader),
            ):

                drug_feat, drug_adj, gexpr, y_true = DataG
                drug_feat = drug_feat.to(device)
                drug_adj = drug_adj.to(device)
                gexpr = gexpr.to(device)
                y_true = y_true.view(-1, 1).to(device)
                t_gexpr = t_gexpr.to(device)

                # 1) ADV 모델 업데이트
                optimizer_adv.zero_grad()
                F_gexpr, _, _ = EN_model(gexpr)
                F_t_gexpr, _, _ = EN_model(t_gexpr)
                F_g_t_gexpr = torch.cat((F_gexpr, F_t_gexpr))
                z_true = torch.cat(
                    (
                        torch.zeros(F_gexpr.shape[0], device=device),
                        torch.ones(F_t_gexpr.shape[0], device=device),
                    )
                )
                z_true = z_true.view(-1, 1)
                z_pred = ADV_model(F_g_t_gexpr)
                if IsNaN(z_pred):
                    print("IsNAN(z_pred)")
                    return -1
                adv_loss = loss(z_pred, z_true)
                adv_loss.backward()
                optimizer_adv.step()

                # Encoder + GCN 모델 업데이트
                optimizer.zero_grad()
                g_latents, _, _ = EN_model(gexpr)
                t_latents, _, _ = EN_model(t_gexpr)
                F_g_t_latents = torch.cat((g_latents, t_latents))
                z_true_ = torch.cat(
                    (
                        torch.ones(g_latents.shape[0], device=device),
                        torch.zeros(t_latents.shape[0], device=device),
                    )
                )
                z_true_ = z_true_.view(-1, 1)
                z_pred_ = ADV_model(F_g_t_latents)
                y_pred = GCN_model(drug_feat, drug_adj, g_latents)
                if IsNaN(z_pred_) or IsNaN(y_pred):
                    print("IsNaN(z_pred_) or IsNaN(y_pred)")
                    return -1
                adv_loss_ = loss(z_pred_, z_true_)
                cdr_loss = loss(y_pred, y_true)

                total_loss = cdr_loss + lam * adv_loss_
                total_loss.backward()
                optimizer.step()

                total_adv += adv_loss_.item()
                total_cdr += cdr_loss.item()
                total_total_loss += total_loss.item()

                train_y_true_list.append(y_true.cpu().numpy().flatten())
                train_y_pred_list.append(y_pred.detach().cpu().numpy().flatten())

            train_y_true = np.concatenate(train_y_true_list)
            train_y_pred = np.concatenate(train_y_pred_list)
            train_auc, train_acc, train_prec, train_rec, train_f1 = scores(
                train_y_true, train_y_pred
            )

            with torch.no_grad():
                EN_model.eval()
                GCN_model.eval()
                ADV_model.eval()

                F_gexpr_val, _, _ = EN_model(X_gexpr_val)
                F_t_gexpr_val, _, _ = EN_model(X_t_gexpr_val)

                F_g_t_gexpr_val = torch.cat((F_gexpr_val, F_t_gexpr_val))
                z_pred_val = ADV_model(F_g_t_gexpr_val)
                y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_gexpr_val)
                y_true_val = Y_val.cpu().detach().numpy().flatten()
                y_pred_val_np = y_pred_val.cpu().detach().numpy().flatten()
                val_auc, val_acc, val_precision, val_recall, val_f1 = scores(
                    y_true_val, y_pred_val_np
                )

                F_TEST_gexpr, _, _ = EN_model(TX_gexpr_data_test)
                y_pred_test = GCN_model(
                    TX_drug_feat_data_test, TX_drug_adj_data_test, F_TEST_gexpr
                )
                y_true_test = TY_test.cpu().detach().numpy().flatten()
                y_pred_test_np = y_pred_test.cpu().detach().numpy().flatten()
                test_auc, test_acc, test_precision, test_recall, test_f1 = scores(
                    y_true_test, y_pred_test_np
                )

            wandb.log(
                {
                    "epoch": epoch,
                    "avg_adv_loss": total_adv / len(GDSC_Loader),
                    "avg_cdr_loss": total_cdr / len(GDSC_Loader),
                    "avg_total_loss": total_total_loss / len(GDSC_Loader),
                    "train_auc": train_auc,
                    "train_acc": train_acc,
                    "train_precision": train_prec,
                    "train_recall": train_rec,
                    "train_f1": train_f1,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                    "test_auc": test_auc,
                    "test_acc": test_acc,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_f1": test_f1,
                }
            )

            # Early stopping on val_auc
            if val_auc > best_auc:
                best_auc = val_auc
                best_metric["Accuracy"] = test_acc
                best_metric["AUC"] = test_auc
                best_metric["F1"] = test_f1
                best_metric["Recall"] = test_recall
                best_metric["Precision"] = test_precision
                wait = 0
                torch.save(
                    {
                        "EN_model": EN_model.state_dict(),
                        "GCN_model": GCN_model.state_dict(),
                        "ADV_model": ADV_model.state_dict(),
                    },
                    weight_path,
                )
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

        EN_model = Encoder(gexpr.shape[1], nz, device)
        GCN_model = GCN(
            drug_feat.shape[2],
            [256, 256, 256],
            h_dims=[d_dim, nz + d_dim],
            use_dropout=False,
        )

        checkpoint = torch.load(weight_path)
        EN_model.load_state_dict(checkpoint["EN_model"])
        GCN_model.load_state_dict(checkpoint["GCN_model"])
        EN_model.to(device)
        GCN_model.to(device)

        with torch.no_grad():
            EN_model.eval()
            GCN_model.eval()

            F_gexpr = EN_model(gexpr)[0]
            y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)

        return y_pred.cpu()


def train_PANCDR_full_cv(
    n_splits,
    data,
    best_params_file,
    result_file="./logs/full_CV_PANCDR/GDSC_results.csv",
    param_summary_file="./logs/full_CV_PANCDR/GDSC_param_summary.csv",
):

    X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = data
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_list = pd.read_csv(best_params_file)["Best_params"].tolist()

    auc_test_df = pd.DataFrame(columns=["Fold", "Test_AUC", "params", "end_epoch"])
    best_params_all_folds = pd.DataFrame(
        columns=["Fold", "Test_AUC", "params", "end_epoch"]
    )
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

            train_data = [
                X_drug_feat_train,
                X_drug_adj_train,
                X_gexpr_train,
                Y_train,
                t_gexpr_feature,
            ]
            test_data = [X_drug_feat_test, X_drug_adj_test, X_gexpr_test, Y_test]

            current_params = eval(param_str)
            model = train_PANCDR(
                train_data, test_data, outer_fold=fold, project="PANCDR_fullCV"
            )

            while True:
                auc_TEST, end_epoch = model.train(
                    current_params,
                    weight_path=f"../checkpoint/GDSC_fullCV_PANCDR/model_{fold}_{i}.pt",
                )
                if auc_TEST != -1:
                    break

            auc_test_df = pd.concat(
                [
                    auc_test_df,
                    pd.DataFrame(
                        [[fold, auc_TEST, current_params, end_epoch]],
                        columns=["Fold", "Test_AUC", "Best_params", "Best_epoch"],
                    ),
                ]
            )
            auc_test_df.to_csv(result_file, index=False)

    return auc_test_df


def IsNaN(pred):
    return torch.isnan(pred).sum() > 0
