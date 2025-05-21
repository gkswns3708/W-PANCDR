    # def train(self, params, weight_path='../checkpoint/model.pt'):
    #     run_name = f"outer_fold_{self.outer_fold}_nz_{params['nz']}_d_dim_{params['d_dim']}_lr_{params['lr']}_lr_adv_{params['lr_adv']}_batch_size_{params['batch_size'][0]}_lam_{params['lam']}"
    
    #     # 각 실험(run)을 새로 시작
    #     run = wandb.init(
    #         project=self.project,
    #         config=params,
    #         name=run_name,
    #         reinit=True
    #     )

            
    #     nz, d_dim, lr, lr_critic, lam, batch_size = params.values()
    #     # print("Hyperparameters:")
    #     # print(f"nz: {nz}")
    #     # print(f"d_dim: {d_dim}")
    #     # print(f"lr: {lr}")
    #     # print(f"lr_critic: {lr_critic}")
    #     # print(f"lam: {lam}")
    #     # print(f"batch_size: {batch_size}")
    #     # print(f"device: {device}")

    #     # 데이터 분할 및 Tensor 변환
    #     X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, t_gexpr_feature = self.train_data
    #     TX_drug_feat_data_test, TX_drug_adj_data_test, TX_gexpr_data_test, TY_test = self.test_data
    #     X_t_train, X_t_val = train_test_split(t_gexpr_feature.T.values, test_size=0.05, random_state=0)
    #     X_drug_feat_data_train, X_drug_feat_data_val, X_drug_adj_data_train, X_drug_adj_data_val, \
    #     X_gexpr_data_train, X_gexpr_data_val, Y_train, Y_val = train_test_split(
    #         X_drug_feat_data, X_drug_adj_data, X_gexpr_data, Y, test_size=0.05, random_state=0)
        
    #     X_drug_feat_train = torch.FloatTensor(X_drug_feat_data_train).to(device)
    #     X_drug_adj_train = torch.FloatTensor(X_drug_adj_data_train).to(device)
    #     X_gexpr_train = torch.FloatTensor(X_gexpr_data_train).to(device)
    #     X_t_gexpr_train = torch.FloatTensor(X_t_train).to(device)
    #     Y_train = torch.FloatTensor(Y_train)

    #     X_drug_feat_val = torch.FloatTensor(X_drug_feat_data_val).to(device)
    #     X_drug_adj_val = torch.FloatTensor(X_drug_adj_data_val).to(device)
    #     X_gexpr_val = torch.FloatTensor(X_gexpr_data_val).to(device)
    #     X_t_gexpr_val = torch.FloatTensor(X_t_val).to(device)
    #     Y_val = torch.FloatTensor(Y_val).to(device)
        
    #     TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test).to(device)
    #     TX_drug_adj_data_test  = torch.FloatTensor(TX_drug_adj_data_test).to(device)
    #     TX_gexpr_data_test     = torch.FloatTensor(TX_gexpr_data_test).to(device)
    #     TY_test                = torch.FloatTensor(TY_test).to(device)

    #     # DataLoader 생성
    #     GDSC_Dataset = torch.utils.data.TensorDataset(X_drug_feat_train, X_drug_adj_train, X_gexpr_train, Y_train)
    #     GDSC_Loader = torch.utils.data.DataLoader(dataset=GDSC_Dataset, batch_size=batch_size[0], shuffle=True, drop_last=True)
    #     E_TEST_Dataset = torch.utils.data.TensorDataset(X_t_gexpr_train)
    #     E_TEST_Loader = torch.utils.data.DataLoader(dataset=E_TEST_Dataset, batch_size=batch_size[1], shuffle=True, drop_last=True)

    #     wait, best_auc = 0, 0
    #     # 모델 생성
    #     EN_model = Encoder_FC(X_gexpr_train.shape[1], nz)
    #     GCN_model = GCN(X_drug_feat_train.shape[2], [256,256,256], h_dims=[d_dim, nz+d_dim], use_dropout=False)
    #     Critic_model = Critic(nz)
    #     EN_model.to(device)
    #     GCN_model.to(device)
    #     Critic_model.to(device)
    #     wandb.watch(EN_model,    log="all", log_freq=50)
    #     wandb.watch(GCN_model,   log="all", log_freq=50)
    #     wandb.watch(Critic_model, log="all", log_freq=50)
    #     optimizer = torch.optim.Adam(itertools.chain(EN_model.parameters(), GCN_model.parameters()), lr=lr)
    #     optimizer_critic = torch.optim.Adam(Critic_model.parameters(), lr=lr_critic)
    #     loss_fn = torch.nn.BCELoss()

    #     current_epoch = -1
    #     for epoch in tqdm(range(1000), desc="Epoch", leave=True):
    #         total_critic_loss = 0.0
    #         total_gen_loss = 0.0
    #         num_batches = 0
    #         # training metric 저장을 위한 리스트
    #         train_y_true_list = []
    #         train_y_pred_list = []
            
    #         # 각 배치마다 학습 (원래 방식 그대로)
    #         for i, data in enumerate(tqdm(zip(GDSC_Loader, cycle(E_TEST_Loader)), desc=f"Batch (Epoch {epoch})", leave=False, total=len(GDSC_Loader))):
    #             DataG = data[0]
    #             t_gexpr = data[1][0]
    #             drug_feat, drug_adj, gexpr, y_true = DataG
    #             drug_feat = drug_feat.to(device)
    #             drug_adj = drug_adj.to(device)
    #             gexpr = gexpr.to(device)
    #             y_true = y_true.view(-1, 1).to(device)
    #             t_gexpr = t_gexpr.to(device)

    #             EN_model.train()
    #             GCN_model.train()
    #             Critic_model.train()

    #             # Critic 업데이트 (Encoder 고정)
    #             optimizer_critic.zero_grad()
    #             with torch.no_grad():
    #                 F_gexpr = EN_model(gexpr)      # fake (GDSC)
    #                 F_t_gexpr = EN_model(t_gexpr)    # real (TCGA)
    #             real_validity = Critic_model(F_t_gexpr)  # D(real)
    #             fake_validity = Critic_model(F_gexpr)      # D(fake)
    #             gp = gradient_penalty(Critic_model, F_gexpr, F_t_gexpr, device, gp_weight=10.0)
    #             critic_loss = (fake_validity.mean() - real_validity.mean()) + gp
    #             critic_loss.backward()
    #             optimizer_critic.step()

    #             # Encoder + GCN 업데이트 (Generator 역할)
    #             optimizer.zero_grad()
    #             # latent 추출 (tuple이면 첫번째 요소 사용)
    #             F_gexpr = EN_model(gexpr)[0] if isinstance(EN_model(gexpr), (list, tuple)) else EN_model(gexpr)
    #             F_t_gexpr = EN_model(t_gexpr)[0] if isinstance(EN_model(t_gexpr), (list, tuple)) else EN_model(t_gexpr)
    #             fake_validity_ = Critic_model(F_gexpr)
    #             adv_loss = -fake_validity_.mean()
    #             y_pred = GCN_model(drug_feat, drug_adj, F_gexpr)
    #             cdr_loss = loss_fn(y_pred, y_true)
    #             gen_loss = cdr_loss + lam * adv_loss
    #             gen_loss.backward()
    #             # Gradient Logging
    #             total_grad_norm = torch.sqrt(
    #             sum(p.grad.data.norm(2)**2 for p in itertools.chain(
    #                     EN_model.parameters(), GCN_model.parameters()
    #                 ) if p.grad is not None)
    #             )
    #             wandb.log({
    #                 "train/gen_loss": gen_loss.item(),
    #                 "train/grad_norm": total_grad_norm.item()
    #             })
    #             optimizer.step()
                
    #             total_critic_loss += critic_loss.item()
    #             total_gen_loss += gen_loss.item()
    #             num_batches += 1
                
    #             # 학습 중 매 배치의 y_true와 y_pred를 저장 (평가를 위해)
    #             train_y_true_list.append(y_true.cpu().detach().numpy().flatten())
    #             train_y_pred_list.append(y_pred.cpu().detach().numpy().flatten())
                
    #         # EN model
    #         for name, p in EN_model.named_parameters():
    #             if "weight" in name:
    #                 wandb.log({f"EN/{name}_w_norm": p.data.norm(2).item()})
    #         # GCN model
    #         for name, p in GCN_model.named_parameters():
    #             if "weight" in name:
    #                 wandb.log({f"GCN/{name}_w_norm": p.data.norm(2).item()})
    #         # Critic model
    #         for name, p in Critic_model.named_parameters():
    #             if "weight" in name:
    #                 wandb.log({f"Critic/{name}_w_norm": p.data.norm(2).item()})
    #         # Epoch마다 평균 training loss 계산
    #         avg_critic_loss = total_critic_loss / num_batches
    #         avg_gen_loss = total_gen_loss / num_batches
            
    #         # training metric 계산 (전체 training 데이터에 대해)
    #         train_y_true = np.concatenate(train_y_true_list)
    #         train_y_pred = np.concatenate(train_y_pred_list)
    #         train_auc, train_acc, train_precision, train_recall, train_f1 = scores(train_y_true, train_y_pred)
            
    #         # Epoch 단위 평가: 전체 Validation 및 Test 데이터를 이용하여 한 번에 예측
    #         with torch.no_grad():
    #             EN_model.eval()
    #             GCN_model.eval()
    #             # Validation 평가
    #             F_val = EN_model(X_gexpr_val)
    #             y_pred_val = GCN_model(X_drug_feat_val, X_drug_adj_val, F_val)
    #             y_true_val = Y_val.cpu().detach().numpy().flatten()
    #             y_pred_val_np = y_pred_val.cpu().detach().numpy().flatten()
    #             val_auc, val_acc, val_precision, val_recall, val_f1 = scores(y_true_val, y_pred_val_np)
                
    #             # Test 평가
    #             F_test = EN_model(TX_gexpr_data_test)
    #             y_pred_test = GCN_model(TX_drug_feat_data_test, TX_drug_adj_data_test, F_test)
    #             y_true_test = TY_test.cpu().detach().numpy().flatten()
    #             y_pred_test_np = y_pred_test.cpu().detach().numpy().flatten()
    #             test_auc, test_acc, test_precision, test_recall, test_f1 = scores(y_true_test, y_pred_test_np)

    #             # wandb에 epoch 단위로 로깅 (평균 training loss 및 training metric 포함)
    #             wandb.log({
    #                 "epoch": epoch,
    #                 "avg_critic_loss": avg_critic_loss,
    #                 "avg_gen_loss": avg_gen_loss,
    #                 "train_auc": train_auc,
    #                 "train_acc": train_acc,
    #                 "train_precision": train_precision,
    #                 "train_recall": train_recall,
    #                 "train_f1": train_f1,
    #                 "val_auc": val_auc,
    #                 "val_acc": val_acc,
    #                 "val_precision": val_precision,
    #                 "val_recall": val_recall,
    #                 "val_f1": val_f1,
    #                 "test_auc": test_auc,
    #                 "test_acc": test_acc,
    #                 "test_precision": test_precision,
    #                 "test_recall": test_recall,
    #                 "test_f1": test_f1,
    #                 "loss_val": gen_loss.item()  # 마지막 배치의 Loss (선택 사항)
    #             })
            
    #         if val_auc > best_auc:
    #             wait = 0
    #             best_auc = val_auc
    #             torch.save({
    #                 'EN_model': EN_model.state_dict(), 
    #                 'GCN_model': GCN_model.state_dict(), 
    #                 'Critic_model': Critic_model.state_dict()
    #             }, weight_path)
    #         else:
    #             wait += 1
    #             if wait >= 10: 
    #                 break
    #         current_epoch = epoch
        
    #     run.finish()  # wandb run 종료
    #     return test_auc, current_epoch