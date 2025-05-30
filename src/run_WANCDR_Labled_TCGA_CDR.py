import random,os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
from sklearn import metrics
import pandas as pd
from utils import DataGenerate, DataFeature, mkdirs
import argparse
from ModelTraining.model_training import train_WANCDR
from config import Config
device = torch.device('cuda')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

TCGA_label_set = ["ACC","ALL","BLCA","BRCA","CESC","CLL","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG","LUSC","MB",
                  "MESO","MM","NB","OV","PAAD","PRAD","SCLC","SKCM","STAD",
                  "THCA","UCEC",'COAD/READ','']
DPATH = '../data'
Drug_info_file = '%s/GDSC/GDSC_drug_binary.csv'%DPATH
Cell_line_info_file = '%s/GDSC/Cell_Lines_Details.txt'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Cancer_response_exp_file = '%s/GDSC/GDSC_binary_response_151.csv'%DPATH
Gene_expression_file = '%s/GDSC/GDSC_expr_z_702.csv'%DPATH
Max_atoms = 100
P_Gene_expression_file = '%s/TCGA/Pretrain_TCGA_expr_702_01A.csv'%DPATH
T_Drug_info_file = '%s/TCGA/TCGA_drug_new.csv'%DPATH
T_Patient_info_file = '%s/TCGA/TCGA_type_new.txt'%DPATH
T_Drug_feature_file = '%s/TCGA/drug_graph_feat'%DPATH
T_Cancer_response_exp_file = '%s/TCGA/TCGA_response_new.csv'%DPATH
T_Gene_expression_file = '%s/TCGA/TCGA_expr_z_702.csv'%DPATH

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='W-PANCDR Full CV')
    args.add_argument('--mode', type=str, default='WANCDR', choices=['PANCDR', 'WANCDR'], help='Mode of training: PANCDR or WANCDR')
    args.add_argument('--strategy', type=str, default='5FoldCV', choices=['5FoldCV', 'CDRTCGA'], help='Cross-validation strategy: 5FoldCV or Nested')
    args.add_argument('--optimization', type=str, default='Test AUC', choices=['Test AUC', 'W_distance'], help='Optimization metric: Test AUC or W_distance')
    args = args.parse_args()
    
    config = Config(args.mode, args.strategy, args.optimization)
    config = config.get_config()
    assert config is not None, "Configuration loading failed. Please check the config file."
    
    mkdirs(config)
    
    
    drug_feature,gexpr_feature, t_gexpr_feature, data_idx = DataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file)
    T_drug_feature, T_gexpr_feature, T_data_idx = DataGenerate(T_Drug_info_file,T_Patient_info_file,T_Drug_feature_file,T_Gene_expression_file,None,T_Cancer_response_exp_file,dataset="TCGA") # -> Labled TCGA
    TX_drug_data_test,TX_gexpr_data_test,TY_test,Tcancer_type_test_list = DataFeature(T_data_idx,T_drug_feature,T_gexpr_feature,dataset="TCGA")

    TX_drug_feat_data_test = [item[0] for item in TX_drug_data_test]
    TX_drug_adj_data_test = [item[1] for item in TX_drug_data_test]
    TX_drug_feat_data_test = np.array(TX_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    TX_drug_adj_data_test = np.array(TX_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  

    X_drug_data,X_gexpr_data,Y,cancer_type_train_list = DataFeature(data_idx,drug_feature,gexpr_feature)
    
    X_drug_feat_data = [item[0] for item in X_drug_data]
    X_drug_adj_data = [item[1] for item in X_drug_data]
    X_drug_feat_data = np.array(X_drug_feat_data)
    X_drug_adj_data = np.array(X_drug_adj_data)

    train_data = [X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature]
    test_data = [TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test]
    print(type(TX_drug_feat_data_test), "-type(TX_drug_feat_data_test)")
    df = pd.read_csv("tuned_hyperparameters/TCGA_CV_params.csv")
    best_params = eval(df.loc[(df["Model"]=="WANCDR") & (df["Classification"]=="T"),"Best_params"].values[0])
    csv_path = config['csv']['result_file_path']

    # ➤ 1. 기존 CSV 불러오기 (있으면)
    if os.path.exists(csv_path):
        result_df = pd.read_csv(csv_path)
        done_iters = set(result_df['Iteration'].dropna().astype(int).tolist())
    else:
        result_df = pd.DataFrame(columns=["Iteration", "Accuracy", "AUC", "F1", "Recall", "Precision"])
        done_iters = set()

    # ➤ 2. 반복 시작
    for iter in range(100):
        if iter in done_iters:
            print(f"Skipping iteration {iter}, already recorded.")
            continue

        weight_path = f'../checkpoint/TCGA_WANCDR/{iter}_model.pt'
        model = train_WANCDR(train_data, test_data, outer_fold=iter, config=config)
        best_metric, end_epoch = model.train(
            config,
            best_params,
            weight_path=os.path.join(config['train']['weight_path'], f'model_{iter}.pt')
        )

        # --- 여기에 추가 ---
        # 1) 현재 iteration 결과를 DataFrame으로 만듭니다.
        metrics = best_metric.copy()
        metrics['iteration'] = iter
        df_current = pd.DataFrame([metrics])

        # 2) 기존 CSV가 있으면 불러와서 concat, 없으면 그대로 사용
        if os.path.exists(csv_path):
            df_history = pd.read_csv(csv_path)
            df_all = pd.concat([df_history, df_current], ignore_index=True)
        else:
            df_all = df_current

        # 3) 다시 덮어쓰기
        df_all.to_csv(csv_path, index=False)
        print(f"Saved metrics for iteration {iter} to {csv_path}")