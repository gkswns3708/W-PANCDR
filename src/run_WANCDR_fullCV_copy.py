import random,os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
israndom=False
from utils import DataGenerate, DataFeature
from ModelTraining.model_training import train_WANCDR_full_cv
import argparse
from config import Config
from utils import summary_results, mkdirs
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

nz_ls = [100, 128, 256]
h_dims_ls = [100, 128, 256]
lr_ls = [0.001, 0.0001]
lam_ls = [0.01, 0.001, 0.0001]
batch_size_ls = [[128,14],[256,28]]



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='W-PANCDR Full CV')
    args.add_argument('--mode', type=str, default='WANCDR', choices=['PANCDR', 'WANCDR'], help='Mode of training: PANCDR or WANCDR')
    args.add_argument('--strategy', type=str, default='5FoldCV', choices=['5FoldCV', 'Nested'], help='Cross-validation strategy: 5FoldCV or Nested')
    args.add_argument('--test_metric', type=str, default='Test AUC', choices=['Test AUC', 'W_distance'], help='Optimization metric: Test AUC or W_distance')
    args = args.parse_args()
    
    config = Config(mode=args.mode, strategy=args.strategy, test_metric=args.test_metric)
    config = config.get_config()
    assert config is not None, "Configuration loading failed. Please check the config file."
    
    mkdirs(config)
    
    drug_feature,gexpr_feature, t_gexpr_feature, data_idx = DataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file)
    T_drug_feature, T_gexpr_feature, T_data_idx = DataGenerate(T_Drug_info_file,T_Patient_info_file,T_Drug_feature_file,T_Gene_expression_file,None,T_Cancer_response_exp_file,dataset="TCGA")
    TX_drug_data_test,TX_gexpr_data_test,TY_test,Tcancer_type_test_list = DataFeature(T_data_idx,T_drug_feature,T_gexpr_feature,dataset="TCGA")

    TX_drug_feat_data_test = [item[0] for item in TX_drug_data_test]
    TX_drug_adj_data_test = [item[1] for item in TX_drug_data_test]
    TX_drug_feat_data_test = np.array(TX_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    TX_drug_adj_data_test = np.array(TX_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  

    TX_drug_feat_data_test = torch.FloatTensor(TX_drug_feat_data_test)
    TX_drug_adj_data_test = torch.FloatTensor(TX_drug_adj_data_test)
    TX_gexpr_data_test = torch.FloatTensor(TX_gexpr_data_test)
    TY_test = torch.FloatTensor(TY_test)

    X_drug_data,X_gexpr_data,Y,cancer_type_train_list = DataFeature(data_idx,drug_feature,gexpr_feature)
    X_drug_feat_data = [item[0] for item in X_drug_data]
    X_drug_adj_data = [item[1] for item in X_drug_data]
    X_drug_feat_data = np.array(X_drug_feat_data)
    X_drug_adj_data = np.array(X_drug_adj_data)

    
    print("t_gexpr_feature shape:", t_gexpr_feature.shape)
    print()
    print("X_drug_feat_data shape:", X_drug_feat_data.shape)
    print("X_drug_adj_data shape:", X_drug_adj_data.shape)
    print("X_gexpr_data shape:", X_gexpr_data.shape)
    print("Y shape:", Y.shape)
    print()
    print("TX_drug_feat_data_test shape:", TX_drug_feat_data_test.shape)
    print("TX_drug_adj_data_test shape:", TX_drug_adj_data_test.shape)
    print("TX_gexpr_data_test shape:", TX_gexpr_data_test.shape)
    print("TY_test shape:", TY_test.shape)

            

    train_data = [X_drug_feat_data,X_drug_adj_data,X_gexpr_data,Y,t_gexpr_feature]
    test_data = [TX_drug_feat_data_test,TX_drug_adj_data_test,TX_gexpr_data_test,TY_test]
    auc_test_df = train_WANCDR_full_cv(train_data, test_data, result_file='', config=config)

    auc_test_df.to_csv(f'GDSC_{config["mode"]}_{config["strategy"]}_{config["test_metric"]}.csv', sep=',')
    # 추가적으로 이렇게 n_outer_split/n_params_per_fold 마다 나온 결과들을 종합해서 
    # config['csv']['total_result_path']에 저장하는 로직 추가 예정
    # 이때 저장할 때는 각 hyperparmeter의 조합별로의 평균/분산 값을 저장.
    # 예시:
    # {
    #     'nz': 100,
    #     'h_dims': 100,
    #     'lr': 0.001,
    #     'lam': 0.01,
    #     'batch_size': [128, 14],
    #     'mean_auc': 0.85,
    #     'std_auc': 0.02,
    #     'mean_f1': 0.80,
    #     'std_f1': 0.03,
    #     'mean_recall': 0.78,
    #     'std_recall': 0.01,
    #     'mean_precision': 0.82,
    #     'std_precision': 0.02
    # }
    # 이 결과는 추후에 모델의 성능을 비교하거나, 최적의 하이퍼파라미터 조합을 찾는 데 사용될 수 있습니다.
    # utils.py에 summary_results 함수를 추가하여 이 작업을 수행할 수 있습니다.
    summary_results(auc_test_df, config)
    
    
    

