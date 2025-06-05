import csv
from fcntl import DN_DELETE
import pandas as pd
import os
import hickle as hkl
import numpy as np
import scipy.sparse as sp
import random
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from umap import UMAP
import umap.umap_ as umap
import torch
import json
import seaborn as sns
from sklearn import metrics
device = torch.device('cuda')

israndom=False
Max_atoms = 100
TCGA_label_set = ["ACC","ALL","BLCA","BRCA","CESC","CLL","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG","LUSC","MB",
                  "MESO","MM","NB","OV","PAAD","PRAD","SCLC","SKCM","STAD",
                  "THCA","UCEC",'COAD/READ','']

def DataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file,is_regr=False,dataset='GDSC'):
    if dataset.upper() == "GDSC":
        data = MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file,is_regr)
    elif dataset.upper() == "TCGA":
        data = T_MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,Cancer_response_exp_file)
    else:
        print("Please check the dataset. This function only works for \"GDSC\" or \"TCGA\"")
    return data

def DataFeature(data_idx,drug_feature,gexpr_feature,dataset="GDSC"):
    if dataset.upper() == "GDSC":
        data = FeatureExtract(data_idx,drug_feature,gexpr_feature)
    elif dataset.upper() == "TCGA":
        data = T_FeatureExtract(data_idx,drug_feature,gexpr_feature)
    else:
        print("Please check the dataset. This function only works for \"GDSC\" or \"TCGA\"")
    return data

#split into training and test set
def DataSplit(data_idx,ratio = 0.95):
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx,data_test_idx

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

def MetadataGenerate(Drug_info_file,Cell_line_info_file,Drug_feature_file,Gene_expression_file,P_Gene_expression_file,Cancer_response_exp_file,is_regr=False):
    #drug_id --> pubchem_id
    print("GDSC Dataset, Loading drug info...")
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[-1]:item[-1] for item in rows if item[-1].isdigit()}

    #map cellline --> cancer type
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[0]
        TCGA_label = line.strip().split('\t')[9]
        #if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    experiment_data.columns = experiment_data.columns.astype(str)
    #filter experiment data
    drug_match_list=[item for item in experiment_data.columns if item in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data[drug_match_list]
    
    #load TCGA pretrain gene expression
    t_gexpr_feature = pd.read_csv(P_Gene_expression_file, index_col=0)
    #t_gexpr_feature = t_gexpr_feature.T

    data_idx = []
    if is_regr:
        for each_drug in experiment_data_filtered.columns:
            for each_cellline in experiment_data_filtered.index:
                pubchem_id = drugid2pubchemid[each_drug]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gexpr_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[each_cellline,each_drug]) and each_cellline in cellline2cancertype.keys():
                        IC50 = float(experiment_data_filtered.loc[each_cellline,each_drug])
                        data_idx.append((each_cellline,pubchem_id,IC50,cellline2cancertype[each_cellline]))  
    else:
        for each_drug in experiment_data_filtered.columns:
            for each_cellline in experiment_data_filtered.index:
                pubchem_id = drugid2pubchemid[each_drug]
                if str(pubchem_id) in drug_pubchem_id_set and each_cellline in gexpr_feature.index:
                    if not np.isnan(experiment_data_filtered.loc[each_cellline,each_drug]) and each_cellline in cellline2cancertype.keys():
                        binary_IC50 = int(experiment_data_filtered.loc[each_cellline,each_drug])
                        data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline])) 
    
    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return drug_feature, gexpr_feature, t_gexpr_feature, data_idx

def FeatureExtract(data_idx,drug_feature,gexpr_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    target = np.zeros(nb_instance,dtype='float32')
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,binary_IC50,cancer_type = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id].values
        target[idx] = binary_IC50
        cancer_type_list.append([cancer_type,cell_line_id,pubchem_id])
    return drug_data,gexpr_data,target,cancer_type_list

def T_MetadataGenerate(T_Drug_info_file,T_Patient_info_file,T_Drug_feature_file,T_Gene_expression_file,T_Cancer_response_exp_file):
        #drug_id --> pubchem_id
    print("TCGA Dataset, Loading drug info...")
    reader = csv.reader(open(T_Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[-1]:item[-1] for item in rows if item[-1].isdigit()}

    #map patient --> cancer type
    patient2cancertype ={}
    for line in open(T_Patient_info_file).readlines()[1:]:
       patient_id = line.split('\t')[0]
       TCGA_label = line.strip().split('\t')[1]
        # if TCGA_label in TCGA_label_set:
       patient2cancertype[patient_id] = TCGA_label

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(T_Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(T_Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(T_Gene_expression_file,sep=',',header=0,index_col=[0])
    
    experiment_data = pd.read_csv(T_Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    #filter experiment data
    drug_match_list=[item for item in experiment_data.columns if item in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data[drug_match_list]

    data_idx = []
    for each_drug in experiment_data_filtered.columns:
        for each_patient in experiment_data_filtered.index:
            pubchem_id = drugid2pubchemid[each_drug]
            if str(pubchem_id) in drug_pubchem_id_set and each_patient in gexpr_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_patient,each_drug]) and each_patient in patient2cancertype.keys():
                    binary_IC50 = int(experiment_data_filtered.loc[each_patient,each_drug])
                    data_idx.append((each_patient,pubchem_id,binary_IC50,patient2cancertype[each_patient])) 
    nb_patient = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d patients and %d drugs were generated.'%(len(data_idx),nb_patient,nb_drugs))
    return drug_feature, gexpr_feature, data_idx

def T_FeatureExtract(data_idx,drug_feature,gexpr_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_gexpr_features = gexpr_feature.shape[1]
    drug_data = [[] for _ in range(nb_instance)]
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    target = np.zeros(nb_instance,dtype='float32')
    for idx in range(nb_instance):
        patient_id,pubchem_id,binary_IC50,cancer_type = data_idx[idx] ###
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        gexpr_data[idx,:] = gexpr_feature.loc[patient_id].values
        target[idx] = binary_IC50
        cancer_type_list.append([cancer_type,patient_id,pubchem_id])
    return drug_data,gexpr_data,target,cancer_type_list

def scores(y_true, y_pred):
    fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr-fpr)
    optimal_thr = thr[optimal_idx]
    y_pred_ = (y_pred > optimal_thr).astype(int)
    auc = metrics.roc_auc_score(y_true,y_pred)
    acc = metrics.accuracy_score(y_true,y_pred_)
    precision = metrics.precision_score(y_true,y_pred_)
    recall = metrics.recall_score(y_true,y_pred_)
    f1 = metrics.f1_score(y_true, y_pred_)
    return auc,acc,precision,recall,f1

def umap_img(model, gexpr, t_gexpr, path):
    ## model: feature extract model
    ## gexpr, t_gexpr: numpy array, GDSC and TCGA gene expression
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['FE_model'])
    model.to(device)
    
    before = pd.concat([gexpr,t_gexpr])
    
    reducer = umap.UMAP(random_state = 12345)
    encoded  = reducer.fit_transform(before.values)  
    embedding_df = pd.DataFrame(encoded, index = before.index)
    
    colors = ['#ED4C67', '#1289A7']
    label_names = ['GDSC', 'TCGA']
    
    fig, ax = plt.subplots(nrows=1,ncols=2)
    fig.set_size_inches(10,5)

    SMALL_SIZE = 10
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 30

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


    encoded_data = embedding_df.values
    labels = np.concatenate((np.zeros(shape=(gexpr.shape[0],),dtype=int), np.ones(shape=(t_gexpr.shape[0],), dtype=int)))
    label_types = np.unique(labels)
    
    label_flags = [0, 0]
    for i in range(encoded_data.shape[0]):
        if label_flags[np.where(label_types == labels[i])[0][0]] == 0:
            ax[0].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                       s = 100,  alpha = 0.4,  linewidth='3',
                       color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]],
                       label = label_names[np.where(label_types == labels[i])[0][0]])
            label_flags[np.where(label_types == labels[i])[0][0]] = 1
        else:
            ax[0].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                       s = 100, alpha = 0.4,  linewidth='3',
                       color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]])
            
    ax[0].legend()
    ax[0].axis("off")
    ax[0].title.set_text("before")
    
    with torch.no_grad():
        model.eval()
        after,_,_ = model(torch.FloatTensor(before.values).to(device))
    after = pd.DataFrame(after.detach().cpu().numpy())
    reducer = umap.UMAP(random_state = 12345)
    encoded  = reducer.fit_transform(after.values)  
    embedding_df = pd.DataFrame(encoded, index = embedding_df.index)
    encoded_data = embedding_df.values
    label_flags = [0, 0]
    for i in range(encoded_data.shape[0]):
        if label_flags[np.where(label_types == labels[i])[0][0]] == 0:
            ax[1].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                   s = 100,  alpha = 0.4,  linewidth='3',
                   color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]],
                   label = label_names[np.where(label_types == labels[i])[0][0]])
            label_flags[np.where(label_types == labels[i])[0][0]] = 1
        else:
            ax[1].scatter(encoded_data[i, 0], encoded_data[i, 1], 
                   s = 100, alpha = 0.4,  linewidth='3',
                   color = '#ffffff', edgecolor = colors[np.where(label_types == labels[i])[0][0]])
            
    ax[1].legend()
    ax[1].axis("off")
    ax[1].title.set_text("after")
    
    plt.show()
    plt.close()

def load_past_params(log_file):
    """실험이 완료된 (즉, AUC가 존재하는) 하이퍼파라미터 조합만 불러온다."""
    if not os.path.exists(log_file):
        return set()
    
    df = pd.read_csv(log_file)
    if 'Best_params' not in df.columns or 'Test_AUC' not in df.columns:
        return set()

    past_params = set()
    for _, row in df.iterrows():
        if pd.isna(row['Best_params']) or pd.isna(row['Test_AUC']):
            continue  # 결과 없는 시도는 무시

        try:
            p = json.loads(row['Best_params'].replace("'", '"'))  # 작은따옴표 → 큰따옴표
            param_tuple = (
                p['nz'], p['d_dim'], p['lr'], p['lr_adv'], p['lam'], tuple(p['batch_size'])
            )
            past_params.add(param_tuple)
        except Exception:
            continue  # JSON 파싱 에러는 무시

    return past_params


def generate_random_params(mode=None, n_params=None, existing_params=set()):
    """기존 조합과 겹치지 않는 새로운 하이퍼파라미터 조합 생성"""
    
    if 'WANCDR' in mode:
        nz_ls = [100, 128, 256]
        d_dims_ls = [100, 128, 256]
        lr_ls = [0.0001, 0.00001] # Encoder + GCN
        lr_adv_ls = [0.0001, 0.00001] # Critic
        lam_ls = [0.001, 0.0001] # BCE + lambda * Adv
        batch_size_ls = [[128, 14], [256, 28]]
        if "5Critic" in mode:
            lr_adv_ls = [value / 5 for value in lr_adv_ls]  # 5개의 Critic을 위한 학습률 조정

        random_params = set()
        attempt_count = 0
        max_attempts = n_params * 50  # 무한루프 방지

        while len(random_params) < n_params and attempt_count < max_attempts:
            attempt_count += 1
            params = (
                random.choice(nz_ls),
                random.choice(d_dims_ls),
                random.choice(lr_ls),
                random.choice(lr_adv_ls),
                random.choice(lam_ls),
                tuple(random.choice(batch_size_ls))
            )
            if params not in existing_params and params not in random_params:
                random_params.add(params)

        # 최종 딕셔너리 포맷으로 변환
        random_params = [
            {'nz': p[0], 'd_dim': p[1], 'lr': p[2], 'lr_adv': p[3], 'lam': p[4], 'batch_size': list(p[5])}
            for p in random_params
        ]
        return random_params
    if mode == "PANCDR":
        nz_ls = [100, 128, 256]
        h_dims_ls = [100, 128, 256]
        lr_ls = [0.001, 0.0001]
        lr_adv_ls = [0.001, 0.0001]
        lam_ls = [1, 0.1, 0.01]
        batch_size_ls = [[128,14],[256,28]]


        random_params = set()
        attempt_count = 0
        max_attempts = n_params * 50  # 무한루프 방지

        while len(random_params) < n_params and attempt_count < max_attempts:
            attempt_count += 1
            params = (
                random.choice(nz_ls),
                random.choice(h_dims_ls),
                random.choice(lr_ls),
                random.choice(lr_adv_ls),
                random.choice(lam_ls),
                tuple(random.choice(batch_size_ls))
            )
            if params not in existing_params and params not in random_params:
                random_params.add(params)

        # 최종 딕셔너리 포맷으로 변환
        random_params = [
            {'nz': p[0], 'd_dim': p[1], 'lr': p[2], 'lr_adv': p[3], 'lam': p[4], 'batch_size': list(p[5])}
            for p in random_params
        ]
        return random_params


# def create_different_random_search_params_df(n_folds=10, n_params_per_fold=20,
#                                    output_file="random_search_params.csv",
#                                    log_file="./logs/GDSC_CV_all.csv"):
#     past_params = load_past_params(log_file)
#     folds_params = {}

#     for fold in range(n_folds):
#         fold_params = generate_random_params(n_params_per_fold, existing_params=past_params)
#         for param in fold_params:
#             param_tuple = (
#                 param['nz'], param['d_dim'], param['lr'], param['lr_adv'], param['lam'], tuple(param['batch_size'])
#             )
#             past_params.add(param_tuple)  # 다음 fold에서도 중복 방지
#         folds_params[f"Fold_{fold}"] = fold_params

#     # DataFrame 변환
#     params_df = pd.DataFrame([
#         [fold, params] for fold, params_list in folds_params.items() for params in params_list
#     ], columns=['Fold', 'Best_params'])

#     params_df.to_csv(output_file, index=False)
#     return params_df

def create_one_random_search_params_df(config = None):
    assert config is not None, "Configuration must be provided."
    past_params = load_past_params(config['csv']['total_result_path'])

    mode = config['mode'] # WANCDR or PANCDR
    n_params = config['hp']['n_params_per_fold']
    params_list = generate_random_params(mode=mode, n_params=n_params, existing_params=past_params)

    # Fold 정보 없이 저장
    params_df = pd.DataFrame([
        [params] for params in params_list
    ], columns=['Hyperparameters'])

    params_df.to_csv(config['csv']['hp_list_path'], index=False)
    return params_df

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
def summary_results(results_df, config):
    """결과를 요약하여 CSV 파일로 저장"""
    summary = {}
    for col in results_df.columns:
        if col == 'Best_params':
            continue
        summary[col] = {
            'mean': results_df[col].mean(),
            'std': results_df[col].std()
        }
    
    # 하이퍼파라미터 정보 추출
    best_params = json.loads(results_df['Best_params'].iloc[0].replace("'", '"'))
    summary.update(best_params)
    
    # DataFrame으로 변환
    summary_df = pd.DataFrame([summary])
    
    # CSV로 저장
    summary_df.to_csv(config['csv']['total_result_path'], index=False)
    
def mkdirs(config):
    all_paths = list(config['csv'].values()) + [config['train']['weight_path']]
    set_paths = set(map(os.path.dirname, all_paths))
    for path in set_paths:
        os.makedirs(path, exist_ok=True)
        
def f1(y_true, y_pred):
    """
    AUC 기반 임계값을 찾아서 F1을 계산하는 헬퍼 함수.
    """
    fpr, tpr, thr = metrics.roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_thr = thr[optimal_idx]
    y_pred_ = (y_pred > optimal_thr).astype(int)
    return metrics.f1_score(y_true, y_pred_)


def load_preprocessed_gdsc(gdsc_root):
    """
    미리 저장해 둔 GDSC .npy 파일을 로드해서 반환
    """
    data = {}
    data["X_drug_feat_data"] = np.load(os.path.join(gdsc_root, "X_drug_feat_data.npy"))
    data["X_drug_adj_data"]  = np.load(os.path.join(gdsc_root, "X_drug_adj_data.npy"))
    data["X_gexpr_data"]     = np.load(os.path.join(gdsc_root, "X_gexpr_data.npy"))
    data["Y"]                = np.load(os.path.join(gdsc_root, "Y.npy"))
    data["t_gexpr_feature"]  = np.load(os.path.join(gdsc_root, "t_gexpr_feature.npy"))
    return data


def load_preprocessed_tcga(tcga_root):
    """
    미리 저장해 둔 TCGA .npy 파일을 로드해서 반환
    """
    data = {}
    data["TX_drug_feat_data_test"] = torch.FloatTensor(
        np.load(os.path.join(tcga_root, "TX_drug_feat_data_test.npy"))
    )
    data["TX_drug_adj_data_test"]  = torch.FloatTensor(
        np.load(os.path.join(tcga_root, "TX_drug_adj_data_test.npy"))
    )
    data["TX_gexpr_data_test"]     = torch.FloatTensor(
        np.load(os.path.join(tcga_root, "TX_gexpr_data_test.npy"))
    )
    data["TY_test"]                = torch.FloatTensor(
        np.load(os.path.join(tcga_root, "TY_test.npy"))
    )
    return data

# UMAP viz
def TCGA_Viz(GDSC_latent, TCGA_latent, save_path=None, title="UMAP Visualization"):
    print(GDSC_latent.shape, TCGA_latent.shape)
    if isinstance(GDSC_latent, torch.Tensor):
        TCGA_cnt = TCGA_latent.shape[0]
        random_indices = np.random.choice(
            GDSC_latent.shape[0], size=TCGA_cnt, replace=False
        )
        GDSC_latent = GDSC_latent[random_indices, :]
        GDSC_latent = GDSC_latent.detach().cpu().numpy()
    if isinstance(TCGA_latent, torch.Tensor):
        TCGA_latent = TCGA_latent.detach().cpu().numpy()

    X = np.concatenate([GDSC_latent, TCGA_latent], axis=0)
    y = np.array(["GDSC"] * len(GDSC_latent) + ["TCGA"] * len(TCGA_latent))

    umap_model = UMAP(
            n_components=2,
            n_neighbors=30,     # 구조 유지
            min_dist=0.05,      # 조밀하게
            spread=0.5,         # 전체 range 줄이기
            random_state=42
        )
    X_embedded = umap_model.fit_transform(X)

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=y,
        palette={"GDSC": "blue", "TCGA": "orange"},
        alpha=0.6,
        s=60
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=["GDSC", "TCGA"], title="Domain")

    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()