DPATH =  '../data'

config = {
    # Data Path
    'Drug_info_file': '%s/GDSC/GDSC_drug_binary.csv' % DPATH,
    'Cell_line_info_file': '%s/GDSC/Cell_Lines_Details.txt' % DPATH,
    'Drug_feature_file': '%s/GDSC/drug_graph_feat' % DPATH,
    'Cancer_response_exp_file': '%s/GDSC/GDSC_binary_response_151.csv' % DPATH,
    'Gene_expression_file': '%s/GDSC/GDSC_expr_z_702.csv' % DPATH,
    'Max_atoms': 100,
    'P_Gene_expression_file': '%s/TCGA/Pretrain_TCGA_expr_702_01A.csv' % DPATH,
    'T_Drug_info_file': '%s/TCGA/TCGA_drug_new.csv' % DPATH,
    'T_Patient_info_file': '%s/TCGA/TCGA_type_new.txt' % DPATH,
    'T_Drug_feature_file': '%s/TCGA/drug_graph_feat' % DPATH,
    'T_Cancer_response_exp_file': '%s/TCGA/TCGA_response_new.csv' % DPATH,
    'T_Gene_expression_file': '%s/TCGA/TCGA_expr_z_702.csv' % DPATH,
    # Model Training Hyperparameter
    'n_outer_splits': 10,
    'n_inner_splits': 5,
    'n_random_search': 20,
    'random_state': 0,
    'device' : 'cuda',
    'weight_path':  f'../checkpoint/FC/kfold/',
    'gpu_num': '0',
}