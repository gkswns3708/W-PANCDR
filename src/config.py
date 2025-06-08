DPATH =  '../data'

class Config(object):
    def __init__(self,mode='WANCDR', test_dataset = 'GDSC', strategy='full', test_metric='AUC', critic_5=False):
        self.test_dataset = test_dataset
        self.mode = mode
        self.strategy = strategy
        self.test_metric = test_metric  # ['AUC', 'W_distance', 'Loss']
        assert self.test_metric in ['AUC', 'W_distance', 'Loss'], "test_metric must be one of ['AUC', 'W_distance', 'Loss']"

    def get_config(self):
        if 'PANCDR' in self.mode:
            return self.get_pancdr_config()
        elif 'WANCDR' in self.mode:
            return self.get_wancdr_config()
        else:
            raise ValueError("Invalid mode. Choose either 'PANCDR' or 'WANCDR'.")

    def get_pancdr_config(self):
        raise NotImplementedError("PANCDR configuration is not implemented yet.")

    def get_wancdr_config(self):
        current_version = f'{self.test_dataset}/{self.mode}/{self.strategy}/{self.test_metric}'
        config = {}
        # For Hyperparameter Settings
        config['mode'] = self.mode
        config['test_dataset'] = self.test_dataset
        config['strategy'] = self.strategy
        config['test_metric'] = self.test_metric
        config['hp'] = {
            'n_params_per_fold': 20,
            'n_outer_splits': 5,
        }
        config['wandb'] = {
            'project_name': f'{self.test_dataset}_{self.mode}_{self.strategy}_{self.test_metric}',
            'mode': 'online',  # or 'disabled' if you don't want to log to WandB
        }
        config['csv'] = {
            'total_result_path': f'./logs/{current_version}/total_result.csv',
            'hp_list_path': f'./logs/{current_version}/hp_list.csv',
            'current_result_path': f'./logs/{current_version}/temp_results.csv',
            'best_params_file_path': f'./logs/{current_version}/params.csv',
            'result_file_path': f'./logs/{current_version}/results.csv',
            'TCGA_result_file_path': f'./logs/{current_version}/TCGA_results.csv',
            'umap_path': f'./logs/{current_version}/umap_result.png',
        }
        config['train'] = {
            'weight_path': f'./logs/{current_version}/',
            'max_epochs': 1000,
        }
        config['preprocessed'] = {
            'gdsc_path': f'{DPATH}/Preprocessed/GDSC',
            'tcga_path': f'{DPATH}/Preprocessed/TCGA',
        }
        config['model'] = {
        }
        return config