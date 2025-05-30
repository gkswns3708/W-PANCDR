DPATH =  '../data'

class Config(object):
    def __init__(self, mode, strategy='full', test_metric='Test AUC'):
        self.mode = mode
        self.strategy = strategy
        self.test_metric = test_metric

    def get_config(self):
        if self.mode == 'PANCDR':
            return self.get_pancdr_config()
        elif self.mode == 'WANCDR':
            return self.get_wancdr_config()
        else:
            raise ValueError("Invalid mode. Choose either 'PANCDR' or 'WANCDR'.")
    
    def get_pancdr_config(self, strategy, test_metric):
        raise NotImplementedError("PANCDR configuration is not implemented yet.")
    
    def get_wancdr_config(self):
        current_version = f'{self.mode}/{self.strategy}/{self.test_metric}'
        config = {}
        # For Hyperparameter Settings
        config['mode'] = self.mode
        config['strategy'] = self.strategy
        config['test_metric'] = self.test_metric
        config['hp'] = {
            'n_params_per_fold': 20,
            'n_outer_splits': 5,
        }
        config['wandb'] = {
            'project_name': f'{self.mode}_{self.strategy}_{self.test_metric}',
            'mode': 'online',  # or 'disabled' if you don't want to log to WandB
        }
        config['csv'] = {
            'total_result_path' : f'./logs/{current_version}/total_result.csv', # 이전까지 실험들의 성능
            'hp_list_path': f'./logs/{current_version}/hp_list.csv',
            'current_result_path': f'./logs/{current_version}/temp_results.csv', # 현재 실험 정리
            'best_params_file_path': f'./logs/{current_version}/params.csv',
            'result_file_path': f'./logs/{current_version}/results.csv',
        }
        config['train'] = {
            'weight_path': f'./logs/{current_version}/',
            'max_epochs' : 200, 
        }
        return config