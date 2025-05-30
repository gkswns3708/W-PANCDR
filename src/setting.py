from config import Config
import os
import pandas as pd
# config에 잇는 모든 .csv 파일에 필요한 paraent dir를 생성해야함.

def create_directories_for_csv_files(config):
    # Create directories for CSV files if they do not exist
    csv_config = config.get_config()['csv']
    directories = [
        os.path.dirname(path) for path in csv_config.values() if path and not os.path.exists(os.path.dirname(path))
    ]
    for path in directories:
        os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    pass