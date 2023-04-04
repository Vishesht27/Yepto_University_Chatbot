import json

class DataConfig:
    __dataset = None
    def __init__(self) -> None:
        if(DataConfig.__dataset is None):
            DataConfig.__dataset = DataConfig.__get_data()
        else:
            raise Exception("DataConfig is a singleton class")
    
    @classmethod
    def get_dataset_instance(cls) -> list:
        return DataConfig.__dataset
    
    
    @classmethod
    def __get_data(cls) -> list:
        with open('chatbot/assets/data.json', 'r') as f:
            dataset = json.load(f)['intents']
        
        return dataset