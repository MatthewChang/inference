from inference.base import InferenceProcessor, SimpleMetadataHandler
from torch.utils.data import Dataset
from typing import List,Any

class Exponent(InferenceProcessor):
    def transform_batch(self, batch) -> List[Any]:
        return [2**x for x in batch]

    def process_results(self, results: List[Any], batch):
        print(results)

class DummyDataset(Dataset):
    def __init__(self,x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index) -> Any:
        return self.x[index]

class Numbers(SimpleMetadataHandler):
    def get_data(self) -> List[Any]:
        return list(range(1,10))