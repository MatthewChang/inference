from typing import List, Any
from torch.utils.data import Dataset

class InferenceProcessor:
    def transform_batch(self, batch) -> List[Any]:
        """
        Transform a batch of data into a list of results. The batch is generated from a dataset returned by the MetadataHandler.get_datset() method
        then collated by the colate_fn
        """
        raise NotImplementedError

    def process_results(self, results: List[Any], batch):
        """
        Process the results of the transform_batch function.
        """
        pass

    def process_batch(self, batch):
        """
        Process a batch of data.
        """
        results = self.transform_batch(batch)
        self.process_results(results, batch)

    def collate_fn(self, batch):
        """
        Collate a batch of data. Passed into the dataloader for collation of batches from MetadataHandler
        """
        return batch

    def on_end(self):
        """
        Called when the inference is finished. Not designed for multi-threaded performance with slurm right now.
        """
        pass

class MetadataHandler:
    """
    MetadataHandler is a class that handles the metadata for the inference process.
    """

    def get_data(self) -> List[Any]:
        """
        Returns a list of minimal data to be processed.
        This list will be split into chunks and distributed to different threads.
        Each of these chunks will be passed into MetadataHandler.get_dataset() to get a Dataset.
        Normally each element in the return list should be the metadata for a single element in the dataset.

        For example, this function would return a list of file paths, then the get_dataset function
        should return a dataset which takes those file paths and loads the files.
        """
        raise NotImplementedError
    
    def info(self):
        """
        Override to print out any information about the status of the run. See file_artifact_handler.py for an example.
        """
        pass


    def get_dataset(self, chunk: List[Any]) -> Dataset:
        """
        Returns a Dataset for a given chunk of elements from get_data().
        """
        raise NotImplementedError

class DummyDataset(Dataset):
    def __init__(self,x):
        self.x = x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index) -> Any:
        return self.x[index]

class SimpleMetadataHandler:
    """
    Metadata handler which passes through chunks directly
    """
    def get_dataset(self, chunk: List[Any]) -> Dataset:
        return DummyDataset(chunk)