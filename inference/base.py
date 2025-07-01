from typing import List, Any
from torch.utils.data import Dataset

class InferenceProcessor:
    def transform_batch(self, batch) -> List[Any]:
        """
        Transform a batch of data into a list of results.
        """
        raise NotImplementedError

    def process_results(self, results: List[Any], batch):
        """
        Process the results of the transform_batch function.
        """
        raise NotImplementedError

    def process_batch(self, batch):
        """
        Process a batch of data.
        """
        results = self.transform_batch(batch)
        self.process_results(results, batch)

    def collate_fn(self, batch):
        """
        Collate a batch of data.
        """
        return batch

    def on_end(self):
        """
        Called when the inference is finished. Not designed for multi-threaded performance with slurm right now.
        """

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
        """
        raise NotImplementedError

    def get_dataset(self, chunk: List[Any]) -> Dataset:
        """
        Returns a Dataset for a given chunk of elements from get_data().
        """
        raise NotImplementedError