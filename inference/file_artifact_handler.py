
from inference.base import MetadataHandler
from typing import List,Any,Tuple
import os

class FileArtifactHandler:
    """
    A for a common pattern where each element should produce a single file as output, when resuming inference already finished
    elements are skipped
    """

    def get_data(self) -> List[Any]:
        data = self.get_file_data()
        filtered = [x for x in data if not os.path.exists(x[1])]
        print(f"{len(filtered)} of {len(data)} elements remaining")
        return filtered
    
    def info(self):
        data = self.get_file_data()
        num_total = len(data)
        num_finished = sum([1 for x in data if os.path.exists(x[1])])
        print(f"{num_finished} of {num_total} elements finished")
    
    def get_file_data(self) -> List[Tuple[Any,str]]:
        """
        Should return a list of tuples where the first element is metadata defining the input and the second element
        is the path to the artifiact the processor should generate. If this file exists then the element will be skipped
        """
        raise NotImplementedError