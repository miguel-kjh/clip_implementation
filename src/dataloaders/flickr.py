import os
import pandas as pd
from typing import Optional

from .ImageRetrievalDataset import ImageRetrievalDataset


class Flickr8kDataset(ImageRetrievalDataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 100,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__(dataset_path, tokenizer, target_size, max_length, lazy_loading)

    def fetch_dataset(self):
        annotations = pd.read_csv(os.path.join(self.dataset_path, "captions.txt"))
        image_files = [
            os.path.join(self.dataset_path, "Images", image_file)
            for image_file in annotations["image"].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations["caption"].to_list()
        return image_files, captions

class Flickr30kDataset(ImageRetrievalDataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer=None,
        target_size: Optional[int] = None,
        max_length: int = 100,
        lazy_loading: bool = False,
    ) -> None:
        super().__init__(dataset_path, tokenizer, target_size, max_length, lazy_loading)

    def fetch_dataset(self):
        annotations = pd.read_csv(
            os.path.join(self.dataset_path, "results.csv"), sep='|'
        )
        annotations = annotations.dropna()
        image_files = [
            os.path.join(self.dataset_path, "flickr30k_images", image_file)
            for image_file in annotations["image_name"].to_list()
        ]
        for image_file in image_files:
            assert os.path.isfile(image_file)
        captions = annotations[" comment"].tolist()
        return image_files, captions