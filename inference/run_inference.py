import json
import os
from glob import glob
from typing import Any, List

import hydra
import numpy as np
import submitit
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from inference.base import InferenceProcessor, MetadataHandler


def process_chunk(metadata_handler: MetadataHandler, cfg, chunk):
    dataset = metadata_handler.get_dataset(chunk)
    processor: InferenceProcessor = hydra.utils.instantiate(cfg.processor)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Adjust based on your GPU memory
        shuffle=False,
        collate_fn=processor.collate_fn,
        num_workers=4,
    )

    # Process in batches
    for batch in tqdm(dataloader, desc="Processing turns"):
        processor.process_batch(batch)
    processor.on_end()


@hydra.main(
    # config_path="../../../conf/finetuning",
    # config_path="configs",
    config_name="inference/inference.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    print(cfg)
    metadata_handler = hydra.utils.instantiate(cfg.metadata)
    num_threads = cfg.num_threads
    chunks = np.array_split(metadata_handler.get_data(), num_threads)

    if num_threads > 1:
        executor = submitit.AutoExecutor(folder="log_submitit")
        executor.update_parameters(
            timeout_min=60 * 24,
            cpus_per_task=12,  # Adjust based on your needs
            gpus_per_node=cfg.gpus_per_thread,  # Adjust based on your needs
            slurm_account="siro",
            slurm_qos="siro_high",
            slurm_job_name="inference",
        )

        jobs = []
        for chunk in chunks:
            job = executor.submit(process_chunk, metadata_handler, cfg, list(chunk))
            jobs.append(job)

        for job in jobs:
            print(f"Job {job.job_id} submitted.")
    else:
        process_chunk(metadata_handler, cfg, list(chunks[0]))


if __name__ == "__main__":
    main()
