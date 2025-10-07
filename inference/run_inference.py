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
    batch_size = cfg.get('batch_size',1)
    num_workers = cfg.get('num_workers',1)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Adjust based on your GPU memory
        shuffle=False,
        collate_fn=processor.collate_fn,
        num_workers=num_workers,
    )

    # Process in batches
    for batch in tqdm(dataloader, desc="Processing turns"):
        processor.process_batch(batch)
    processor.on_end()

def run_config(cfg: DictConfig):
    metadata_handler = hydra.utils.instantiate(cfg.metadata)
    if cfg.get('info',False):
        metadata_handler.info()
        exit()
    num_threads = cfg.num_threads
    data = metadata_handler.get_data()
    if num_threads > len(data):
        print(f'Requested {num_threads} threads, but only {len(data)} data chunks available. Reducing threads.')
        num_chunks = len(data)
    else:
        num_chunks = num_threads
    chunks = np.array_split(metadata_handler.get_data(), num_chunks)

    if num_threads > 1:
        executor = submitit.AutoExecutor(folder="log_submitit")
        executor.update_parameters(
            timeout_min=60 * 24,
            cpus_per_task=cfg.get('cpus_per_task',12),
            gpus_per_node=cfg.gpus_per_thread,
            slurm_account=cfg.get('slurm_account',""),
            slurm_qos=cfg.get('slurm_qos',""),
            slurm_job_name=cfg.get('slurm_job_name',"inference"),
        )

        jobs = []
        for chunk in chunks:
            job = executor.submit(process_chunk, metadata_handler, cfg, list(chunk))
            jobs.append(job)

        for job in jobs:
            print(f"Job {job.job_id} submitted.")
    else:
        process_chunk(metadata_handler, cfg, list(chunks[0]))


# todo implement numpy array split in a way which does not transform everything to numpy arrays

@hydra.main(
    # config_path="../../../conf/finetuning",
    # config_path="configs",
    version_base=None,
)
def main(cfg: DictConfig):
    print(cfg)
    run_config(cfg)

if __name__ == "__main__":
    main()
