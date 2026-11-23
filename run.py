import gc
import logging
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import get_config_regression
from abide_dataloader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import hpmd
import sys
from datetime import datetime
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
logger = logging.getLogger('HPMD')
torch.cuda.set_device(0)

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('HPMD')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def HPMD_run(
    model_name, dataset_name, config=None, config_file="", seeds=[], is_tune=False,
    tune_times=500, feature_T="", feature_A="", feature_V="",
    model_save_dir="", res_save_dir="", log_dir="",
    gpu_ids=[0], num_workers=4, verbose_level=1, mode = ''
):
    model_name = model_name.lower()
    dataset_name = dataset_name.lower()
    
    if config_file != "":
        config_file = Path(config_file)
    else:
        config_file = Path(__file__).parent / "config" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")
    
    base_dir = Path.cwd() / dataset_name
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    ts_dir = base_dir / timestamp
    ts_dir.mkdir(parents=True, exist_ok=True)

    subdirs = [Path(model_save_dir).name, Path(res_save_dir).name, Path(log_dir).name]
    for sub in subdirs:
        subdir = ts_dir / sub
        subdir.mkdir(parents=True, exist_ok=True)

    model_save_dir = ts_dir / Path(model_save_dir).name
    res_save_dir   = ts_dir / Path(res_save_dir).name
    log_dir        = ts_dir / Path(log_dir).name

   
    seeds = seeds if seeds != [] else [1111, 1112, 1113, 1114, 1115]
    logger = _set_logger(log_dir, model_name, dataset_name, verbose_level)
    
    args = get_config_regression(model_name, dataset_name, config_file)
    args.mode = mode
    args['model_save_path'] = Path(model_save_dir)
    args['device'] = assign_gpu(gpu_ids)
    args['train_mode'] = 'classification'
    args['feature_T'] = feature_T
    args['feature_A'] = feature_A
    args['feature_V'] = feature_V
    if config:
        args.update(config)


    nor_save_dir = Path(res_save_dir) / "normal"
    nor_save_dir.mkdir(parents=True, exist_ok=True)
    con_save_dir = Path(res_save_dir) / "contributions"
    con_save_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for i, seed in enumerate(seeds):
        setup_seed(seed)
        args['cur_seed'] = i + 1
        args['seed'] = seed
        result = _run(args, num_workers, is_tune, save_path=nor_save_dir, con_path=con_save_dir)
        print(result)
        model_results.append(result)
    if args.is_distill:             
        criterions = list(model_results[0].keys())
        csv_file = nor_save_dir / f"{dataset_name}.csv"

        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=["Model"] + criterions)

        res = [model_name]
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.mean(values)*100, 2)
            std = round(np.std(values)*100, 2)
            res.append((mean, std))

        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")

def _run(args, num_workers=4, is_tune=False, from_sena=True, save_path=None, con_path=None):
    dataloader = MMDataLoader(args, num_workers)
    print("training for HPMD")

    args.gd_size_low = 64 
    args.w_losses_low = [1, 10]
    args.metric_low = 'l1'

    args.gd_size_high = 32
    args.w_losses_high = [1, 10]
    args.metric_high = 'l1'

    from_idx = [0, 1, 2]
    assert len(from_idx) >= 1

    model = getattr(hpmd, 'HPMD')(args).cuda()

    trainer = ATIO().getTrain(args)

    if args.mode == 'test':
        model.load_state_dict(torch.load(args.model_save_path))
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        sys.stdout.flush()      
        input('[Press Any Key to start another run]')   
    else:
        seed = args.seed
        epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena, con_path=con_path)
        plot_epoch_metrics(epoch_results, save_path=save_path, seed=seed)
        result_path = os.path.join(args.model_save_path, f"emoe_seed{seed}.pth")
        model.load_state_dict(torch.load(result_path))

        results = trainer.do_test(model, dataloader['valid'], mode="VAL", f=1)

        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)
    return results

def plot_epoch_metrics(epoch_results, save_path=None, seed=None):
    splits = ['train', 'valid', 'test']
    metrics_row1 = ['Acc_3', 'F1_score_3', 'Loss']
    metrics_row2 = ['Sensitivity', 'Specificity', 'auc_value']
    
    epochs = list(range(1, len(epoch_results['train']) + 1))
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, metric in enumerate(metrics_row1):
        ax = axs[0, idx]
        for split in splits:
            values = [epoch_results[split][i][metric] for i in range(len(epochs))]
            ax.plot(epochs, values, label=split, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric + ' over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    for idx, metric in enumerate(metrics_row2):
        ax = axs[1, idx]
        for split in splits:
            values = [epoch_results[split][i][metric] for i in range(len(epochs))]
            ax.plot(epochs, values, label=split, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(metric + ' over Epochs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        result_path = os.path.join(str(save_path), f"results_seed{seed}.png")
        plt.savefig(result_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {result_path}")
    
    plt.close(fig)