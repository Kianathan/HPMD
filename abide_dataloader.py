import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
__all__ = ['MMDataLoader']      
logger = logging.getLogger('HPMD')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'abide': self.__init_zs,
        }
        DATASET_MAP[args['dataset_name']]()

    def __init_zs(self):
        with open(self.args['featurePath'], 'rb') as f:
            data = pickle.load(f)
        self.PHENO = data[self.mode]['PHENO']
        self.ANAT = data[self.mode]['ANAT']
        self.FUNC = data[self.mode]['FUNC']
        self.CORRELATION = data[self.mode]['CORRELATION']
        self.labels = {
            'M': np.array(data[self.mode]['label'])
        }
        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        return (self.PHENO.shape[1], self.ANAT.shape[1], self.FUNC.shape[1], self.CORRELATION.shape[1])

    def get_feature_dim(self):
        return self.PHENO.shape[2], self.ANAT.shape[2], self.FUNC.shape[2], self.CORRELATION.shape[2]

    def __getitem__(self, index):
        sample = {
            'PHENO': torch.Tensor(self.PHENO[index]), 
            'ANAT': torch.Tensor(self.ANAT[index]), 
            'FUNC': torch.Tensor(self.FUNC[index]), 
            'CORRELATION': torch.Tensor(self.CORRELATION[index]),  
            'index': index,
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        return sample

def inspect_dataloader(dataLoader, splits=('train','valid','test'), n_batches=1):
    for split in splits:
        loader = dataLoader[split]
        print(f"\n=== Inspecting split: {split} | dataset size={len(loader.dataset)} | batch_size={loader.batch_size} ===")
        it = iter(loader)
        for i in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                print(f"  [Info] only {i} batches available.")
                break

            print(f"  -- Batch {i} --")

            if isinstance(batch, dict):
                for k, v in batch.items():
                    if k == 'labels':
                        print(f"    {k}:")
                        if isinstance(v, dict):
                            for lk, lv in v.items():
                                if isinstance(lv, torch.Tensor):
                                    t = lv.detach().cpu()
                                    if t.ndim == 0:
                                        vals = [float(t.item())]
                                    elif t.ndim == 1:
                                        vals = t.tolist()
                                    elif t.ndim == 2:
                                        if t.size(1) == 1:
                                            vals = t.squeeze(1).tolist()
                                        else:
                                            vals = [row.tolist() for row in t]
                                    else:
                                        b = t.shape[0]
                                        vals = [t[i].reshape(-1).tolist() for i in range(b)]
                                    if not isinstance(vals, list):
                                        vals = [vals]
                                    print(f"      - {lk}: length={len(vals)} values={vals}")
                                elif isinstance(lv, np.ndarray):
                                    arr = lv.reshape(-1) if lv.ndim != 1 else lv
                                    vals = arr.tolist()
                                    print(f"      - {lk}: length={len(vals)} values={vals}")
                                # list / tuple
                                elif isinstance(lv, (list, tuple)):
                                    if len(lv) > 0 and isinstance(lv[0], torch.Tensor):
                                        vals = [x.detach().cpu().squeeze().tolist() if isinstance(x, torch.Tensor) else x for x in lv]
                                    else:
                                        vals = list(lv)
                                    print(f"      - {lk}: length={len(vals)} values={vals}")
                                else:
                                    try:
                                        print(f"      - {lk}: type={type(lv)} value={lv}")
                                    except Exception:
                                        print(f"      - {lk}: (unprintable type={type(lv)})")
                        else:
                            lv = v
                            if isinstance(lv, torch.Tensor):
                                t = lv.detach().cpu()
                                if t.ndim == 0:
                                    vals = [float(t.item())]
                                else:
                                    vals = t.squeeze().tolist()
                                    if not isinstance(vals, list):
                                        vals = [vals]
                                print(f"      - (labels): length={len(vals)} values={vals}")
                            elif isinstance(lv, np.ndarray):
                                vals = lv.reshape(-1).tolist() if lv.ndim != 1 else lv.tolist()
                                print(f"      - (labels): length={len(vals)} values={vals}")
                            elif isinstance(lv, (list, tuple)):
                                print(f"      - (labels): length={len(lv)} values={list(lv)}")
                            else:
                                print(f"      - (labels): type={type(lv)} value={lv}")
                        continue 
            
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: tensor shape={tuple(v.shape)}, dtype={v.dtype}")
                    elif isinstance(v, (list, tuple)):
                        if len(v) > 0 and isinstance(v[0], torch.Tensor):
                            print(f"    {k}: list of tensors len={len(v)}, first tensor shape={tuple(v[0].shape)}, dtype={v[0].dtype}")
                        else:
                            print(f"    {k}: list/tuple len={len(v)}, sample type={type(v[0]) if len(v)>0 else None}")
                    else:
                        print(f"    {k}: type={type(v)}; value_preview={str(v)[:100]}")

            elif isinstance(batch, (list, tuple)):
                print(f"    batch is a list of {len(batch)} samples; inspecting labels across samples:")
                for si, sample in enumerate(batch):
                    if isinstance(sample, dict) and 'labels' in sample:
                        lab = sample['labels']
                        print(f"      sample {si}:")
                        if isinstance(lab, dict):
                            for lk, lv in lab.items():
                                if isinstance(lv, torch.Tensor):
                                    val = lv.detach().cpu().squeeze().tolist()
                                    print(f"        - {lk}: {val}")
                                else:
                                    print(f"        - {lk}: {lv}")
                        else:
                            if isinstance(lab, torch.Tensor):
                                val = lab.detach().cpu().squeeze().tolist()
                                print(f"        - (labels): {val}")
                            else:
                                print(f"        - (labels): {lab}")
                    else:
                        print(f"      sample {si}: no labels field or unexpected sample type: {type(sample)}")
            else:
                print("    Unhandled batch type:", type(batch))
            print("")


def MMDataLoader(args, num_workers):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    if 'seq_lens' in args:      
        args['seq_lens'] = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args['batch_size'],
                       num_workers=num_workers,
                       shuffle=True)
        for ds in datasets.keys()
    }

    return dataLoader

