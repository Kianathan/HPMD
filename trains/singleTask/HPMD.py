import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str, eva_imp, entropy_balance, GGDUnitDistillLoss
import matplotlib.pyplot as plt
import os
import seaborn as sns
sns.set(style="whitegrid")
logger = logging.getLogger('HPMD')

class HPMD():
    def __init__(self, args):
        self.args = args
        if args.train_mode == 'classification':
            loss_weights=torch.FloatTensor([1.16, 1])
            self.criterion = nn.CrossEntropyLoss(loss_weights.cuda())
        else:
            self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False, con_path=None):
        params = list(model.parameters())
        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        w_allmean = []
        if return_epoch_results:    #False
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'  
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1
            y_pred, y_true,y_pred_probabilities, w_total = [], [], [],[]
            model.train()
            train_loss = 0.0

            left_epochs = self.args.update_epochs   
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    pheno = batch_data['PHENO'].to(self.args.device)
                    anat = batch_data['ANAT'].to(self.args.device)
                    func = batch_data['FUNC'].to(self.args.device)
                    corr = batch_data['CORRELATION'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                
                    output = model(pheno, anat, func, corr, labels.view(-1).long())
                    w = output['channel_weight']    

                    if self.args.train_mode == 'classification':
                        lbl = labels.view(-1).long()
                        y_true.append(lbl.cpu())
                        pred_cls = torch.argmax(output['logits_c'], dim=1)
                        pred_probabilities = F.softmax(output['logits_c'], dim=1).cpu().detach()
                        y_pred_probabilities.append(pred_probabilities)
                        y_pred.append(pred_cls.cpu())
                    else:
                        y_true.append(labels.view(-1,1).cpu())
                        pred_probabilities = F.softmax(output['logits_c'], dim=1).cpu().detach()
                        y_pred_probabilities.append(pred_probabilities.cpu())
                        y_pred.append(output['logits_c'].cpu())

                    if self.args.train_mode == 'classification':
                        loss_task_pheno = self.criterion(output['logits_pheno'], lbl)
                        loss_task_anat = self.criterion(output['logits_anat'], lbl)
                        loss_task_func = self.criterion(output['logits_func'], lbl)
                        loss_task_corr = self.criterion(output['logits_corr'], lbl)
                        loss_task_m = self.criterion(output['logits_c'], lbl)
                    else:       
                        loss_task_pheno = self.criterion(output['logits_pheno'], labels)
                        loss_task_anat = self.criterion(output['logits_anat'], labels)
                        loss_task_func = self.criterion(output['logits_func'], labels)
                        loss_task_corr = self.criterion(output['logits_corr'], labels)
                        loss_task_m = self.criterion(output['logits_c'], labels)

                    pheno_dist = eva_imp(torch.argmax(output['logits_pheno'], dim=1), labels.view(-1))
                    anat_dist = eva_imp(torch.argmax(output['logits_anat'], dim=1), labels.view(-1))
                    func_dist = eva_imp(torch.argmax(output['logits_func'], dim=1), labels.view(-1))
                    corr_dist = eva_imp(torch.argmax(output['logits_corr'], dim=1), labels.view(-1))


                    dist = torch.zeros(func_dist.shape[0], 4).to(self.args.device)
                    
                    for i,_ in enumerate(func_dist):
                        s = 1/(pheno_dist[i]+0.1) + 1/(anat_dist[i]+0.1) + 1/(func_dist[i]+0.1) + 1/(corr_dist[i]+0.1)
                        dist[i][0] = (1/(pheno_dist[i]+0.1)) / s
                        dist[i][1] = (1/(anat_dist[i]+0.1)) / s
                        dist[i][2] = (1/(func_dist[i]+0.1)) / s
                        dist[i][3] = (1/(corr_dist[i]+0.1)) / s

                    loss_sim = torch.mean(torch.mean((dist.detach() - w) ** 2, dim=-1))

                    logits_p_list = [
                        output['logits_pheno'],  # pheno
                        output['logits_anat'],  # anat
                        output['logits_func'],  # func
                        output['logits_corr'],  # corr
                    ]
                    feat_p_list = [
                        output['pheno_proj'],  # pheno
                        output['anat_proj'],  # anat
                        output['func_proj'],  # func
                        output['corr_proj'],  # corr
                    ]

                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
                    ggd_distill = GGDUnitDistillLoss(d_feat=256, logits_dim=output['logits_c'].shape[1]).to(device)
                    loss_ud = ggd_distill(output['logits_c'], output['c_proj'], logits_p_list, feat_p_list)
                
                    loss = loss_task_m + (loss_task_pheno+ loss_task_anat+ loss_task_func+ loss_task_corr)/4 + 0.001*output['L_loss'] + 0.5*loss_sim + 0.1*loss_ud

                    w_total.append(w)
                    loss.backward()
                    train_loss += loss.item()

                    if not left_epochs: 
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()

            w_total = torch.cat(w_total,dim=0)
            w_mean = w_total.mean(dim=0, keepdim=True)
            w_allmean.append(w_mean)
            expert_names = ['PHENO','ANAT','FUNC','CORRELATION']

            train_loss = train_loss / len(dataloader['train'])
            pred, true, probabilities  = torch.cat(y_pred), torch.cat(y_true),torch.cat(y_pred_probabilities)
            train_results = self.metrics(pred, true, probabilities)
            logger.info(
                f">> Epoch: {epochs} "  
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "    
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )

            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            test_results = self.do_test(model, dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]  
            scheduler.step(val_results['Loss'])
            
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                seed = self.args.seed
                model_save_path = self.args.model_save_path
                result_path = os.path.join(model_save_path, f"emoe_seed{seed}.pth")
                torch.save(model.state_dict(), result_path)
                plot_gate_contributions(w_total, con_path, filename=f"best_contributions_seed{seed}.png",expert_names = expert_names)
            w_total = []

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                # test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            if epochs - best_epoch >= self.args.early_stop:     #"early_stop": 10
                plot_all_epochs(w_allmean,con_path, filename=f'all_epochs_summary_seed{seed}.png', expert_names=expert_names)
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False, f=0):
        model.eval()
        y_pred, y_true, y_pred_probabilities = [], [], []
        weight, ability = [], []
        c_fea = []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_pheno": [],
                "Feature_anat": [],
                "Feature_func": [],
                "Feature_corr": [],
                "Feature_f": [],
            }

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    pheno = batch_data['PHENO'].to(self.args.device)
                    anat = batch_data['ANAT'].to(self.args.device)
                    func = batch_data['FUNC'].to(self.args.device)
                    corr = batch_data['CORRELATION'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    output = model(pheno, anat, func, corr)
    
                    if self.args.train_mode=='classification':
                        lbl = labels.view(-1).long()
                        loss = self.criterion(output['logits_c'], lbl)
                        pred_cls = torch.argmax(output['logits_c'], dim=1)
                        pred_probabilities = F.softmax(output['logits_c'], dim=1).cpu().detach()
                        y_pred_probabilities.append(pred_probabilities)
                        y_pred.append(pred_cls.cpu())
                        y_true.append(lbl.cpu())
                    else:
                        labels = labels.view(-1,1)
                        loss = self.criterion(output['logits_c'], labels)
                        y_pred.append(output['logits_c'].cpu())
                        y_true.append(labels.cpu())
                        pred_probabilities = F.softmax(output['logits_c'], dim=1).cpu().detach()
                        y_pred_probabilities.append(pred_probabilities.cpu())
                    eval_loss += loss.item()
        
        eval_loss = eval_loss / len(dataloader)
        pred, true, probabilities  = torch.cat(y_pred), torch.cat(y_true),torch.cat(y_pred_probabilities)
        eval_results = self.metrics(pred, true, probabilities)
        eval_results["Loss"] = round(eval_loss, 4)      
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        
        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels
        return eval_results

def plot_gate_contributions(w_epoch, save_dir=None, filename='contributions.png', expert_names=None):
    if isinstance(w_epoch, torch.Tensor):
        w_epoch = w_epoch.cpu().detach().numpy()
    w_epoch = np.asarray(w_epoch, dtype=float)
    N, E = w_epoch.shape

    if expert_names is None:
        expert_labels = [f'Expert{i}' for i in range(E)]
    else:
        if len(expert_names) != E:
            raise ValueError(f"Length of expert_names ({len(expert_names)}) != number of experts ({E})")
        expert_labels = list(expert_names)
    
    mean_w = w_epoch.mean(axis=0)
    primary_idx = w_epoch.argmax(axis=1)
    primary_counts = np.bincount(primary_idx, minlength=E)
    heatmap_data = w_epoch

    width = min(16, max(8, E * 1.1))                     
    height = min(12, max(6, 4 + 0.02 * N))               
    fig, axes = plt.subplots(2, 2, figsize=(width, height), constrained_layout=True)

    base_fontsize = 10 if E <= 6 else 10 + int((E - 6) / 2)
    label_fontsize = max(8, base_fontsize - 1)
    title_fontsize = base_fontsize + 2
    tick_fontsize = max(8, base_fontsize - 2)

    bar_colors_mean = sns.color_palette('Set2', E)
    bar_colors_primary = sns.color_palette('Paired', E)

    ax = axes[0, 0]
    bars = ax.bar(expert_labels, mean_w, color=bar_colors_mean, width=0.6)  
    ax.set_title('Mean Expert Contributions', fontsize=title_fontsize)
    ax.set_ylabel('Mean Gate Weight', fontsize=label_fontsize)
    ax.set_xticks(range(E))
    ax.set_xticklabels(expert_labels, rotation=35, ha='right', fontsize=tick_fontsize) 
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.grid(False)
    for bar, val in zip(bars, mean_w):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.006, f"{val:.3f}",
                ha='center', va='bottom', fontsize=max(7, tick_fontsize-1))
    sns.heatmap(heatmap_data,
                ax=ax,
                cmap='viridis',
                cbar_kws={'label': 'Gate Weight', 'shrink': 0.6, 'aspect': 15},
                xticklabels=expert_labels)
    ax.set_title(f'Patient-Expert Gates (All {N} Samples)', fontsize=title_fontsize)
    ax.set_ylabel('Sample Index', fontsize=label_fontsize)
    ax.set_xticklabels(expert_labels, rotation=35, ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=max(6, tick_fontsize-2))


    ax = axes[1, 0]
    bars2 = ax.bar(expert_labels, primary_counts, color=bar_colors_primary, width=0.6)
    ax.set_title('Primary Expert Distribution', fontsize=title_fontsize)
    ax.set_ylabel('Number of Samples', fontsize=label_fontsize)
    ax.set_xticks(range(E))
    ax.set_xticklabels(expert_labels, rotation=35, ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.grid(False)

    for bar, cnt in zip(bars2, primary_counts):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max(0.2, 0.02 * N),
                str(int(cnt)), ha='center', va='bottom', fontsize=max(7, tick_fontsize-1))


    ax = axes[1, 1]
    sns.boxplot(data=w_epoch, ax=ax, width=0.5,
                flierprops=dict(marker='o', markersize=3, alpha=0.6))
    ax.set_title('Expert Weight Distribution', fontsize=title_fontsize)
    ax.set_ylabel('Gate Weight', fontsize=label_fontsize)
    ax.set_xticks(range(E))
    ax.set_xticklabels(expert_labels, rotation=35, ha='right', fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.grid(False)

    if save_dir is None:
        save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(str(save_dir), filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    
def plot_all_epochs(all_w, save_dir=None, filename='all_epochs_summary.png', expert_names=None):
    num_epochs = len(all_w)
    if num_epochs == 0:
        raise ValueError("all_w is empty")

    mean_list = []
    primary_list = []
    for w in all_w:
        w_np = w.cpu().detach().numpy() if isinstance(w, torch.Tensor) else np.asarray(w)
        # mean per expert (E,)
        mean_list.append(w_np.mean(axis=0))
        # primary expert counts for this epoch: length E
        primary = np.bincount(w_np.argmax(axis=1), minlength=w_np.shape[1])
        primary_list.append(primary)

    mean_over_epochs = np.stack(mean_list, axis=0)         # (num_epochs, E)
    primary_counts = np.stack(primary_list, axis=1)       # (E, num_epochs)

    epochs = np.arange(1, num_epochs + 1)
    E = mean_over_epochs.shape[1]
    
    if expert_names is None:
        expert_labels = [f'Expert{i}' for i in range(E)]
    else:
        if len(expert_names) != E:
            raise ValueError(f"Length of expert_names ({len(expert_names)}) != number of experts ({E})")
        expert_labels = list(expert_names)

    width = max(10, min(24, num_epochs * 0.5 + 6))   
    height = max(6, min(24, E * 0.6 + 4))            
    fig, axes = plt.subplots(2, 1, figsize=(width, height), gridspec_kw={'height_ratios': [2, 1.6]})

    fig.subplots_adjust(hspace=0.6, right=0.85)  
    base_fs = 10 if E <= 5 else 10 + int((E - 5) / 2)
    tick_fs = max(8, base_fs - 1)
    title_fs = base_fs + 2
    legend_cols = min(3, E)

    ax = axes[0]
    for j in range(E):
        ax.plot(epochs, mean_over_epochs[:, j], marker='o', label=expert_labels[j], linewidth=1.5)
    ax.set_title('Mean Gate Weight Evolution', fontsize=title_fs)
    ax.set_xlabel('Epoch', fontsize=base_fs)
    ax.set_ylabel('Mean Gate Weight', fontsize=base_fs)
    ax.set_xticks(epochs)
    ax.tick_params(axis='both', labelsize=tick_fs)
    
    ax.legend(
        loc='center left',         
        bbox_to_anchor=(1.0, 0.5), 
        ncol=1,                    
        fontsize=max(8, tick_fs-1)
    )

    ax = axes[1]
    sns.heatmap(
        primary_counts,
        ax=ax,
        cmap='YlGnBu',
        cbar_kws={'label': 'Number of Samples', 'shrink': 0.8},
        xticklabels=epochs,
        yticklabels=expert_labels,
        linewidths=0.3,
        linecolor='gray',
        annot=False
    )
    ax.set_title('Primary Expert Distribution Over Epochs', fontsize=title_fs)
    ax.set_xlabel('Epoch', fontsize=base_fs)
    ax.set_ylabel('Expert', fontsize=base_fs)
    ax.tick_params(axis='x', rotation=0, labelsize=tick_fs)
    ax.tick_params(axis='y', rotation=0, labelsize=tick_fs)

    plt.tight_layout()
    if save_dir is None:
        save_dir = '.'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(str(save_dir), filename)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return out_path