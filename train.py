from run import HPMD_run

HPMD_run(model_name='hpmd', dataset_name='abide', is_tune=False, seeds=[41], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train')