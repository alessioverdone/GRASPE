import os
import torch
import math

from src.dataset.parse_datasets import parse_datasets
from src.layers.grape import Grape


def get_model(hparams):
    if hparams.model == 'grape':
        model = Grape(hparams)
    else:
        raise Exception('Error in select the model!')

    model = model.to(hparams.device)
    return model


def set_params_wrt_dataset(run_params, dataModuleInstance):
    ### Model setting DGM ###
    if run_params.model == 'grape':
        # Hi-patch/grape parameters
        run_params.ndim = dataModuleInstance["input_dim"]
        run_params.npatch = int(math.ceil((run_params.history - run_params.patch_size) / run_params.stride)) + 1
        run_params.patch_layer = layer_of_patches(run_params.npatch)
        run_params.scale_patch_size = run_params.patch_size / (run_params.history + run_params.pred_window)
        run_params.task = 'forecasting'

        # DGM params update
        run_params.dgm_layers[0][0] = run_params.hid_dim
        run_params.conv_layers[0][0] = run_params.hid_dim
        run_params.pre_fc[0] = run_params.hid_dim
        run_params.pre_fc[-1] = run_params.hid_dim

    else:
        raise ValueError('Define model name correctly!')

    return run_params

# Recursive function to determine patch layers
def layer_of_patches(n_patch):
    if n_patch == 1:
        return 1
    if n_patch % 2 == 0:
        return 1 + layer_of_patches(n_patch / 2)
    else:
        return layer_of_patches(n_patch + 1)


def make_dgm_network_parameters_v2(emb_dim):
    pre_fc = [-1, emb_dim]

    dgm_layers = [[emb_dim, int(emb_dim / 2)], [emb_dim, int(emb_dim / 2)], []]
    conv_layers = [[emb_dim, int(emb_dim / 2)], [int(emb_dim / 2), int(emb_dim / 2)],
                   [int(emb_dim / 2), int(emb_dim / 4)]]

    fc_layers = [int(emb_dim / 4), -1]
    return conv_layers, dgm_layers, fc_layers, pre_fc


def initialize_log_parameters(cont: int, combo: dict) -> dict:
    METRICS = ['mse', 'rmse', 'mae']
    SPLITS = ['val', 'test']

    # colonne metriche: val_mse_mean, val_mse_std, ...
    metric_keys = [f'{split}_{metric}_{stat}'
                   for split in SPLITS
                   for metric in METRICS
                   for stat in ('mean', 'std')]

    grid_params = {'Run': cont, **combo, **{k: 0. for k in metric_keys}}

    print(' '.join(f'{k}: {v}' for k, v in grid_params.items()))
    return grid_params


def get_datamodule(run_params):
    # Parse dataset and initialize model
    if run_params.dataset_name in ["physionet", "mimic", "ushcn", "activity"]:
        data_module_instance = parse_datasets(run_params,
                                              run_params.patch_ts)
    else:
        raise ValueError('Define dataset name correct!')

    run_params = set_params_wrt_dataset(run_params, data_module_instance)  #TODO: da adattare al modello

    return data_module_instance, run_params


def update_seed_metrics(model, res_test, val_results, test_results):
    best_val_mse, best_val_rmse, best_val_mae = model.best_mse, model.best_rmse, model.best_mae

    # Testing
    test_mse = res_test[0]['test_mse']
    test_rmse = res_test[0]['test_rmse']
    test_mae = res_test[0]['test_mae']

    val_results.append([best_val_mse, best_val_rmse, best_val_mae])
    test_results.append([test_mse, test_rmse, test_mae])

    print(f'best_val_mse: {best_val_mse}')
    print(f'best_val_rmse: {best_val_rmse}')
    print(f'best_val_mae: {best_val_mae}')
    print(f'test_mse: {test_mse}')
    print(f'test_rmse {test_rmse}')
    print(f'test_mae: {test_mae}')
    return val_results, test_results


def update_run_metrics(val_results,
                       test_results,
                       grid_params_dict,
                       run_params):
    metrics = ['mse', 'rmse', 'mae']
    splits = ['val', 'test']
    results = {'val':  torch.tensor(val_results),
               'test': torch.tensor(test_results)}

    grid_params_dict.update({f'{split}_{metric}_{stat}': float(getattr(torch, stat)(results[split][:, i]))
                            for split in splits
                            for i, metric in enumerate(metrics)
                            for stat in ('mean', 'std')})

    print(' '.join(f'{k}: {v}' for k, v in grid_params_dict.items()))
    output_string = ' '.join([f'{k}: {v}' for k, v in grid_params_dict.items()])

    if run_params.save_logs:
        os.makedirs(run_params.logs_dir, exist_ok=True)
        with open(os.path.join(run_params.logs_dir, 'log.txt'), 'a') as file:
            print(output_string, file=file)
