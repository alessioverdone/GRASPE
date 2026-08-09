import torch
from torch import optim, nn
from torch.nn import functional as F

from src.dataset.evaluation import compute_error
from src.utils.utils import get_model

class Training:
    def __init__(self, hparams):
        super(Training, self).__init__()
        self.args = hparams

        # Import model
        self.model = get_model(hparams)
        if hparams.compile_model:
            self.model = torch.compile(self.model)
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f'Parameters: {num_params}')

        # Training params
        self.scheduler = None
        self.avg_accuracy = None
        self.debug = False
        self.optimizer = None
        self.best_mse, self.best_mae, self.best_mape, self.best_rmse = float('inf'), float('inf'), float('inf'), float('inf')

    def forward(self, batch_dict):
        if self.args.model == 'grape':
            return self.model(batch_dict["tp_to_predict"],
                              batch_dict["observed_data"],
                              batch_dict["observed_tp"],
                              batch_dict["observed_mask"])

        else:
            raise ValueError("model must be either 'hi-patch' or 'dgm'")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='min',
                                                              patience=self.args.lr_patience,
                                                              factor=self.args.lr_factor)

    def training_step(self, train_batch):
        self.optimizer.zero_grad()
        res_forward= self.forward(train_batch)

        # Compute losses and optimize DGM
        if self.args.model == 'grape':
            results = self.compute_metrics_and_losses(train_batch,
                                                      res_forward,
                                                     'train')
            results["train_loss"].backward(retain_graph=False)
            if 'train_graph_loss' in results.keys():
                results['train_graph_loss'].backward()

        else:
            raise Exception("model must be either 'hi-patch' or 'dgm'")

        self.optimizer.step()

        # Update lr
        results['learning_rate'] = self.optimizer.param_groups[0]['lr']
        results["train_loss"] = results["train_loss"].detach().cpu().float()
        if 'train_graph_loss' in results.keys():
            results["train_graph_loss"] = results["train_graph_loss"].detach().cpu().float()
        return results

    def compute_metrics_and_losses(self, batch, res_forward, set_):
        results = {}

        mse = compute_error(batch["data_to_predict"],
                            res_forward['pred_y'],
                            mask=batch["mask_predicted_data"],
                            func="MSE",
                            reduce="mean")
        rmse = torch.sqrt(mse)
        loss = mse

        # Use MSE as the loss function
        with torch.no_grad():  # mae non serve per backprop
            mae = compute_error(batch["data_to_predict"],
                                res_forward['pred_y'],
                                mask=batch["mask_predicted_data"],
                                func="MAE",
                                reduce="mean")

        # DGM part
        if 'l_probs' in res_forward.keys():
            logprobs = res_forward['l_probs']
            corr_pred_ = F.mse_loss(res_forward['pred_y'].squeeze(), batch["data_to_predict"], reduction='none').detach()
            corr_pred = torch.sum(corr_pred_, dim=[0,1]).unsqueeze(0)

            if self.avg_accuracy is None:
                self.avg_accuracy = torch.ones_like(corr_pred) * compute_error(batch["data_to_predict"],
                                                                          res_forward['pred_y'],
                                                                          mask=batch["mask_predicted_data"],
                                                                          func="MSE",
                                                                          reduce="mean").detach()

            delta = self.avg_accuracy - corr_pred
            delta = torch.clamp(delta, min=-1, max=1)
            alpha = 0.1
            point_w = delta ** 2 * torch.exp(-alpha * delta)
            logprobs_ = logprobs.exp().mean([0, -1, -2])
            graph_loss = point_w.squeeze(0) * logprobs_
            graph_loss = graph_loss.mean()
            results[f'{set_}_graph_loss'] = graph_loss

            # Update moving average
            self.avg_accuracy = (self.avg_accuracy.to(corr_pred.device) * 0.95 + 0.05 * corr_pred)

        # Store the loss and error metrics
        results[f'{set_}_loss'] = loss
        results[f'{set_}_mse'] = mse.item()
        results[f'{set_}_rmse'] = rmse.item()
        results[f'{set_}_mae'] = mae.item()
        return results

    def validation_step(self, val_batch):
        res_forward = self.forward(val_batch)

        # Compute metrics
        if self.args.model == 'grape':
            results = self.compute_metrics_and_losses(val_batch,
                                                      res_forward,
                                                      'val')
            results["val_loss"] = results["val_loss"].detach().cpu().float()
            if 'val_graph_loss' in results.keys():
                results["val_graph_loss"] = results["val_graph_loss"].detach().cpu().float()

        else:
            raise Exception("model must be either 'hi-patch' or 'dgm'")
        return results

    def test_step(self, test_batch,):
        res_forward = self.forward(test_batch)

        # Compute losses and optimize DGM
        if  self.args.model == 'grape':
            results = self.compute_metrics_and_losses(test_batch,
                                                      res_forward,
                                                      'test')
            results["test_loss"] = results["test_loss"].detach().cpu().float()
            if 'test_graph_loss' in results.keys():
                results["test_graph_loss"] = results["test_graph_loss"].detach().cpu().float()
        else:
            raise Exception("model must be either 'hi-patch' or 'dgm'")
        return results

    def on_validation_epoch_end(self, val_metrics):
        actual_loss = val_metrics['val_mse']
        actual_rmse = val_metrics['val_rmse']
        actual_mae = val_metrics['val_mae']
        # actual_mape = val_metrics['val_mape']
        if actual_loss < self.best_mse:
            self.best_mse = actual_loss
            self.best_rmse = actual_rmse
            self.best_mae = actual_mae
            # self.best_mape = actual_mape