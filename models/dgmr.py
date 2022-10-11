import torch
from torch.utils.data import DataLoader
from losses import (
    NowcastingLoss,
    GridCellLoss,
    loss_hinge_disc,
    loss_hinge_gen,
    grid_cell_regularizer,
)
import pytorch_lightning as pl
import torchvision
from dataset import MyDataset, MyIterableDataset
from common import LatentConditioningStack, ContextConditioningStack
from generators import Sampler, Generator
from discriminators import Discriminator
from visualization import Visualization
from visual import Visualize
from pytorch_lightning.trainer.supporters import CombinedLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#from ssim import computeSSIM
from fss import computeFSS


class DGMR(pl.LightningModule):
    """Deep Generative Model of Radar"""

    def __init__(
        self,
        forecast_steps: int = 12, #18
        input_channels: int = 1,
        output_shape: int = 256,
        gen_lr: float = 5e-5,
        disc_lr: float = 2e-4,
        visualize: bool = True,
        pretrained: bool = False,
        conv_type: str = "standard",
        num_samples: int = 6,
        grid_lambda: float = 20.0,
        beta1: float = 0.0,
        beta2: float = 0.999,
        latent_channels: int = 384, #768
        context_channels: int = 192, #384
    ):
        """
        Nowcasting GAN is an attempt to recreate DeepMind's Skillful Nowcasting GAN from https://arxiv.org/abs/2104.00954
        but slightly modified for multiple satellite channels
        Args:
            forecast_steps: Number of steps to predict in the future
            input_channels: Number of input channels per image
            visualize: Whether to visualize output during training
            gen_lr: Learning rate for the generator
            disc_lr: Learning rate for the discriminators, shared for both temporal and spatial discriminator
            conv_type: Type of 2d convolution to use, see satflow/models/utils.py for options
            beta1: Beta1 for Adam optimizer
            beta2: Beta2 for Adam optimizer
            num_samples: Number of samples of the latent space to sample for training/validation
            grid_lambda: Lambda for the grid regularization loss
            output_shape: Shape of the output predictions, generally should be same as the input shape
            latent_channels: Number of channels that the latent space should be reshaped to,
                input dimension into ConvGRU, also affects the number of channels for other linked inputs/outputs
            pretrained:
        """
        super(DGMR, self).__init__()
        self.gen_lr = gen_lr
        self.disc_lr = disc_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.discriminator_loss = NowcastingLoss()
        self.grid_regularizer = GridCellLoss()
        self.grid_lambda = grid_lambda
        self.num_samples = num_samples
        self.visualize = visualize
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.input_channels = input_channels
        self.conditioning_stack = ContextConditioningStack(
            input_channels=input_channels,
            conv_type=conv_type,
            output_channels=self.context_channels,
        )
        self.latent_stack = LatentConditioningStack(
            shape=(self.num_samples * self.input_channels, output_shape // 32, output_shape // 32),
            output_channels=self.latent_channels,
        )
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=self.latent_channels,
            context_channels=self.context_channels,
        )
        self.generator = Generator(self.conditioning_stack, self.latent_stack, self.sampler)
        self.discriminator = Discriminator(input_channels)
        self.visual = Visualization()
        #self.visual = Visualize()
        #self.ssim = computeSSIM()
        self.save_hyperparameters()
        self.global_iteration = 0
        self.test_iterable = 0
        self.valid_loss = torch.nn.MSELoss()
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)
        self.in_radar = np.load("../../../data/CAPPI-256/in_radar.npy")
        self.fs = computeFSS()


    def forward(self, x):
        x = self.generator(x)
        return x

    def training_step(self, batch, batch_idx):
        images, future_images = batch[:,:6], batch[:,6:]
        self.global_iteration += 1
        g_opt, d_opt = self.optimizers()
        ##########################
        # Optimize Discriminator #
        ##########################
        # Two discriminator steps per generator step
        for _ in range(2):
            predictions = self(images)
            # Cat along time dimension [B, C, T, H, W]
            generated_sequence = torch.cat([images, predictions], dim=1)
            real_sequence = torch.cat([images, future_images], dim=1)
            # Cat long batch for the real+generated
            concatenated_inputs = torch.cat([real_sequence, generated_sequence], dim=0)

            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = concatenated_outputs.chunk(2, dim=0)
            discriminator_loss = loss_hinge_disc(score_generated, score_real)
            d_opt.zero_grad()
            self.manual_backward(discriminator_loss)
            d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        predictions = [self(images) for _ in range(6)]
        grid_cell_reg = grid_cell_regularizer(torch.stack(predictions, dim=0), future_images)
        # Concat along time dimension
        generated_sequence = [torch.cat([images, x], dim=1) for x in predictions]
        real_sequence = torch.cat([images, future_images], dim=1)
        # Cat long batch for the real+generated, for each example in the range
        # For each of the 6 examples
        generated_scores = []
        for g_seq in generated_sequence:
            concatenated_inputs = torch.cat([real_sequence, g_seq], dim=0)
            concatenated_outputs = self.discriminator(concatenated_inputs)
            score_real, score_generated = concatenated_outputs.chunk(2, dim=0)
            generated_scores.append(score_generated)
        generator_disc_loss = loss_hinge_gen(torch.cat(generated_scores, dim=0))
        generator_loss = generator_disc_loss + self.grid_lambda * grid_cell_reg
        g_opt.zero_grad()
        self.manual_backward(generator_loss)
        g_opt.step()

        self.log_dict(
            {
                "train/d_loss": discriminator_loss,
                "train/g_loss": generator_loss,
                "train/grid_loss": grid_cell_reg,
            },
            prog_bar=True,
        )

        if batch_idx == 0 and (self.trainer.current_epoch) % 10 == 0:
            generated_images = torch.as_tensor(self(images), dtype=torch.float)
            self.visualize_step(
                future_images[0],
                generated_images[0].to(dtype=float),
                self.trainer.current_epoch
            )

            
    def validation_step(self, batch, batch_idx):
        images, future_images = batch[:,:6], batch[:,6:]
        predictions = self(images)
        p = np.where(predictions.cpu().numpy() < 1.0, 0, 1)
        t = np.where((future_images.cpu().numpy()) < 1.0, 0, 1)
        TP = (np.logical_and(t, p)).sum()
        TN = (np.logical_not(np.logical_or(t, p))).sum()
        FN = (np.logical_and(p, np.logical_not(t))).sum()
        FP = (np.logical_xor(t, p)).sum() - FN
        valid_score = np.where(TP > 0, TP / (TP + FN + FP), 0)
        return valid_score



    def validation_epoch_end(self, outputs):
        csi = np.average(outputs)
        self.log_dict({
            "csi": csi
        })

        
    def test_step(self, batch, batch_idx):
        images, future_images = batch[:,:6], batch[:,6:]
        predictions = self(images)
        true = future_images[:,:,0].cpu().numpy()
        false = predictions[:,:,0].cpu().numpy()
        """
        scores = np.zeros((len(images), 12))
        for b in range(len(images)):
            for t in range(12):
                scores[b][t] = self.ssim.compute(true[b][t], false[b][t])

        scores = np.mean(scores, axis=0)  
        """ 
        losses = []
        pre_score_mm = [1.0, 2.0, 8.0]
        for mm in pre_score_mm:
            loss = self.evaluation(
                    future_images.cpu().numpy(),
                    predictions.cpu().numpy(),
                    mm
            )
            losses.append(loss)
        """ 
        if self.test_iterable%4==0:
            #for b in range(16):
            self.visual.run(
                    future_images[0,:,0].cpu(), 
                    predictions[0,:,0].cpu().to(dtype=float), 
                    'result2/test_' + str(self.test_iterable)
            )

        self.test_iterable += 1  
        """
        return np.array(losses)
    

    def computeIOU(self, label, pred):
        iou_batch = np.zeros((len(label), 12))
        pred = np.where(pred < 0.5, 0,
                        np.where(pred < 2.0, 1,
                            np.where(pred < 4.0, 2,
                                np.where(pred < 8.0, 3, 4))))
        label = np.where(label < 0.5, 0,
                np.where(label < 2.0, 1,
                    np.where(label < 4.0, 2,
                        np.where(label < 8.0, 3, 4))))
        for b in range(len(label)):
            batch_p, batch_t = pred[b,:,0], label[b,:,0]
            for s in range(12):
                p, t = batch_p[s].flatten(), batch_t[s].flatten()
                p, t = p[self.in_radar], t[self.in_radar]
                intersection = np.logical_and(t, p)
                union = np.logical_or(t, p)
                iou_batch[b][s] = np.sum(intersection) / np.sum(union)
        return iou_batch


    def evaluation(self, images, generated, mm):
        loss = np.zeros((7, len(images), 12))
        out_index = np.delete(range(256*256), self.in_radar)
        for idx in range(len(images)):
            pred, label = generated[idx, :, 0], images[idx, :, 0]
            for s in range(12):
                p, t = pred[s].flatten(), label[s].flatten()
                p[out_index] = 0
                t[out_index] = 0
                loss[6][idx][s] = self.fs.fss(p.reshape(256, 256), t.reshape(256, 256), mm, 4)
                p, t = p[self.in_radar], t[self.in_radar]
                mse = np.mean((t - p) ** 2, dtype=np.float64)
                #ranges = t.max() - t.min()
                loss[0][idx][s] = mse
                loss[1][idx][s] = ((p * t).sum()) / ((p ** 2).sum() + (t ** 2).sum() + (10 ** -9))
                _p = np.where(p < mm, 0, 1)
                _t = np.where(t < mm, 0, 1)
                TP = (np.logical_and(_t, _p)).sum() + 1
                TN = (np.logical_not(np.logical_or(_t, _p))).sum()
                FN = (np.logical_and(_p, np.logical_not(_t))).sum()
                FP = (np.logical_xor(_t, _p)).sum() - FN
                loss[2][idx][s] = np.where(TP > 0, TP / (TP + FN + FP), 0)
                loss[3][idx][s] = np.where(TP > 0, TP / (TP + (FP + FN)/2), 0)
                loss[4][idx][s] = np.sum(np.subtract(t, p)**2)/len(p)
                loss[5][idx][s] = 10 * math.log10(96**2/mse)
        return loss

     
    def test_epoch_end(self, outputs):
        #outputs = np.mean(outputs, axis=0)
        #print("score : ", outputs)
        pre_score_mm = [1.0, 2.0, 8.0]
        outputs = np.concatenate(outputs, axis=2)
        evaluation = {}
        for i, output in enumerate(outputs):
            evaluation['mse'] = np.mean(output[0], axis=0)
            evaluation['cor'] = np.mean(output[1], axis=0)
            evaluation['csi'] = np.mean(output[2], axis=0)
            evaluation['f1'] = np.mean(output[3], axis=0)
            evaluation['bs'] = np.mean(output[4], axis=0)
            evaluation['psnr'] = np.mean(output[5], axis=0)
            evaluation['fss'] = np.mean(output[6], axis=0)
            csvpath = './scores-{}.csv'.format(pre_score_mm[i])
            data = {'MSE': evaluation['mse'], 'CORR': evaluation['cor'], 'CSI': evaluation['csi'],
                    'F1': evaluation['f1'], 'BS': evaluation['bs'], 'PSNR': evaluation['psnr'], 'FSS': evaluation['fss']}
            df = pd.DataFrame(data)
            df.to_csv(csvpath, encoding='UTF-8') 


    def configure_optimizers(self):
        b1 = self.beta1
        b2 = self.beta2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.gen_lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.disc_lr, betas=(b1, b2))

        return [opt_g, opt_d], []


    def visualize_step(
        self, y: torch.Tensor, y_hat: torch.Tensor, idx: int) -> None:
        future_images = y[:,0].detach().cpu().numpy()
        generated_images = y_hat[:,0].detach().cpu().numpy()
        label, pred = future_images, generated_images
        self.visual.visualPNG(label, pred, idx)


    def train_dataloader(self):
        year = range(2012, 2020)
        month  = range(6, 9)
        train_set = []
        for y in year:
            for m in month:
                ym = str(y) + '0' + str(m)
                train_set.append(ym)
        dataset = MyIterableDataset(train_set)
        train_loader = DataLoader(dataset, batch_size=16, num_workers=2, worker_init_fn=self.worker_init_fn)
        return train_loader


    def worker_init_fn(self, _):
        worker_info = torch.utils.data.get_worker_info()
        dataset_obj = worker_info.dataset
        worker_id = worker_info.id
        split_size = len(dataset_obj.data_list) // worker_info.num_workers
        dataset_obj.data_list = dataset_obj.data_list[worker_id * split_size: (worker_id + 1) * split_size]


    def val_dataloader(self):
        month = range(6, 9)
        valid_set = []
        for m in month:
            ym = '20200' + str(m)
            valid_set.append(MyDataset(ym=ym))
        concat_dataset = torch.utils.data.ConcatDataset(valid_set)
        valid_loader = DataLoader(concat_dataset, batch_size=64, shuffle=False, num_workers=2)
        return valid_loader


    def test_dataloader(self):
        test_set = MyDataset(ym='2020')
        test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=1)
        return test_loader
