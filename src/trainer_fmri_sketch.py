import math
import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import data
import utils
import wandb

torch.backends.cuda.matmul.allow_tf32 = True


class Trainer_fmri_sketch:
    def __init__(self, args, accelerator, voxel2clip, clip_extractor, device) -> None:
        # train logs path
        self.outdir = os.path.abspath(f'../train_logs/{args.model_name}')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok = True)

        self.args = args
        self.accelerator = accelerator
        self.voxel2clip = voxel2clip
        self.clip_extractor = clip_extractor
        self.device = device
        self.num_devices = max(torch.cuda.device_count(), 1)
        self.epoch_start = 0

        self.prepare_dataloader()
        self.prepare_optimizer()
        self.prepare_scheduler()

    def prepare_dataloader(self):
        ## Load data
        # subj_num_voxels = {1: 15724, 2: 14278, 3: 15226, 4: 13153, 5: 13039, 6: 17907, 7: 12682, 8: 14386}
        # self.args.num_voxels = subj_num_voxels[self.args.subj_test]

        test_path = "{}/webdataset_avg_split/test/subj0{}".format(self.args.data_path, self.args.subj_test)
        test_dl = data.get_dataloader(
            test_path,
            batch_size = self.args.batch_size,
            num_workers = self.args.num_workers,
            seed = self.args.seed,
            is_shuffle = False,
            extensions = ['nsdgeneral.npy', "jpg", "subj"],
            pool_type = self.args.pool_type,
            pool_num = self.args.pool_num,
        )

        return test_dl

    def prepare_optimizer(self, ):
        # Prepare optimizer
        no_decay = ['bias', 'Norm', 'temperature']
        opt_grouped_parameters = [
            {
                'params': [p for n, p in self.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 5e-2
            }, {
                'params': [p for n, p in self.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = torch.optim.AdamW(opt_grouped_parameters, lr = self.args.max_lr)

    def prepare_scheduler(self):
        # prepare lr scheduler
        one_epoch_steps = self.num_batches
        if self.accelerator.state.deepspeed_plugin is not None:  # Multi GPU
            one_epoch_steps = math.ceil(one_epoch_steps / self.num_devices)
        total_steps = self.args.num_epochs * one_epoch_steps
        print("one_epoch_steps_per_gpu:", one_epoch_steps)
        print("total_steps:", total_steps)

        if self.args.lr_scheduler_type == 'linear':
            self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, total_iters = total_steps, last_epoch = -1
            )
        elif self.args.lr_scheduler_type == 'cycle':
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr = self.args.max_lr,
                total_steps = total_steps,
                final_div_factor = 20,
                last_epoch = -1,
                pct_start = 2 / self.args.num_epochs,
            )
        # elif self.args.lr_scheduler_type == 'plateau':
        #     self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         self.optimizer, mode = 'min', factor = 0.5, patience = 3
        #     )

    def prepare_wandb(self, local_rank, args):
        ## Weights and Biases
        if local_rank == 0 and args.wandb_log:  # only use main process for wandb logging
            import wandb
            wandb_run = args.model_name
            wandb_notes = ''

            print(f"Wandb project {args.wandb_project} run {wandb_run}")
            wandb.login(host = 'https://api.wandb.ai')
            wandb_config = vars(args)
            print("wandb_config:\n", wandb_config)
            if args.resume:  # wandb_auto_resume
                if args.resume_id is None:
                    args.resume_id = args.model_name
                print("wandb_id:", args.resume_id)
                wandb.init(
                    id = args.resume_id,
                    project = args.wandb_project,
                    name = wandb_run,
                    config = wandb_config,
                    notes = wandb_notes,
                    resume = "allow",
                )
            else:
                wandb.init(project = args.wandb_project, name = wandb_run, config = wandb_config, notes = wandb_notes, )

    def prepare_multi_gpu(self):
        self.voxel2clip, self.optimizer, self.lr_scheduler, _ = self.accelerator.prepare(
            self.voxel2clip, self.optimizer, self.lr_scheduler, self.train_dls[0]
        )

        for i, dls in enumerate(zip(self.train_dls, self.val_dls)):
            train_dl, val_dl = dls
            self.train_dls[i] = self.accelerator.prepare(train_dl)
            self.val_dls[i] = self.accelerator.prepare(val_dl)

    def input(self, voxel, subj_id):
        return (voxel, subj_id)

    def train(self, local_rank):
        epoch = self.epoch_start
        self.losses, self.val_losses, self.lrs = [], [], []
        self.best_sim = 0
        self.best_epoch = 0

        self.val_voxel0 = self.val_sketch0 = None

        self.soft_loss_temps = utils.cosine_anneal(
            0.004, 0.0075, self.args.num_epochs - int(self.args.mixup_pct * self.args.num_epochs)
        )

        ## Main loop
        print(f"{self.args.model_name} starting with epoch {epoch} / {self.args.num_epochs}")
        progress_bar = tqdm(range(epoch, self.args.num_epochs), disable = (local_rank != 0))

        for epoch in progress_bar:
            self.voxel2clip.train()

            self.sims_sketch = 0.
            self.loss_nce_sketch_sum = 0.
            self.loss_mse_sketch_sum = 0.
            self.loss_mae_sketch_sum = 0.
            self.loss_cos_sketch_sum = 0.
            self.loss_rec_sum = 0.
            self.loss_cyc_sum = 0.
            self.fwd_percent_correct = 0.
            self.bwd_percent_correct = 0.

            self.val_sims_sketch = 0.
            self.val_loss_nce_sketch_sum = 0.
            self.val_loss_mse_sketch_sum = 0.
            self.val_loss_mae_sketch_sum = 0.
            self.val_loss_cos_sketch_sum = 0.
            self.val_loss_rec_sum = 0.
            self.val_loss_cyc_sum = 0.
            self.val_fwd_percent_correct = 0.
            self.val_bwd_percent_correct = 0.

            # wandb logging
            self.train_epoch(epoch)
            self.log_train()

            if epoch % self.args.eval_interval == 0:
                self.eval_epoch(epoch)
                self.log_val()

            progress_dict = {"epoch": epoch, "lr": self.logs["train/lr"], "loss": self.logs["train/loss"], }

            progress_bar.set_postfix(progress_dict)

            # Main process
            if local_rank == 0:
                # Uploading logs to wandb
                if self.args.wandb_log:
                    wandb.log(self.logs)
                # Save model
                if epoch % self.args.ckpt_interval == 0 or epoch == self.args.num_epochs - 1:
                    self.save(epoch)

            # wait for other GPUs to catch up if needed
            self.accelerator.wait_for_everyone()

    def train_epoch(self, epoch):
        # train loop
        for train_i, datas in enumerate(zip(*self.train_dls)):
            self.train_i = train_i
            repeat_index = train_i % 3  # randomly choose the one in the repeated three

            # ensemble data from multiple subjects
            voxel_list, sketch_list, subj_id_list = [], [], []
            for voxel, sketch, subj_id in datas:
                # for voxel, sketch, coco, subj_id in datas:
                voxel_list.append(voxel[:, repeat_index, ...])
                sketch_list.append(sketch)
                subj_id_list.append(subj_id[[0], ...])
            voxel = torch.cat(voxel_list, dim = 0)
            sketch = torch.cat(sketch_list, dim = 0)
            subj_id = torch.cat(subj_id_list, dim = 0)

            print(">>> Epoch{} | Iter{} | voxel: {}".format(epoch, train_i, voxel.shape), flush = True)
            self.train_step(voxel, sketch, subj_id, epoch)

    def train_step(self, voxel, sketch, subj_id, epoch):
        loss = 0.
        self.optimizer.zero_grad()

        _, feature_maps = self.clip_extractor.embed_sketch(sketch)
        feature_maps = [feature_maps[2], feature_maps[3], feature_maps[4]]
        feature_maps = torch.mean(torch.stack(feature_maps), dim = 0)

        results = self.voxel2clip(self.input(voxel, subj_id))

        # sketch clip loss
        clip_sketch_pred = results[0]

        clip_sketch_pred_norm = nn.functional.normalize(clip_sketch_pred.flatten(1), dim = -1)
        feature_maps_norm = nn.functional.normalize(feature_maps.flatten(1), dim = -1)
        # if self.args.clip_mult:
        # if epoch < int(self.args.mixup_pct * self.args.num_epochs):
        #     loss_nce_sketch = utils.mixco_nce(
        #         clip_sketch_pred_norm, feature_maps_norm, temp = .006, perm = perm, betas = betas, select = select
        #     )
        #     loss += self.args.nce_mult * loss_nce_sketch
        # else:
        #     self.epoch_temp = self.soft_loss_temps[epoch - int(self.args.mixup_pct * self.args.num_epochs)]
        #     loss_nce_sketch = utils.soft_clip_loss(clip_sketch_pred_norm, feature_maps_norm, temp = self.epoch_temp)
        #     loss += self.args.clip_mult * loss_nce_sketch
        # self.epoch_temp = self.soft_loss_temps[epoch - int(self.args.mixup_pct * self.args.num_epochs)]
        # loss_nce_sketch = utils.soft_clip_loss(clip_sketch_pred_norm, feature_maps_norm)
        # loss += self.args.clip_mult * loss_nce_sketch

        # utils.check_loss(loss_nce_sketch, "loss_nce_sketch")
        # loss += loss_nce_sketch
        # self.loss_nce_sketch_sum += loss_nce_sketch.item()

        # sketch mse loss
        if self.args.mse_mult:
            loss_mse_sketch = nn.MSELoss()(clip_sketch_pred_norm, feature_maps_norm)
            utils.check_loss(loss_mse_sketch, "loss_mse_sketch")
            loss += self.args.mse_mult * loss_mse_sketch
            self.loss_mse_sketch_sum += loss_mse_sketch.item()

        # sketch mae loss
        if self.args.mae_mult:
            loss_mae_sketch = nn.L1Loss()(clip_sketch_pred_norm, feature_maps_norm)
            utils.check_loss(loss_mae_sketch, "loss_mae_sketch")
            loss += self.args.mae_mult * loss_mae_sketch
            self.loss_mae_sketch_sum += loss_mae_sketch.item()

        # sketch cos_sim loss
        if self.args.cos_mult:
            loss_cos_sketch = 1. - torch.cosine_similarity(clip_sketch_pred_norm, feature_maps_norm).mean()
            loss += self.args.cos_mult * loss_cos_sketch
            self.loss_cos_sketch_sum += loss_cos_sketch.item()

        # brain reconstruction loss
        if self.args.rec_mult:
            voxel_rec = results[2]
            loss_rec = nn.MSELoss()(voxel, voxel_rec)
            utils.check_loss(loss_rec, "loss_rec")
            loss += self.args.rec_mult * loss_rec
            self.loss_rec_sum += loss_rec.item()

        # cycle loss
        if self.args.cyc_mult:
            loss_cyc = results[3]
            utils.check_loss(loss_cyc, "loss_cyc")
            loss += self.args.cyc_mult * loss_cyc
            self.loss_cyc_sum += loss_cyc.item()

        utils.check_loss(loss)
        self.accelerator.backward(loss)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.lr_scheduler.step()

        self.sims_sketch += nn.functional.cosine_similarity(feature_maps_norm, clip_sketch_pred_norm).mean().item()

        # forward and backward top 1 accuracy
        labels = torch.arange(len(feature_maps_norm)).to(self.device)
        self.fwd_percent_correct += utils.topk(
            utils.batchwise_cosine_similarity(clip_sketch_pred_norm, feature_maps_norm), labels, k = 1
        )
        self.bwd_percent_correct += utils.topk(
            utils.batchwise_cosine_similarity(feature_maps_norm, clip_sketch_pred_norm), labels, k = 1
        )

    def log_train(self):
        self.logs = {
            "train/loss": np.mean(self.losses[-(self.train_i + 1):]),
            "train/lr": self.lrs[-1],
            "train/num_steps": len(self.losses),
            "train/cosine_sim_sketch": self.sims_sketch / (self.train_i + 1),
            "train/loss_nce_sketch": self.loss_nce_sketch_sum / (self.train_i + 1),
            "train/loss_mse_sketch": self.loss_mse_sketch_sum / (self.train_i + 1),
            "train/loss_mae_sketch": self.loss_mae_sketch_sum / (self.train_i + 1),
            "train/loss_cos_sketch": self.loss_cos_sketch_sum / (self.train_i + 1),
            "train/loss_rec": self.loss_rec_sum / (self.train_i + 1),
            "train/loss_cyc": self.loss_cyc_sum / (self.train_i + 1),
            "train/fwd_pct_correct": self.fwd_percent_correct / (self.train_i + 1),
            "train/bwd_pct_correct": self.bwd_percent_correct / (self.train_i + 1),
        }
        print(f"train/loss: {np.mean(self.losses[-(self.train_i + 1):])}")
        print(f"train/lr: {self.lrs[-1]}")
        print(f"train/num_steps: {len(self.losses)}")
        print(f"train/cosine_sim_sketch: {self.sims_sketch / (self.train_i + 1)}")
        print(f"train/loss_nce_sketch: {self.loss_nce_sketch_sum / (self.train_i + 1)}")
        print(f"train/loss_mse_sketch: {self.loss_mse_sketch_sum / (self.train_i + 1)}")
        print(f"train/loss_mae_sketch: {self.loss_mae_sketch_sum / (self.train_i + 1)}")
        print(f"train/loss_cos_sketch: {self.loss_cos_sketch_sum / (self.train_i + 1)}")
        print(f"train/loss_rec: {self.loss_rec_sum / (self.train_i + 1)}")
        print(f"train/loss_cyc: {self.loss_cyc_sum / (self.train_i + 1)}")
        print(f"train/fwd_pct_correct: {self.fwd_percent_correct / (self.train_i + 1)}")
        print(f"train/bwd_pct_correct: {self.bwd_percent_correct / (self.train_i + 1)}")

    def log_val(self):
        self.logs.update(
            {
                "val/loss": np.mean(self.val_losses[-(self.val_i + 1):]),
                "val/num_steps": len(self.val_losses),
                "val/cosine_sim_sketch": self.val_sims_sketch / (self.val_i + 1),
                "val/loss_nce_sketch": self.val_loss_nce_sketch_sum / (self.val_i + 1),
                "val/loss_mse_sketch": self.val_loss_mse_sketch_sum / (self.val_i + 1),
                "val/loss_mae_sketch": self.val_loss_mae_sketch_sum / (self.val_i + 1),
                "val/loss_cos_sketch": self.val_loss_cos_sketch_sum / (self.val_i + 1),
                "val/loss_rec": self.val_loss_rec_sum / (self.val_i + 1),
                "val/loss_cyc": self.val_loss_cyc_sum / (self.val_i + 1),
                "val/fwd_pct_correct": self.val_fwd_percent_correct / (self.val_i + 1),
                "val/bwd_pct_correct": self.val_bwd_percent_correct / (self.val_i + 1),
            }
        )
        print(f"val/loss: {np.mean(self.val_losses[-(self.val_i + 1):])}")
        print(f"val/num_steps: {len(self.val_losses)}")
        print(f"val/cosine_sim_sketch: {self.val_sims_sketch / (self.val_i + 1)}")
        print(f"val/loss_nce_sketch: {self.val_loss_nce_sketch_sum / (self.val_i + 1)}")
        print(f"val/loss_mse_sketch: {self.val_loss_mse_sketch_sum / (self.val_i + 1)}")
        print(f"val/loss_mae_sketch: {self.val_loss_mae_sketch_sum / (self.val_i + 1)}")
        print(f"val/loss_cos_sketch: {self.val_loss_cos_sketch_sum / (self.val_i + 1)}")
        print(f"val/loss_rec: {self.val_loss_rec_sum / (self.val_i + 1)}")
        print(f"val/loss_cyc: {self.val_loss_cyc_sum / (self.val_i + 1)}")
        print(f"val/fwd_pct_correct: {self.val_fwd_percent_correct / (self.val_i + 1)}")
        print(f"val/bwd_pct_correct: {self.val_bwd_percent_correct / (self.val_i + 1)}")
