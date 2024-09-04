import math
import os
import random
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

import data
import utils
import wandb

torch.backends.cuda.matmul.allow_tf32 = True


class Trainer_fmri_vae:
    def __init__(self, args, accelerator, voxel2vae, vae_extractor, device) -> None:
        # train logs path
        self.outdir = os.path.abspath(f'../train_logs/{args.model_name}')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok = True)

        self.args = args
        self.accelerator = accelerator
        self.voxel2vae = voxel2vae
        self.vae_extractor = vae_extractor
        self.device = device
        self.num_devices = max(torch.cuda.device_count(), 1)
        self.epoch_start = 0

        self.prepare_dataloader()
        self.prepare_optimizer()
        self.prepare_scheduler()
        # self.prepare_multi_gpu()

    @abstractmethod
    def prepare_dataloader(self):
        pass

    def print_cuda_memory_usage(self):
        t = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        r = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        a = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        f = r - a  # free inside reserved
        print(f"Total memory: {t:.2f} GB")
        print(f"Reserved memory: {r:.2f} GB")
        print(f"Allocated memory: {a:.2f} GB")
        print(f"Free inside reserved memory: {f:.2f} GB")

    def prepare_optimizer(self, ):
        # Prepare optimizer
        no_decay = ['bias', 'Norm', 'temperature']
        opt_grouped_parameters = [
            {
                'params': [p for n, p in self.voxel2vae.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 1e-2
            }, {
                'params': [p for n, p in self.voxel2vae.named_parameters() if any(nd in n for nd in no_decay)],
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
                final_div_factor = 100,
                last_epoch = -1,
                pct_start = 2 / self.args.num_epochs,
            )

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

    @abstractmethod
    def prepare_multi_gpu(self):
        pass

    def input(self, voxel, subj_id):
        return (voxel, subj_id)

    def train(self, local_rank):
        epoch = self.epoch_start
        self.losses, self.val_losses, self.lrs = [], [], []
        self.best_sim = 0
        self.best_epoch = 0

        self.val_voxel0 = self.val_image0 = None

        self.soft_loss_temps = utils.cosine_anneal(
            0.004, 0.0075, self.args.num_epochs - int(self.args.mixup_pct * self.args.num_epochs)
        )

        ## Main loop
        print(f"{self.args.model_name} starting with epoch {epoch} / {self.args.num_epochs}")
        progress_bar = tqdm(range(epoch, self.args.num_epochs), disable = (local_rank != 0))

        for epoch in progress_bar:
            self.voxel2vae.train()

            self.sims_embedding = 0.
            self.sims_image = 0.
            self.loss_mse_embedding_sum = 0.
            self.loss_mae_image_sum = 0.
            self.loss_rec_sum = 0.
            self.loss_cyc_sum = 0.
            self.loss_ms_ssim_image_sum = 0.

            self.val_sims_embedding = 0.
            self.val_sims_image = 0.
            self.val_loss_mse_embedding_sum = 0.
            self.val_loss_mae_image_sum = 0.
            self.val_loss_rec_sum = 0.
            self.val_loss_cyc_sum = 0.
            self.val_loss_ms_ssim_image_sum = 0.

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

    @abstractmethod
    def train_epoch(self, epoch):
        pass

    # def train_step(self, voxel, image, subj_id, epoch):
    def train_step(self, voxel, subj_id, embedding, rec_img, epoch):
        """
            embedding: [batch_size*n_subj, 4, 64, 64]
            rec_img:   [batch_size*n_subj, 3, 512, 512]
        """
        loss = 0.
        self.optimizer.zero_grad()

        vae_embedding = embedding

        results = self.voxel2vae(self.input(voxel, subj_id))

        vae_embedding_pred = results[0].half()
        vae_image_pred = self.vae_extractor.decode_image(vae_embedding_pred)

        loss_mse_embedding = F.mse_loss(vae_embedding_pred, vae_embedding)
        loss_mae_image = F.l1_loss(rec_img, vae_image_pred)
        loss_ms_ssim_image = utils.ms_ssim_loss(vae_image_pred, rec_img)

        loss_rec = F.mse_loss(voxel, results[1])

        loss += self.args.ssim_mult * loss_ms_ssim_image
        loss += self.args.mae_mult * loss_mae_image
        loss += self.args.mse_mult * loss_mse_embedding
        loss += self.args.rec_mult * loss_rec
        loss += self.args.cyc_mult * results[2]

        utils.check_loss(loss)

        self.loss_mse_embedding_sum += loss_mse_embedding.item()
        self.loss_mae_image_sum += loss_mae_image.item()
        self.loss_rec_sum += loss_rec.item()
        self.loss_cyc_sum += results[2].item()
        self.loss_ms_ssim_image_sum += loss_ms_ssim_image.item()

        self.accelerator.backward(loss)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.lrs.append(self.optimizer.param_groups[0]['lr'])
        self.lr_scheduler.step()

        self.sims_embedding += nn.functional.cosine_similarity(vae_embedding, vae_embedding_pred).mean().item()

        self.sims_image += utils.ms_ssim_val(rec_img, vae_image_pred).item()

    @abstractmethod
    def eval_epoch(self, epoch):
        pass

    def eval_step(self, voxel, subj_id, embedding, rec_img, epoch):
        val_loss = 0.
        with torch.no_grad():
            vae_embedding = embedding

            results = self.voxel2vae(self.input(voxel, subj_id))

            vae_embedding_pred = results[0].half()
            vae_image_pred = self.vae_extractor.decode_image(vae_embedding_pred)

            val_loss_mse_embedding = F.mse_loss(vae_embedding_pred, vae_embedding)
            val_loss_mae_image = F.l1_loss(rec_img, vae_image_pred)
            val_loss_ms_ssim_image = utils.ms_ssim_loss(vae_image_pred, rec_img)

            val_loss_rec = F.mse_loss(voxel, results[1])

            val_loss += self.args.ssim_mult * val_loss_ms_ssim_image
            val_loss += self.args.mae_mult * val_loss_mae_image
            val_loss += self.args.mse_mult * val_loss_mse_embedding
            val_loss += self.args.rec_mult * val_loss_rec
            val_loss += self.args.cyc_mult * results[2]

            utils.check_loss(val_loss)

            if epoch >= 10:
                select_img_idx = random.sample(range(vae_image_pred.shape[0]), 20)
                for j, idx in enumerate(select_img_idx):
                    img = vae_image_pred[idx].to('cpu')
                    image = transforms.functional.to_pil_image(img)
                    image.save(f"{j}_th.png")

            self.val_loss_mse_embedding_sum += val_loss_mse_embedding.item()
            self.val_loss_mae_image_sum += val_loss_mae_image.item()
            self.val_loss_rec_sum += val_loss_rec.item()
            self.val_loss_cyc_sum += results[2].item()
            self.val_loss_ms_ssim_image_sum += val_loss_ms_ssim_image.item()

            self.val_losses.append(val_loss.item())
            self.val_sims_embedding += nn.functional.cosine_similarity(vae_embedding, vae_embedding_pred).mean().item()
            self.val_sims_image += utils.ms_ssim_val(rec_img, vae_image_pred).item()

    def save_ckpt(self, tag, epoch):
        if epoch >= 10:
            ckpt_path = self.outdir + f"/{tag}.pth"
            print(f"saving {ckpt_path}", flush = True)
            unwrapped_model = self.accelerator.unwrap_model(self.voxel2vae)
            try:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "lr_scheduler": self.lr_scheduler.state_dict(),
                        "train_losses": self.losses,
                        "val_losses": self.val_losses,
                        "lrs": self.lrs,
                    }, ckpt_path,
                )
            except:
                print("Couldn't save... moving on to prevent crashing.")
            del unwrapped_model

    def save(self, epoch):
        self.save_ckpt(f'last', epoch)
        # save best model
        current_sim = (self.val_sims_image) / (self.val_i + 1)
        if current_sim > self.best_sim:
            self.best_sim = current_sim
            self.best_epoch = epoch
            self.save_ckpt(f'best', epoch)
        else:
            print(
                f'Not best - current_similarity: {current_sim:.3f} @ epoch {epoch}, best_similarity: {self.best_sim:.3f} @ epoch {self.best_epoch}'
            )

    def load(self, ):
        print("\n---load from ckpt: {}---\n".format(self.args.load_from))
        checkpoint = torch.load(self.args.load_from, map_location = 'cpu')
        self.voxel2vae.load_state_dict(checkpoint['model_state_dict'], strict = False)
        print("loaded keys", checkpoint['model_state_dict'].keys())
        del checkpoint

    def resume(self, ):
        print("\n---resuming from last.pth ckpt---\n")
        checkpoint = torch.load(self.outdir + '/last.pth', map_location = 'cpu')
        self.epoch_start = checkpoint['epoch']
        print("Resume at Epoch", self.epoch_start)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        self.voxel2vae.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint

    def log_train(self):
        self.logs = {
            "train/loss": np.mean(self.losses[-(self.train_i + 1):]),
            "train/lr": self.lrs[-1],
            "train/num_steps": len(self.losses),
            "train/cosine_sim_embedding": self.sims_embedding / (self.train_i + 1),
            "train/ms_ssim_image": self.sims_image / (self.train_i + 1),
            "train/loss_mae_image": self.loss_mae_image_sum / (self.train_i + 1),
            "train/loss_rec": self.loss_rec_sum / (self.train_i + 1),
            "train/loss_cyc": self.loss_cyc_sum / (self.train_i + 1),
            "train/loss_ms_ssim_image": self.loss_ms_ssim_image_sum / (self.train_i + 1),
            "train/loss_mse_embedding": self.loss_mse_embedding_sum / (self.train_i + 1),
        }
        print(f"train/loss: {np.mean(self.losses[-(self.train_i + 1):])}")
        print(f"train/lr: {self.lrs[-1]}")
        print(f"train/num_steps: {len(self.losses)}")
        print(f"train/cosine_sim_embedding: {self.sims_embedding / (self.train_i + 1)}")
        print(f"train/ms_ssim_image: {self.sims_image / (self.train_i + 1)}")
        print(f"train/loss_mae_image: {self.loss_mae_image_sum / (self.train_i + 1)}")
        print(f"train/loss_rec: {self.loss_rec_sum / (self.train_i + 1)}")
        print(f"train/loss_cyc: {self.loss_cyc_sum / (self.train_i + 1)}")
        print(f"train/loss_ms_ssim_image: {self.loss_ms_ssim_image_sum / (self.train_i + 1)}")
        print(f"train/loss_mse_embedding: {self.loss_mse_embedding_sum / (self.train_i + 1)}")

    def log_val(self):
        self.logs.update(
            {
                "val/loss": np.mean(self.val_losses[-(self.val_i + 1):]),
                "val/num_steps": len(self.val_losses),
                "val/cosine_sim_embedding": self.val_sims_embedding / (self.val_i + 1),
                "val/ms_ssim_image": self.val_sims_image / (self.val_i + 1),
                "val/loss_mae_image": self.val_loss_mae_image_sum / (self.val_i + 1),
                "val/loss_rec": self.val_loss_rec_sum / (self.val_i + 1),
                "val/loss_cyc": self.val_loss_cyc_sum / (self.val_i + 1),
                "val/loss_ms_ssim_image": self.val_loss_ms_ssim_image_sum / (self.val_i + 1),
                "val/loss_mse_embedding": self.val_loss_mse_embedding_sum / (self.val_i + 1),
            }
        )
        print(f"val/loss: {np.mean(self.val_losses[-(self.val_i + 1):])}")
        print(f"val/num_steps: {len(self.val_losses)}")
        print(f"val/cosine_sim_embedding: {self.val_sims_embedding / (self.val_i + 1)}")
        print(f"val/ms_ssim_image: {self.val_sims_image / (self.val_i + 1)}")
        print(f"val/loss_mae_image: {self.val_loss_mae_image_sum / (self.val_i + 1)}")
        print(f"val/loss_rec: {self.val_loss_rec_sum / (self.val_i + 1)}")
        print(f"val/loss_cyc: {self.val_loss_cyc_sum / (self.val_i + 1)}")
        print(f"val/loss_ms_ssim_image: {self.val_loss_ms_ssim_image_sum / (self.val_i + 1)}")
        print(f"val/loss_mse_embedding: {self.val_loss_mse_embedding_sum / (self.val_i + 1)}")


class Trainer_fmri_vae_single(Trainer_fmri_vae):
    def __init__(self, args, accelerator, voxel2vae, vae_extractor, device) -> None:
        super().__init__(args, accelerator, voxel2vae, vae_extractor, device)

    def prepare_dataloader(self):
        # Prepare data and dataloader
        print("Preparing data and dataloader...")
        self.train_dl, self.val_dl = data.get_dls(
            subject = self.args.subj_list[0],
            data_path = self.args.data_path,
            batch_size = self.args.batch_size,
            val_batch_size = self.args.val_batch_size,
            num_workers = self.args.num_workers,
            pool_type = self.args.pool_type,
            pool_num = self.args.pool_num,
            length = self.args.length,
            seed = self.args.seed,
        )
        self.num_batches = len(self.train_dl)

    def prepare_multi_gpu(self):
        self.voxel2vae, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler = self.accelerator.prepare(
            self.voxel2vae, self.optimizer, self.train_dl, self.val_dl, self.lr_scheduler
        )

    def input(self, voxel, subj_id):
        # adapting need to know subj_id
        return voxel

    def train_epoch(self, epoch):
        # train loop
        for train_i, data_i in enumerate(self.train_dl):
            self.train_i = train_i
            repeat_index = train_i % 3  # randomly choose the one in the repeated three

            voxel, image, _, subj_id = data_i
            voxel = voxel[:, repeat_index, ...].float()
            subj_id = subj_id[[0], ...]

            print(">>> Epoch{} | Iter{} | voxel: {}".format(epoch, train_i, voxel.shape), flush = True)
            self.train_step(voxel, image, subj_id, epoch)

    def eval_epoch(self, epoch):
        print("evaluating...")
        self.voxel2vae.eval()

        for val_i, data_i in enumerate(self.val_dl):
            self.val_i = val_i
            repeat_index = val_i % 3  # randomly choose the one in the repeated three
            voxel, image, _, subj_id = data_i
            voxel = voxel[:, repeat_index, ...].float()
            # voxel = torch.mean(voxel, axis = 1)
            subj_id = subj_id[[0], ...]

            print(">>> Epoch{} | Eval{} | voxel: {}".format(epoch, val_i, voxel.shape), flush = True)
            self.eval_step(voxel, image, subj_id, epoch)


class Trainer_fmri_vae_bridge(Trainer_fmri_vae):
    def __init__(self, args, accelerator, voxel2vae, vae_extractor, device) -> None:
        super().__init__(args, accelerator, voxel2vae, vae_extractor, device)

    def prepare_dataloader(self):
        # Prepare data and dataloader
        print("Preparing data and dataloader...")
        self.train_dls = []  # tarin_dls contains all subjects separately
        self.val_dls = []  # tarin_dls contains all subjects separately

        for subj in self.args.subj_list:
            train_dl, val_dl = data.get_dls(
                subject = subj,
                data_path = self.args.data_path,
                batch_size = self.args.batch_size,
                val_batch_size = self.args.val_batch_size,
                num_workers = self.args.num_workers,
                pool_type = self.args.pool_type,
                pool_num = self.args.pool_num,
                length = self.args.length,
                seed = self.args.seed,
            )
            self.train_dls.append(train_dl)
            self.val_dls.append(val_dl)

        self.num_batches = len(self.train_dls[0])

    def prepare_multi_gpu(self):
        self.voxel2vae, self.optimizer, self.lr_scheduler, _ = self.accelerator.prepare(
            self.voxel2vae, self.optimizer, self.lr_scheduler, self.train_dls[0]
        )

        for i, dls in enumerate(zip(self.train_dls, self.val_dls)):
            train_dl, val_dl = dls
            self.train_dls[i] = self.accelerator.prepare(train_dl)
            self.val_dls[i] = self.accelerator.prepare(val_dl)

    def train_epoch(self, epoch):
        # train loop
        for train_i, datas in enumerate(zip(*self.train_dls)):
            self.train_i = train_i
            repeat_index = train_i % 3  # randomly choose the one in the repeated three

            # ensemble data from multiple subjects
            voxel_list, subj_id_list, embedding_list, rec_img_list = [], [], [], []
            # voxel_list, image_list, subj_id_list, embedding_list, rec_img_list = [], [], [], [], []
            # for voxel, image, _, subj_id in datas:
            for voxel, subj_id, embedding, rec_img in datas:
                # for voxel, image, _, subj_id, embedding, rec_img in datas:
                voxel_list.append(voxel[:, repeat_index, ...])
                # image_list.append(image)
                subj_id_list.append(subj_id[[0], ...])
                embedding_list.append(embedding)
                rec_img_list.append(rec_img)

            voxel = torch.cat(voxel_list, dim = 0)
            # image = torch.cat(image_list, dim = 0)
            subj_id = torch.cat(subj_id_list, dim = 0)
            embedding = torch.cat(embedding_list, dim = 0)
            rec_img = torch.cat(rec_img_list, dim = 0)

            print(">>> Epoch{} | Iter{} | voxel: {}".format(epoch, train_i, voxel.shape), flush = True)
            self.train_step(voxel, subj_id, embedding, rec_img, epoch)
            # self.train_step(voxel, image, subj_id, embedding, rec_img, epoch)

    def eval_epoch(self, epoch):
        print("Evaluating...")
        self.voxel2vae.eval()
        for val_i, datas in enumerate(zip(*self.val_dls)):
            self.val_i = val_i
            repeat_index = val_i % 3  # randomly choose the one in the repeated three

            # ensemble data from multiple subjects
            # voxel_list, image_list, subj_id_list = [], [], []
            # voxel_list, image_list, subj_id_list, embedding_list, rec_img_list = [], [], [], [], []
            voxel_list, subj_id_list, embedding_list, rec_img_list = [], [], [], []
            # for voxel, image, _, subj_id in datas:
            for voxel, subj_id, embedding, rec_img in datas:
                # voxel_list.append(torch.mean(voxel, axis = 1))
                # voxel = voxel[:, repeat_index, ...].float()
                voxel_list.append(voxel[:, repeat_index, ...])
                # image_list.append(image)
                subj_id_list.append(subj_id[[0], ...])
                embedding_list.append(embedding)
                rec_img_list.append(rec_img)

            voxel = torch.cat(voxel_list, dim = 0)
            # image = torch.cat(image_list, dim = 0)
            subj_id = torch.cat(subj_id_list, dim = 0)
            embedding = torch.cat(embedding_list, dim = 0)
            rec_img = torch.cat(rec_img_list, dim = 0)

            print(">>> Epoch{} | Eval{} | voxel: {}".format(epoch, val_i, voxel.shape), flush = True)
            self.eval_step(voxel, subj_id, embedding, rec_img, epoch)
            # self.eval_step(voxel, image, subj_id, embedding, rec_img, epoch)
