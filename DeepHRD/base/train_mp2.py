# training loop
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import numpy as np
import argparse
import sys
from model_with_res import ResNet_dropout as RNN
from snorkel.classification import cross_entropy_with_probs
import utilsModel as ut
import tqdm
import warnings
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from pytorch_metric_learning import losses
from torch.cuda.amp import autocast
import gc

# To ignore all warnings (they get annoying and not helpful)
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="tensorflow")
# ============== PARSER =====================
parser = argparse.ArgumentParser(description='Multi-Resolution biomarker classifier training script - 2022')
parser.add_argument('--train_lib', type=str, default='',
                    help='Path to the training data structure (metadata .pt file). See README for more details on formatting')
parser.add_argument('--val_lib', type=str, default='',
                    help='Path to the validation data structure. See README for more details on formatting')
parser.add_argument('--output', type=str, default='.',
                    help='Path to the output where the checkpoints and training files are saved')
parser.add_argument('--batch_size', type=int, default=64, help='Mini-batch size.')
parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs.')
parser.add_argument('--validation_interval', default=1, type=int,
                    help='How often to run inference on the validation set.')
parser.add_argument('--k', default=1, type=int,
                    help='The top k tiles based on predicted model probabilities used as representative features of the training classes for each slide.')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint model to restart a given training session or for transfer learning.')
parser.add_argument('--resolution', type=str, default='5x', help='Current magnification resolution')
parser.add_argument('--dropoutRate', default=0.2, type=float,
                    help='Rate of dropout to be used within the fully connected layers.')
parser.add_argument('--weights', default=0.5, type=float, help='Unbalanced positive class weight (for CE loss).')
parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers.')
parser.add_argument('--gpu', default=0, type=int, help='Gpu device selection.')
parser.add_argument('--patience', default=40, type=int, help='Gpu device selection.')

parser.add_argument('--sampling_mode', type=str, default='dampened_combined',
                    choices=['none',
                             'dampened_subtype', 'balanced_subtype',
                             'dampened_target', 'balanced_target',
                             'dampened_combined', 'balanced_combined'],
                    help='Strategy for pre-epoch slide sampling.')

parser.add_argument('--lambda_sup', default=0.3, type=float, help='Weight for the supervised contrastive loss (lambda_reg).')
parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'focal'],
                    help='Loss function to use (ce or focal).')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
parser.add_argument('--focal_alpha', type=float, default=None,
                    help='Alpha parameter (class weight) for Focal Loss. If None, uses --weights argument if not 0.5.')
parser.add_argument('--k_sup', default=10, type=int,
                    help='The top k tiles per slide used specifically for the supervised contrastive loss.')
parser.add_argument('--train_inference_dropout_enabled', action='store_true',
                    help='Enable dropout for the inference step on the training set (top-k selection). Default is False.')
parser.add_argument('--train_inference_transforms_enabled', action='store_true',
                    help='Apply training transforms (flipping, color jitter) during the inference step on the training set. Default is False.')
parser.add_argument('--warmup_epochs', type=int, default=1,
                    help='Number of initial epochs to train on random tiles instead of top-k.')
parser.add_argument('--lambda_reg_mse', type=float, default=1.0,
                    help='Weight for the MSE regression loss.')
# ================ LOSS ======================

class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss function that works with soft probability targets.
    """
    def __init__(self, weight=None, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs)

        if self.weight is not None:
            weight = self.weight.to(logits.device).unsqueeze(0)
            loss = loss * weight

        # Sum over classes. Resulting 'loss' shape is (64)
        loss = loss.sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# for when there is a LARGE class dif
class FocalLossWithProbs(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        focal_weight = (1 - probs) ** self.gamma


        loss = -targets * focal_weight * log_probs

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device).unsqueeze(0)
            loss = alpha * loss # Now (N, 2) * (1, 2) is possible

        loss = loss.sum(dim=1)

        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

class MultiTaskLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', lambda_mse=1.0):
        super(MultiTaskLoss, self).__init__()
        self.classification_criterion = SoftCrossEntropyLoss(weight=weight, reduction=reduction)
        self.regression_criterion = nn.MSELoss(reduction=reduction)
        self.lambda_mse = lambda_mse

    def forward(self, logits, hrd_score_pred, targets):
        """
        targets: should contain soft labels in the first two columns [P_HRP, P_HRD]
                 and the normalized continuous HRD score in the third column.
        """
        soft_targets = targets[:, :2]
        loss_cls = self.classification_criterion(logits, soft_targets)

        # Task 2: Regression (Continuous Score)
        hrd_score_gt = targets[:, 2].unsqueeze(1)
        loss_mse = self.regression_criterion(hrd_score_pred, hrd_score_gt)

        # Combined Loss
        total_loss = loss_cls + (self.lambda_mse * loss_mse)

        return total_loss, loss_cls, loss_mse





# ================= EVAL AND TRAIN METHODS =================================================
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def inference(loader, model, criterion, enable_dropout_flag=False):
    model.eval()
    if len(loader.dataset) == 0:
        print("[WARNING] Inference dataset is empty.")
        return np.array([]), 0.0, np.array([])

    if enable_dropout_flag:
        enable_dropout(model)

    probs_list = []
    feature_list = []

    # Tracking total and individual sub-losses
    running_total_loss = 0.0
    running_cls_loss = 0.0
    running_mse_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[INFERENCE]"):
            input, target_score, soft_label, _ = batch
            input = input.to(device, non_blocking=True)
            target_score = target_score.to(device, non_blocking=True).float()
            soft_label = soft_label.to(device, non_blocking=True).float()

            # Combine into [P_HRP, P_HRD, Score]
            combined_target = torch.cat([soft_label, target_score.unsqueeze(1)], dim=1).float()
            batch_size = input.size(0)
            total_samples += batch_size

            if device == "cuda":
                with torch.cuda.amp.autocast():
                    logits, hrd_score_pred, features = model(input)
                    # Unpack the three values from your MultiTaskLoss
                    loss_total, loss_cls, loss_mse = criterion(logits, hrd_score_pred, combined_target)
            else:
                logits, hrd_score_pred, features = model(input)
                loss_total, loss_cls, loss_mse = criterion(logits, hrd_score_pred, combined_target)

            # Accumulate scalars
            running_total_loss += loss_total.item() * batch_size
            running_cls_loss += loss_cls.item() * batch_size
            running_mse_loss += loss_mse.item() * batch_size

            probs_list.append(torch.nn.functional.softmax(logits, dim=1).detach()[:, 1].cpu())
            feature_list.append(features.detach().cpu())

    # Calculate final means
    mean_total = running_total_loss / total_samples
    mean_cls = running_cls_loss / total_samples
    mean_mse = running_mse_loss / total_samples

    print(f"\n[INFERENCE SUMMARY] Total Loss: {mean_total:.4f} | Cls (CE): {mean_cls:.4f} | Reg (MSE): {mean_mse:.4f}")

    all_probs = torch.cat(probs_list).numpy() if probs_list else np.array([])
    all_features = torch.cat(feature_list).numpy() if feature_list else np.array([])

    return all_probs, mean_total, all_features


def train(run, loader,supcon_loader ,model, criterion, criterion_supcon, optimizer, lambda_reg=0.2, device="cpu"):
    """
    Updated train function with Supervised Contrastive Loss (SupCon)
    that only runs on tiles with "definitive" hard labels (e.g., [0, 1] or [1, 0]).
    """
    model.train()
    running_loss = 0.0
    running_inst_loss = 0.0
    total_samples = 0
    if len(loader.dataset) == 0:
        print(f"[EPOCH {run}] Warning: Training dataset is empty. Skipping train step.")
        return 0.0, 0.0
    for i, (input, target, softLabel, slide_ids) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[TRAINING]"):
        input = input.to(device, non_blocking= True )
        target = target.to(device, non_blocking= True )
        softLabel = softLabel.to(device, non_blocking= True)
        batch_size = input.size(0)
        total_samples += batch_size
        combined_target = torch.cat([softLabel, target.unsqueeze(1)], dim=1).float()

        logits, hrd_score_pred, features = model(input)
        loss, loss_cls, loss_mse = criterion(logits, hrd_score_pred, combined_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        del input, target, softLabel, combined_target # do not remove unless you want computer to crash
        gc.collect()
        torch.cuda.empty_cache()

    if total_samples == 0:
        epoch_loss = 0.0
        epoch_inst_loss = 0.0
    else:
        epoch_loss = running_loss / total_samples
        epoch_inst_loss = running_inst_loss / total_samples

    print(
        f"[EPOCH {run}] Mean train loss: {epoch_loss:.6f} | "
    )
    return epoch_loss, 0.0


# ================= TRAIN LOOP ==================
best_val_loss = np.inf

def main():
    # torch.set_num_threads(1)
    global best_val_loss, device, args
    args = parser.parse_args()
    resolution = args.resolution
    os.makedirs(args.output, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(f"[INFO] no CUDA device available. If this is a mistake please check your setup.")
    else:
        torch.cuda.set_device(args.gpu)
        print(f"[INFO] CUDA is available. Process will be run on {torch.cuda.get_device_name(args.gpu)}")

    print(f"[INFO] Reading metadata from {args.train_lib} to calculate class weights...")
    lib = torch.load(args.train_lib, map_location='cpu')
    if 'targets' not in lib:
        print(
            "[WARN] Loaded .pt file must contain a 'targets' key (for soft labels) for auto-weighting. Falling back to args.")
    else:
        soft_labels_list = lib['softLabels']
        hard_labels = np.array([torch.argmax(t).item() for t in soft_labels_list])
        classes, class_counts = np.unique(hard_labels, return_counts=True)
        if len(classes) != 2:
            print(
                f"[WARNING] Expected 2 classes (0, 1) but found {len(classes)} after argmax. Disabling auto-weighting.")
        else:
            total_samples = class_counts.sum()
            num_classes = len(classes)
            sorted_counts = class_counts[np.argsort(classes)]
            weights = total_samples / (num_classes * sorted_counts)
            class_weights = torch.FloatTensor(weights).to(device, non_blocking= True )
            print(f"Auto-calculated weights from argmax of 'targets' key: {class_weights.cpu().tolist()}")
    model = RNN(args.dropoutRate)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    model.to(device)
    # 4. Initiate criterion, optimizer
    criterion = MultiTaskLoss(weight=class_weights, lambda_mse=args.lambda_reg_mse).to(device)
    criterion_supcon = losses.SupConLoss(temperature=0.07).to(device, non_blocking= True )
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    normalize = transforms.Normalize(mean=[0.485, 0.406, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),

        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),

        transforms.ToTensor(),
        normalize
    ])

    infer_trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    if args.train_inference_transforms_enabled:
        infer_train_transforms = trans
    else:
        infer_train_transforms = infer_trans

    train_dset = ut.MILdataset(args.train_lib, trans)
    if device == "cpu":
        pin_memory = False
    else:
        pin_memory = True

    if args.val_lib:
        val_dset = ut.MILdataset(args.val_lib, infer_trans)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory)
    log_path = os.path.join(args.output, f'training_log_{resolution}.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_inst_loss', 'val_loss', 'val_acc',
            'val_auc', 'val_f1', 'val_err', 'val_fpr', 'val_fnr'
        ])
    early_stop = 0
    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        if early_stop == args.patience:
            print(f"[INFO]: STOPPED AT EPOCH {epoch + 1} AFTER {args.patience} EPOCHS OF NO IMPROVEMENT")
            break

        # i. make sure that inference is evaluated BY the warmup
        train_dset.preselect_epoch_slides(sampling_mode=args.sampling_mode)
        train_dset.modelState(1)
        train_dset.setTransforms(infer_train_transforms)
        infer_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory)
        probs, loss, features = inference(infer_loader,model, criterion, enable_dropout_flag=args.train_inference_dropout_enabled)
        if epoch< args.warmup_epochs:
            train_dset.make_clustered_warmup_data(probs, features)
        else:
            train_dset.maket_data(probs, args.k)

        # ii. start with training based on top instances

        supcon_dset = ut.MILdataset(args.train_lib, trans)
        supcon_dset.epoch_tile_info = train_dset.epoch_tile_info.copy()
        supcon_dset.epoch_slide_id_map = train_dset.epoch_slide_id_map.copy()
        supcon_dset.epoch_target_map = train_dset.epoch_target_map.copy()
        supcon_dset.epoch_subtype_map = train_dset.epoch_subtype_map.copy()
        # supcon_dset.maket_data(probs, args.k_sup)

        train_loader_new = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory)
        train_dset.modelState(2)
        train_dset.setTransforms(trans)

        supcon_loader = torch.utils.data.DataLoader(
            supcon_dset,
            batch_size=args.batch_size*4,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory)

        train_loss, train_inst_loss = train(epoch + 1, train_loader_new,supcon_loader, model,
                                            criterion, criterion_supcon, optimizer, # <-- Pass new criterion
                                            lambda_reg=args.lambda_sup,
                                            device=device)
        scheduler.step()
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_inst_loss': train_inst_loss,
            'val_loss': np.nan,
            'val_acc': np.nan,
            'val_auc': np.nan,
            'val_f1': np.nan,
            'val_err': np.nan,
            'val_fpr': np.nan,
            'val_fnr': np.nan
        }
        # iii. inference on validation
        if (epoch + 1) % args.validation_interval == 0 and args.val_lib:
            val_dset.modelState(1)
            probs, val_loss, features = inference(val_loader, model, criterion, enable_dropout_flag=False)
            maxs = ut.groupTopKtilesAverage(np.array(val_dset.slideIDX), probs, len(val_dset.targets), k = 25)

            true_labels_tensors = val_dset.softLabels
            # true_labels_1d = np.array([torch.argmax(t).item() for t in true_labels_tensors])
            true_labels_1d = np.array([1 if t[1] >= 0.5 else 0 for t in true_labels_tensors])
            pred_binary = np.array([1 if x >= 0.5 else 0 for x in maxs])
            auc = roc_auc_score(true_labels_1d, maxs)
            accuracy = accuracy_score(true_labels_1d, pred_binary)
            f1 = f1_score(true_labels_1d, pred_binary)
            torch.cuda.empty_cache()

            try:
                tn, fp, fn, tp = confusion_matrix(true_labels_1d, pred_binary).ravel()

                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            except ValueError:
                print(f"Warning: Could not calculate confusion matrix at epoch {epoch + 1}. True labels might contain only one class.")
                fpr = np.nan
                fnr = np.nan
            err = (fpr + fnr) / 2. if not (np.isnan(fpr) or np.isnan(fnr)) else np.nan

            log_data['val_loss'] = val_loss
            log_data['val_acc'] = accuracy
            log_data['val_auc'] = auc
            log_data['val_f1'] = f1
            log_data['val_err'] = err
            log_data['val_fpr'] = fpr
            log_data['val_fnr'] = fnr
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop = 0
                print(f"\n  ** New best validation loss: {best_val_loss:.6f} at epoch {epoch + 1}. Saving model. **")
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.output, f'checkpoint_best_{args.resolution}_epoch_{epoch + 1}.pth')
                torch.save(obj, save_path)
            else:
                early_stop += 1
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.output, f'checkpoint_{args.resolution}_epoch_{epoch + 1}.pth')
                torch.save(obj, save_path)

        elif not args.val_lib:
            pass
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                log_data['epoch'],
                f"{log_data['train_loss']:.6f}",
                f"{log_data['train_inst_loss']:.6f}",
                f"{log_data['val_loss']:.6f}" if not np.isnan(log_data['val_loss']) else '',
                f"{log_data['val_acc']:.6f}" if not np.isnan(log_data['val_acc']) else '',
                f"{log_data['val_auc']:.6f}" if not np.isnan(log_data['val_auc']) else '',
                f"{log_data['val_f1']:.6f}" if not np.isnan(log_data['val_auc']) else '',
                f"{log_data['val_err']:.6f}" if not np.isnan(log_data['val_err']) else '',
                f"{log_data['val_fpr']:.6f}" if not np.isnan(log_data['val_fpr']) else '',
                f"{log_data['val_fnr']:.6f}" if not np.isnan(log_data['val_fnr']) else ''
            ])


if __name__ == '__main__':
    main()
