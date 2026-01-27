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
parser.add_argument('--warmup_epochs', type=int, default=10,
                    help='Number of initial epochs to train on random tiles instead of top-k.')
parser.add_argument('--lambda_reg_mse', type=float, default=1.0,
                    help='Weight for the MSE regression loss.')
# ================ LOSS ======================

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.15): # Lowered to 0.05 for stability
        super(SoftCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # targets: [batch, 2] soft labels
        if self.smoothing > 0:
            # Smooths [0, 1] to [0.025, 0.975]
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing

        log_probs = F.log_softmax(logits, dim=1)

        # Calculate weighted cross entropy for soft targets
        # Formula: -sum(target * log_prob)
        loss = -(targets * log_probs).sum(dim=1)

        if self.weight is not None:
            sample_weights = targets[:, 0] * self.weight[0] + targets[:, 1] * self.weight[1]
            loss = loss * sample_weights

        return loss.mean()
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
class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(self, weight=None, smoothing=0.1):
        super().__init__()
        # Pass smoothing to the classification criterion
        self.classification_criterion = SoftCrossEntropyLoss(weight=weight, smoothing=smoothing)
        self.regression_criterion = nn.MSELoss()

        # Learnable log-variance parameters
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_reg = nn.Parameter(torch.zeros(1))

    def forward(self, logits, hrd_score_pred, targets):
        # targets: [P_HRP, P_HRD, Continuous_Score]
        loss_cls = self.classification_criterion(logits, targets[:, :2])
        loss_reg = self.regression_criterion(hrd_score_pred, targets[:, 2].unsqueeze(1))

        # Precision calculation with clamping for the regression task
        prec_cls = torch.exp(-self.log_var_cls)

        # Limit s_reg to ~7.4 to ensure it doesn't drown out the cls gradients
        log_var_reg_clamped = torch.clamp(self.log_var_reg, min=-2.0)
        prec_reg = torch.exp(-log_var_reg_clamped)

        # Weighted combination
        # The penalty terms use the variables, but we use the clamped version for reg
        total_loss = (prec_cls * loss_cls + self.log_var_cls) + \
                     (prec_reg * loss_reg + log_var_reg_clamped)

        return total_loss, loss_cls, loss_reg


# ================= EVAL AND TRAIN METHODS =================================================
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            m.p = 0.1
def reset_dropout(model, original_rate):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.p = original_rate
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

                    loss_total =criterion(logits, soft_label)
            else:
                logits, hrd_score_pred, features = model(input)
                loss_total = criterion(logits, soft_label)

            # Accumulate scalars
            running_total_loss += loss_total.item() * batch_size
            # running_cls_loss += loss_cls.item() * batch_size
            # running_mse_loss += loss_mse.item() * batch_size

            probs_list.append(torch.nn.functional.softmax(logits, dim=1).detach()[:, 1].cpu())
            feature_list.append(features.detach().cpu())

    # Calculate final means
    mean_total = running_total_loss / total_samples
    # mean_cls = running_cls_loss / total_samples
    # mean_mse = running_mse_loss / total_samples

    # print(f"\n[INFERENCE SUMMARY] Total Loss: {mean_total:.4f} | Cls (CE): {mean_cls:.4f} | Reg (MSE): {mean_mse:.4f}")

    all_probs = torch.cat(probs_list).numpy() if probs_list else np.array([])
    all_features = torch.cat(feature_list).numpy() if feature_list else np.array([])

    return all_probs, mean_total, all_features


def train(run, loader, supcon_loader, model, criterion, criterion_supcon, optimizer, lambda_reg=0.2, device="cpu", freeze_backbone_bn=True):
    model.train()

    # CRITICAL: Freeze BN stats if the backbone is technically frozen
    if freeze_backbone_bn:
        for m in model.resnet.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    running_loss = 0.0
    total_samples = 0
    for i, (input, target, softLabel, slide_ids) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[TRAINING]"):
        input = input.to(device, non_blocking=True)
        softLabel = softLabel.to(device, non_blocking=True)
        batch_size = input.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        logits, _, _ = model(input)
        loss = criterion(logits, softLabel)

        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if torch.isnan(total_norm) or torch.isinf(total_norm):
            print(f"[CRITICAL] Exploding gradients detected at Epoch {run}. Skipping batch.")
            optimizer.zero_grad()
        else:
            optimizer.step()

        running_loss += loss.item() * batch_size

    return running_loss / total_samples, 0.0


# ================= TRAIN LOOP ==================
best_val_loss = np.inf

def set_trainable_layers(model, stage):
    for param in model.parameters():
        param.requires_grad = False

    for param in model.head.parameters():
        param.requires_grad = True

    if stage >= 1:
        for param in model.resnet.layer4.parameters():
            param.requires_grad = True
    if stage >= 2:
        for param in model.resnet.parameters():
            param.requires_grad = True

    return [p for p in model.parameters() if p.requires_grad]

def get_optim_and_sched(model, stage, epoch, E_UNFREEZE_L4):
    """
    Returns an optimizer with a linear warmup for the backbone LR
    to prevent 'shock' during stage transitions.
    """
    # Stage 0: Warmup (Frozen Backbone)
    if stage == 0:
        for param in model.resnet.parameters(): param.requires_grad = False
        params = [{"params": model.head.parameters(), "lr": 1e-4, "weight_decay": 1e-2}]

    # Stage 1: Fine-tuning Layer 4
    elif stage == 1:
        for param in model.resnet.layer4.parameters(): param.requires_grad = True

        # --- LINEAR WARMUP LOGIC ---
        warmup_duration = 5  # Number of epochs to reach full LR
        target_lr = 5e-6
        base_lr = 1e-7

        if epoch < E_UNFREEZE_L4 + warmup_duration:
            # Linear interpolation: current = start + (end - start) * (progress)
            progress = (epoch - E_UNFREEZE_L4) / warmup_duration
            current_backbone_lr = base_lr + (target_lr - base_lr) * progress
        else:
            current_backbone_lr = target_lr

        params = [
            {"params": model.head.parameters(), "lr": 5e-5, "weight_decay": 1e-2},
            {"params": model.resnet.layer4.parameters(), "lr": current_backbone_lr, "weight_decay": 1e-4}
        ]

    # Stage 2: Full Fine-tuning
    else:
        for param in model.resnet.parameters(): param.requires_grad = True
        params = [
            {"params": model.head.parameters(), "lr": 1e-5, "weight_decay": 1e-2},
            {"params": model.resnet.parameters(), "lr": 5e-6, "weight_decay": 1e-5}
        ]

    optimizer = torch.optim.AdamW(params)
    # T_0 matches the stage length to ensure full decay cycles
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
    return optimizer, scheduler

def main():
    # torch.set_num_threads(1)
    global best_val_loss, device, args
    args = parser.parse_args()


    # ======= params for train
    E_EXPLORE = 5
    E_UNCERTAIN = 10
    E_UNFREEZE_L4 = 25
    E_UNFREEZE_ALL = 60
    # E_EXPLORE = 1
    # E_UNCERTAIN = 2
    # E_UNFREEZE_L4 = 3
    # E_UNFREEZE_ALL = 4
    resolution = args.resolution
    os.makedirs(args.output, exist_ok=True)

    # =========== weighing values + device set up
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


    # ================= model set up ===========================
    model = RNN(args.dropoutRate)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    model.to(device)
    criterion = SoftCrossEntropyLoss().to(device)
    criterion_supcon = losses.SupConLoss(temperature=0.07).to(device, non_blocking= True )
    cudnn.benchmark = True


    # ===================== training =============
    best_val_loss = float('inf')
    current_stage = 2
    # get_optim_and_sched
    optimizer, scheduler = get_optim_and_sched(model, current_stage, 0, E_UNFREEZE_L4)

# transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    if args.train_inference_transforms_enabled:
        infer_train_transforms = trans
    else:
        infer_train_transforms = infer_trans
    # ================== datasets =======================
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
            'epoch', 'train_loss', 'train_auc', 'train_acc', 'train_f1', 'val_loss', 'val_acc',
            'val_auc', 'val_f1', 'val_err', 'val_fpr', 'val_fnr', "SAVED"
        ])
    early_stop = 0
    best_auc = 0

    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        new_stage = current_stage
        is_warmup_window = (epoch >= E_UNFREEZE_L4 and epoch < E_UNFREEZE_L4 + 5)
        if early_stop == args.patience:
            print(f"[INFO]: STOPPED AT EPOCH {epoch + 1} AFTER {args.patience} EPOCHS OF NO IMPROVEMENT")
            break
        if epoch < E_UNFREEZE_L4:
            new_stage = 0
            # should_freeze_bn = True
            should_freeze_bn = False
        elif epoch < E_UNFREEZE_ALL:
            new_stage = 1
            # should_freeze_bn = True # Keep BN frozen while initial L4 tuning happens
            should_freeze_bn = False

        else:
                new_stage = 2
                should_freeze_bn = False # Fully unfreeze for final convergence


        # if new_stage != current_stage or is_warmup_window:
        #         current_stage = new_stage
        #         print(f"\n>>> Transitioning to Stage {current_stage}")
        #         optimizer, scheduler = get_optim_and_sched(model, current_stage, epoch, E_UNFREEZE_L4)
        # ==== preselecting slides for this epoch -> based on subtype sampler
        train_dset.preselect_epoch_slides(sampling_mode=args.sampling_mode)
        train_dset.modelState(1)
        train_dset.setTransforms(infer_train_transforms)
        infer_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory)
        # ========= perform inference on training set + detail performance ============
        probs, loss, features = inference(infer_loader,model, criterion, enable_dropout_flag=args.train_inference_dropout_enabled)
        epoch_slide_ids = np.array([x[1] for x in train_dset.epoch_tile_info])
        num_epoch_slides = len(train_dset.epoch_slide_id_map)
        train_slide_preds = ut.groupTopKtilesAverage(
            epoch_slide_ids,
            probs,
            num_epoch_slides,
            percentile=0.05,
            min_k=2,
            max_k=15
        )

        train_true_labels = np.array([
            1 if train_dset.epoch_softlabel_map[i][1] >= 0.5 else 0
            for i in range(num_epoch_slides)
        ])


        train_pred_binary = np.array([1 if x >= 0.5 else 0 for x in train_slide_preds])
        import numpy as np

        train_true_labels = np.asarray(train_true_labels)
        train_slide_preds = np.asarray(train_slide_preds)

        print("NaNs in labels:", np.isnan(train_true_labels).sum())
        print("NaNs in preds :", np.isnan(train_slide_preds).sum())

        train_auc = roc_auc_score(train_true_labels, train_slide_preds)
        train_acc = accuracy_score(train_true_labels, train_pred_binary)
        train_f1 = f1_score(train_true_labels, train_pred_binary)

        # ===== perform warmup ===============
        if epoch < E_UNFREEZE_L4:
            train_dset.make_smart_warmup_data(
                probs,
                epoch,
                explore_thresh=E_EXPLORE,
                uncertain_thresh=E_UNCERTAIN
            )

        else:
            # Standard Top-K MIL
            p_factor = 1.5 if current_stage == 1 else 1.1
            train_dset.maket_data(probs, percentile=0.1, min_k=5, max_k=25, pool_factor=p_factor)
            # train_dset.maket_data(probs, percentile=0.05, min_k=2, max_k=15)
        # ====== test on actual ================

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
        reset_dropout(model, args.dropoutRate)
        supcon_loader = torch.utils.data.DataLoader(
            supcon_dset,
            batch_size=args.batch_size*4,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory)

        train_loss, _ = train(epoch + 1, train_loader_new, None, model, criterion, None, optimizer,
                              device=device, freeze_backbone_bn=should_freeze_bn)
        scheduler.step()
        log_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            "train_auc": train_auc,
            "train_acc": train_acc,
            "train_f1": train_f1,
            'val_loss': np.nan,
            'val_acc': np.nan,
            'val_auc': np.nan,
            'val_f1': np.nan,
            'val_err': np.nan,
            'val_fpr': np.nan,
            'val_fnr': np.nan,
            "SAVED": ""
        }
        # iii. inference on validation
        if (epoch + 1) % args.validation_interval == 0 and args.val_lib:
            val_dset.modelState(1)
            probs, val_loss, features = inference(val_loader, model, criterion, enable_dropout_flag=False)
            maxs = ut.groupTopKtilesAverage(np.array(val_dset.slideIDX), probs, len(val_dset.targets), percentile=0.05, min_k=5, max_k=25)

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
                patience_counter = 0
                print(f"\n  ** New best validation loss: {best_val_loss:.6f} at epoch {epoch + 1}. Saving model. **")
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.output, f'checkpoint_best_{args.resolution}_epoch_{epoch + 1}.pth')
                torch.save(obj, save_path)
                log_data['SAVED'] = "YES"
            elif log_data['val_auc'] > best_auc:
                best_auc = log_data['val_auc']
                early_stop = 0
                patience_counter = 0
                print(f"\n  ** New best validation auc: {log_data['val_auc']} at epoch {epoch + 1}. Saving auc model. **")
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.output, f'checkpoint_best_auc_{args.resolution}_epoch_{epoch + 1}.pth')
                torch.save(obj, save_path)
                log_data['SAVED'] = "AUC"
            else:
                early_stop += 1
                patience_counter +=1
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
                f"{log_data['train_auc']:.6f}",
                f"{log_data['train_acc']:.6f}",
                f"{log_data['train_f1']:.6f}",
                f"{log_data['val_loss']:.6f}" if not np.isnan(log_data['val_loss']) else '',
                f"{log_data['val_acc']:.6f}" if not np.isnan(log_data['val_acc']) else '',
                f"{log_data['val_auc']:.6f}" if not np.isnan(log_data['val_auc']) else '',
                f"{log_data['val_f1']:.6f}" if not np.isnan(log_data['val_auc']) else '',
                f"{log_data['val_err']:.6f}" if not np.isnan(log_data['val_err']) else '',
                f"{log_data['val_fpr']:.6f}" if not np.isnan(log_data['val_fpr']) else '',
                f"{log_data['val_fnr']:.6f}" if not np.isnan(log_data['val_fnr']) else '',
                # f"{log_data['s_cls']:.6f}" if not np.isnan(log_data['s_cls']) else '',
                # f"{log_data['s_reg']:.6f}" if not np.isnan(log_data['s_reg']) else '',
                f"{log_data['SAVED']}" ,



            ])


if __name__ == '__main__':
    main()
