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
import tqdm
import warnings
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from pytorch_metric_learning import losses
from torch.cuda.amp import autocast, GradScaler

# Internal imports
from model import ResNet_dropout as RNN
import utilsModel as ut

warnings.filterwarnings("ignore")

# ============== PARSER =====================
parser = argparse.ArgumentParser(description='Multi-Resolution Biomarker Classifier - 2026 Updated')
parser.add_argument('--train_lib', type=str, required=True)
parser.add_argument('--val_lib', type=str, default='')
parser.add_argument('--output', type=str, default='.')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--validation_interval', default=1, type=int)
parser.add_argument('--k', default=1, type=int)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--resolution', type=str, default='5x')
parser.add_argument('--dropoutRate', default=0.2, type=float)
parser.add_argument('--weights', default=0.5, type=float)
parser.add_argument('--workers', default=16, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--patience', default=40, type=int)
parser.add_argument('--sampling_mode', type=str, default='dampened_combined')
parser.add_argument('--lambda_sup', default=0.3, type=float)
parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'focal'])
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--wd', type=float, default=1e-2)
parser.add_argument('--start_unfrozen', action='store_true')
parser.add_argument('--unfreeze_epoch', type=int, default=5, help='Interval for sequential unfreezing')
parser.add_argument('--train_inference_transforms_enabled', action='store_true')
parser.add_argument('--train_inference_dropout_enabled', action='store_true')
parser.add_argument('--focal_alpha', type=float, default=None)
parser.add_argument('--focal_gamma', type=float, default=2.0)

# ========= unfreezing
def get_optimizer_groups(model, lr, wd):
    """Groups params with /10 rule for early layers."""
    early_backbone = []
    late_backbone = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if any(x in name for x in ['resnet.conv1', 'resnet.bn1', 'resnet.layer1', 'resnet.layer2', 'resnet.layer3']):
            early_backbone.append(param)
        elif 'resnet.layer4' in name:
            late_backbone.append(param)
        else:
            head_params.append(param)
    return [
        {'params': early_backbone, 'lr': lr / 10, 'weight_decay': wd / 10},
        {'params': late_backbone, 'lr': lr, 'weight_decay': wd},
        {'params': head_params, 'lr': lr, 'weight_decay': wd}
    ]

def set_unfreezing_stage(model, stage=0):
    """Stage 0: Heads | 1: Layer 4 | 2: Layer 3 | 3: Full Backbone"""
    for param in model.parameters():
        param.requires_grad = False
    
    # Always unfreeze Heads (Classifier + Projection)
    for param in model.resnet.fc.parameters(): param.requires_grad = True
    for param in model.projection_head.parameters(): param.requires_grad = True

    if stage >= 1:
        for param in model.resnet.layer4.parameters(): param.requires_grad = True
    if stage >= 2:
        for param in model.resnet.layer3.parameters(): param.requires_grad = True
    if stage >= 3:
        for param in model.resnet.parameters(): param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[UNFREEZE] Entering Stage {stage}. Trainable params: {trainable:,}")
# ================ LOSS ======================
class SoftCrossEntropyLoss(nn.Module):
 
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.register_buffer('weight', weight)
        self.reduction = reduction

    def forward(self, input, target):
        num_points, num_classes = input.shape
        cum_losses = input.new_zeros(num_points)
        for y in range(num_classes):
          target_temp = input.new_full((num_points,), y, dtype=torch.long)
          y_loss = F.cross_entropy(input, target_temp, reduction="none")
          if self.weight is not None:
            y_loss = y_loss * self.weight[y]
          cum_losses += target[:, y].float() * y_loss
        return cum_losses.mean()
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


# ================= EVAL AND TRAIN METHODS =================================================
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def inference(loader, model, criterion, enable_dropout_flag = False):
    model.eval()
    if len(loader.dataset) == 0:
        print("[WARNING] Warning: Inference dataset is empty.")
        return np.array([]), 0.0
    if enable_dropout_flag:
        print("[INFO] inference is running with ENABLED dropout for top-k selection")
        enable_dropout(model)

    probs_list = []
    running_loss = 0.0
    total_samples = 0
    slide_outputs = {}

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[INFERENCE]"):
            input, target, slide_ids = batch
            input = input.to(device)
            target = target.to(device)
            batch_size = input.size(0)
            total_samples += batch_size
            if device == "cuda":
                with autocast(): #run with autocast bc we dont care as much about FP16 v FP32 for simple prediction
                    logits,_, _ = model(input)
                    output = F.softmax(logits, dim=1)
                    loss = criterion(logits, target)
            else:
                logits,_,_ = model(input)
                output = F.softmax(logits, dim=1)
                loss = criterion(logits, target)

            running_loss += loss.item() * batch_size
            probs_list.append(output.detach()[:, 1].clone().cpu())

            for sid, t, p in zip(slide_ids.cpu().numpy(), target.cpu(), output[:, 1].cpu()):
                sid_key = int(sid)
                if sid_key not in slide_outputs:
                    slide_outputs[sid_key] = {"probs": [], "target": t}
                slide_outputs[sid_key]["probs"].append(p.item())

    if total_samples == 0:
        mean_loss = 0.0
    else:
        mean_loss = running_loss / total_samples
    if not probs_list:
        all_probs = np.array([])
    else:
        all_probs = torch.cat(probs_list, dim=0).numpy()

    # probs for all tiles + the mean loss
    return all_probs, mean_loss

def train_epoch(run, loader, model, criterion, criterion_supcon,
                optimizer, lambda_reg, device, scaler):

    model.train()
    running_loss, running_inst, total_samples = 0.0, 0.0, 0
    all_targets, all_preds = [], []

    supcon_buffer_feats = []
    supcon_buffer_labels = []

    def flush_supcon_buffer():
        """Runs SupCon on accumulated features and clears buffer."""
        nonlocal running_inst

        if lambda_reg <= 0 or len(supcon_buffer_feats) == 0:
            return torch.tensor(0.0, device=device)

        b_feats = torch.cat(supcon_buffer_feats)
        b_labels = torch.cat(supcon_buffer_labels)

        # Need at least 2 classes
        if len(torch.unique(b_labels)) < 2:
            supcon_buffer_feats.clear()
            supcon_buffer_labels.clear()
            return torch.tensor(0.0, device=device)

        inst_loss = criterion_supcon(b_feats, b_labels)

        supcon_buffer_feats.clear()
        supcon_buffer_labels.clear()

        return inst_loss

    for i, (input, target, _) in tqdm.tqdm(
        enumerate(loader),
        total=len(loader),
        desc=f"TRAIN EP {run}"
    ):

        input, target = input.to(device), target.to(device)

        with autocast():
            logits, _, proj_feats = model(input)

            # Normalize projection features (important for SupCon)
            proj_feats = torch.nn.functional.normalize(proj_feats, dim=1)

            loss_cls = criterion(logits, target)

            # ---- Collect definitive samples ----
            definitive_mask = (target[:, 0] < 0.3) | (target[:, 0] > 0.7)

            if definitive_mask.any():
                supcon_buffer_feats.append(proj_feats[definitive_mask])
                supcon_buffer_labels.append(
                    torch.argmax(target[definitive_mask], dim=1)
                )

            inst_loss = torch.tensor(0.0, device=device)

            if (i + 1) % 8 == 0:
                inst_loss = flush_supcon_buffer()

            loss = (1 - lambda_reg) * loss_cls + lambda_reg * inst_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * input.size(0)
        running_inst += inst_loss.item() * input.size(0)
        total_samples += input.size(0)

        all_targets.append(torch.argmax(target, dim=1).cpu())
        all_preds.append(torch.argmax(logits, dim=1).cpu())
    with autocast():
        inst_loss = flush_supcon_buffer()

        if inst_loss.item() > 0:
            optimizer.zero_grad()
            scaler.scale(lambda_reg * inst_loss).backward()
            scaler.step(optimizer)
            scaler.update()

    acc = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))

    return (
        running_loss / total_samples,
        running_inst / total_samples,
        acc
    )


# ================= TRAIN LOOP ==================
best_val_loss = np.inf

def main():
    # torch.set_num_threads(1)
    global best_val_loss, device, args
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)
    scaler = GradScaler()
    lib = torch.load(args.train_lib, map_location='cpu')
    if 'targets' not in lib:
        print(
            "[WARN] Loaded .pt file must contain a 'targets' key (for soft labels) for auto-weighting. Falling back to args.")
    else:
        soft_labels_list = lib['targets']
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
    # INITATE MODEL
    cudnn.benchmark = True
    model = RNN(args.dropoutRate).to(device)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    # staging 
    current_stage = 3 if args.start_unfrozen else 0
    set_unfreezing_stage(model, stage=current_stage)
    # optimizers 
    optimizer = torch.optim.Adam(get_optimizer_groups(model, args.lr, args.wd))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion_supcon = losses.SupConLoss(temperature=0.1).to(device)
    criterion = SoftCrossEntropyLoss().to(device)
  
    # 4. Initiate criterion, optimizer
    if args.loss_fn == "ce":
        w = class_weights
        if w is None and args.weights != 0.5:
            w = torch.Tensor([1 - args.weights, args.weights]).to(device,non_blocking= True )
            print(f"[INFO] Using Weighted Soft Cross-Entropy Loss (from --weights: {w.cpu().tolist()})")
        elif w is not None:
            print(f"[INFO] Using Weighted Soft Cross-Entropy Loss (from metadata: {w.cpu().tolist()})")
        else:
            print("Using Soft Cross-Entropy Loss (unweighted)")
        # w = torch.tensor([0.5, 0.5])
        criterion = SoftCrossEntropyLoss(weight=w).to(device, non_blocking= True )
    elif args.loss_fn == 'focal':
        alpha_w = class_weights
        if alpha_w is None:
            if args.focal_alpha is not None:
                alpha_w = torch.Tensor([1 - args.focal_alpha, args.focal_alpha]).to(device, non_blocking= True )
                print(
                    f"Using Focal Loss (Gamma={args.focal_gamma}, Alpha from --focal_alpha: {alpha_w.cpu().tolist()})")
            elif args.weights != 0.5:
                alpha_w = torch.Tensor([1 - args.weights, args.weights]).to(device, non_blocking= True )
                print(f"Using Focal Loss (Gamma={args.focal_gamma}, Alpha from --weights: {alpha_w.cpu().tolist()})")
            else:
                print(f"Using Focal Loss (Gamma={args.focal_gamma}, No Alpha)")
        else:
            print(f"Using Focal Loss (Gamma={args.focal_gamma}, Alpha from metadata: {alpha_w.cpu().tolist()})")

        criterion = FocalLossWithProbs(alpha=alpha_w, gamma=args.focal_gamma).to(device, non_blocking= True )
    # =============================== NORMALIZERS & TRANSFORMS
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.05,
        hue=0.02 ), transforms.GaussianBlur(
        kernel_size=3,
        sigma=(0.1, 0.6)),transforms.ToTensor(),normalize
    ])

    infer_trans = transforms.Compose([
        transforms.ToTensor(),

        normalize
    ])
    if args.train_inference_transforms_enabled:
        infer_train_transforms = trans
    else:
        infer_train_transforms = infer_trans
    # =============================== DATASET INIT

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
    log_path = os.path.join(args.output, f'training_log_{args.resolution}.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_inst_loss', 'val_loss', 'val_acc',
            'val_auc', 'val_f1', 'val_err', 'val_fpr', 'val_fnr', "SAVED"
        ])
    early_stop = 0
    best_val_loss = float('inf')
    best_val_auc = 0
    best_val_err = float('inf')
    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        if early_stop == args.patience:
            print(f"[INFO]: STOPPED AT EPOCH {epoch + 1} AFTER {args.patience} EPOCHS OF NO IMPROVEMENT")
            break
        if not args.start_unfrozen and epoch > 0 and epoch % args.unfreeze_epoch == 0 and current_stage < 3:
            current_stage += 1
            set_unfreezing_stage(model, stage=current_stage)
            optimizer = torch.optim.Adam(get_optimizer_groups(model, args.lr, args.wd))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
        current_lambda = min(args.lambda_sup, (epoch / 10) * args.lambda_sup) if epoch < 10 else args.lambda_sup
        train_dset.preselect_epoch_slides(sampling_mode=args.sampling_mode)
        train_dset.modelState(1)
        train_dset.setTransforms(infer_train_transforms)
        infer_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory)
        probs, loss = inference(infer_loader, model, criterion, enable_dropout_flag=args.train_inference_dropout_enabled)
     
        # ii. start with training based on top instances
        train_dset.maket_data(probs, args.k)
        # supcon_dset.maket_data(probs, args.k_sup)

        train_loader_new = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory)
        train_dset.modelState(2)
        train_dset.setTransforms(trans)
        t_loss, i_loss, t_acc = train_epoch(epoch+1, train_loader_new, model, criterion, criterion_supcon, optimizer, current_lambda, device, scaler)
        log_data = {
            'epoch': epoch + 1,
            'train_loss': t_loss,
            'train_inst_loss': i_loss,
            'train_acc': t_acc,
            'val_loss': np.nan, 'val_acc': np.nan, 'val_auc': np.nan, 'val_err': np.nan,
            'SAVED': ""
        }
        # iii. inference on validation
        if (epoch + 1) % args.validation_interval == 0 and args.val_lib:
            val_dset.modelState(1)
            probs, val_loss = inference(val_loader, model, criterion, enable_dropout_flag=False)
            maxs = ut.groupTopKtilesProbabilities(np.array(val_dset.slideIDX), probs, len(val_dset.targets))

            true_labels_tensors = val_dset.targets
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
            if log_data['val_loss'] < best_val_loss:
                best_val_loss = log_data['val_loss']
                early_stop = 0
                log_data['SAVED'] = "BEST_LOSS"
                print(f"\n  ** New best validation loss: {best_val_loss:.6f} at epoch {epoch + 1}. Saving model. **")
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.output, f'checkpoint_best_{args.resolution}_epoch_{epoch + 1}.pth')
                torch.save(obj, save_path)
            elif log_data['val_err'] < best_val_err:
                best_val_err = log_data['val_err']
                early_stop = 0
                log_data['SAVED'] = "BEST_ERR"

                print(f"\n  ** New best validation err: {best_val_err:.6f} at epoch {epoch + 1}. Saving model. **")
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.output, f'checkpoint_best_err_{args.resolution}_epoch_{epoch + 1}.pth')
                torch.save(obj, save_path)
            elif log_data['val_auc'] > best_val_auc:
                best_val_auc = log_data['val_auc']
                early_stop = 0
                log_data['SAVED'] = "BEST_AUC"
                print(f"\n  ** New best validation AUC: {best_val_auc:.6f} at epoch {epoch + 1}. Saving model. **")
                obj = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val_loss': best_val_loss,
                    'optimizer': optimizer.state_dict()
                }
                save_path = os.path.join(args.output, f'checkpoint_best_auc_{args.resolution}_epoch_{epoch + 1}.pth')
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
                f"{log_data['val_fnr']:.6f}" if not np.isnan(log_data['val_fnr']) else '',
                f"{log_data['SAVED']}
            ])

        scheduler.step()
if __name__ == '__main__':
    main()
