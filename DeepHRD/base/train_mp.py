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
# sys.path.append("/home/lolmomarchal/IdeaProjects/dphrd/DeepHRD/base")
from model import ResNet_dropout as RNN
from snorkel.classification import cross_entropy_with_probs
import utilsModel as ut
import tqdm
import warnings
import csv
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from pytorch_metric_learning import losses
from torch.cuda.amp import autocast
# To ignore all warnings:
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

# ================ LOSS ======================

class SoftCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss function that works with soft probability targets.
    """
    def __init__(self, weight=None, reduction='mean'):
        super(SoftCrossEntropyLoss, self).__init__()
        self.weight = weight # Shape (2)
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(targets * log_probs)

        if self.weight is not None:
            # Convert (2) weight tensor to (1, 2) and broadcast across the batch dimension (64)
            weight = self.weight.to(logits.device).unsqueeze(0)
            loss = loss * weight # Now, (64, 2) * (1, 2) is possible via broadcasting, resulting in (64, 2)

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

        # Calculate the focal term (1 - p_c)^gamma
        focal_weight = (1 - probs) ** self.gamma

        # Base cross-entropy term (y_c * log(p_c)) combined with focal weight
        # 'loss' shape is (N, C) -> (Batch_size, 2)
        loss = -targets * focal_weight * log_probs

        # --- FIX: Apply alpha weighting using broadcasting ---
        if self.alpha is not None:
            # unsqueeze(0) changes shape from (2) to (1, 2) for broadcasting
            alpha = self.alpha.to(logits.device).unsqueeze(0)
            loss = alpha * loss # Now (N, 2) * (1, 2) is possible

        loss = loss.sum(dim=1)  # sum over classes, result shape (N)

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
                with autocast():
                    logits = model(input)
                    output = F.softmax(logits, dim=1)
                    loss = criterion(logits, target)
            else:
                logits = model(input)
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

    slide_outputs = {}
    supcon_iter = iter(supcon_loader)
    for i, (input, target, slide_ids) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[TRAINING]"):
        input = input.to(device, non_blocking= True )
        target = target.to(device, non_blocking= True )  # Target shape is (B, 2), e.g., [[0.0, 1.0], [0.3, 0.7]]
        slide_ids = slide_ids.to(device, non_blocking= True )
        batch_size = input.size(0)
        total_samples += batch_size

        # 1. Forward Pass & main classification loss w/ k =1
        logits, _, projected_features = model(input)
        loss_cls = criterion(logits, target)


        inst_loss = torch.tensor(0.0, device=device)
        input_sup, target_sup = None, None
        if lambda_reg != 0:
            # do supervise contrast loss on k_sup tiles and only for super sure
            try:
                input_sup, target_sup, _ = next(supcon_iter)
            except StopIteration:
                supcon_iter = iter(supcon_loader)
                input_sup, target_sup, _ = next(supcon_iter)
            except RuntimeError as e:
                if "expected more than 0 tensors" in str(e):
                    print("[WARNING] SupCon loader is empty for this step. Skipping SupCon Loss.")
                else:
                    raise e
            if input_sup is not None:
                input_sup = input_sup.to(device, non_blocking=True)
            target_sup = target_sup.to(device, non_blocking=True)

            _, _, projected_features_sup = model(input_sup)
            definitive_mask = (target_sup[:, 0] < 0.1) | (target_sup[:, 0] > 0.9)
            hard_labels = torch.argmax(target_sup, dim=1)

            features_for_supcon = projected_features_sup[definitive_mask]
            labels_for_supcon = hard_labels[definitive_mask]
            if labels_for_supcon.shape[0] > 1:
                inst_loss = criterion_supcon(features_for_supcon, labels_for_supcon)
            del features_for_supcon, target_sup
        del input, target
        torch.cuda.empty_cache()

        # 4. Total Loss Calculation
        loss = (1 - lambda_reg) * loss_cls + lambda_reg * inst_loss


        # 5. Backpropagation and Optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 6. Aggregation (for epoch-end reporting)
        running_loss += loss.item() * batch_size
        running_inst_loss += inst_loss.item() * batch_size # Accumulate the (potentially 0) inst_loss

        # 7. Slide Output Tracking
        # We need to calculate probs for logging, but not for the loss
        # with torch.no_grad():
        #     probs = F.softmax(logits, dim=1)[:, 1]

        # for sid, t, p in zip(slide_ids.cpu().numpy(), target.cpu(), probs.cpu()):
        #     sid_key = int(sid)
        #     if sid_key not in slide_outputs:
        #         slide_outputs[sid_key] = {"probs": [], "target": t}
        #     slide_outputs[sid_key]["probs"].append(p.item())

    if total_samples == 0:
        epoch_loss = 0.0
        epoch_inst_loss = 0.0
    else:
        epoch_loss = running_loss / total_samples
        epoch_inst_loss = running_inst_loss / total_samples

    print(
        f"[EPOCH {run}] Mean train loss: {epoch_loss:.6f} | "
        f"Mean Inst. Loss: {epoch_inst_loss:.6f}"
    )
    return epoch_loss, epoch_inst_loss


# ================= TRAIN LOOP ==================
best_val_loss = np.inf

def main():
    # 1. Initiate params + output
    torch.set_num_threads(1)
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

    # 2. Calculate class weights -> done by using hard labels
    print(f"[INFO] Reading metadata from {args.train_lib} to calculate class weights...")
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
    # 3. Initiate model
    model = RNN(args.dropoutRate)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint)["state_dict"])
    model.to(device)
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
        w = torch.tensor([0.5, 0.5])
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

    criterion_supcon = losses.SupConLoss(temperature=0.07).to(device, non_blocking= True )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    cudnn.benchmark = True
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',         # Monitor a minimum metric (validation loss)
        factor=0.5,         # Halve the learning rate when plateauing
        patience=args.patience // 2, # Wait half the early stop patience before reducing LR
        verbose=True,
        min_lr=1e-6         # Do not let the LR drop below this
    )

    # 5. Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.406, 0.406], std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomRotation(180),
                                transforms.ColorJitter(brightness=0.5, contrast=[0.2, 1.8], saturation=0, hue=0),
                                transforms.ToTensor(), normalize])

    infer_trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    if args.train_inference_transforms_enabled:
        infer_train_transforms = trans
    else:
        infer_train_transforms = infer_trans

    # 6. Dataset start
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
    # 7. Logging
    log_path = os.path.join(args.output, f'training_log_{resolution}.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_inst_loss', 'val_loss', 'val_acc',
            'val_auc', 'val_f1', 'val_err', 'val_fpr', 'val_fnr'
        ])
    # 8. training loop
    early_stop = 0
    for epoch in tqdm.tqdm(range(args.epochs), total=args.epochs):
        if early_stop == args.patience:
            print(f"[INFO]: STOPPED AT EPOCH {epoch + 1} AFTER {args.patience} EPOCHS OF NO IMPROVEMENT")
            break
        # i. start with overall inference
        train_dset.preselect_epoch_slides(sampling_mode=args.sampling_mode) # to select trained slides
        train_dset.modelState(1)
        train_dset.setTransforms(infer_train_transforms)
        infer_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory)
        probs, loss = inference(infer_loader, model, criterion, enable_dropout_flag=args.train_inference_dropout_enabled)
        # topk = ut.groupTopKtiles(np.array(train_dset.slideIDX), probs,
        #                          args.k)  # get top-k tiles for training + prediction
        # ii. start with training based on top instances
        # train_dset.maketraindata(topk)
        train_dset.maket_data(probs, args.k)

        supcon_dset = ut.MILdataset(args.train_lib, trans)
        supcon_dset.epoch_tile_info = train_dset.epoch_tile_info.copy()
        supcon_dset.epoch_slide_id_map = train_dset.epoch_slide_id_map.copy()
        supcon_dset.epoch_target_map = train_dset.epoch_target_map.copy()
        supcon_dset.epoch_subtype_map = train_dset.epoch_subtype_map.copy()
        supcon_dset.maket_data(probs, args.k_sup)

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
            probs, val_loss = inference(val_loader, model, criterion, enable_dropout_flag=False)
            scheduler.step(val_loss)
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

            # --- Update log_data ---
            log_data['val_loss'] = val_loss
            log_data['val_acc'] = accuracy
            log_data['val_auc'] = auc
            log_data['val_f1'] = f1
            log_data['val_err'] = err
            log_data['val_fpr'] = fpr
            log_data['val_fnr'] = fnr
            if err < best_val_loss:
                best_val_loss = err
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
