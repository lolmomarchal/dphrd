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
parser.add_argument('--k', default=100, type=int,
                    help='The top k tiles based on predicted model probabilities used as representative features of the training classes for each slide.')
parser.add_argument('--checkpoint', default=None, type=str,
                    help='Checkpoint model to restart a given training session or for transfer learning.')
parser.add_argument('--resolution', type=str, default='5x', help='Current magnification resolution')
parser.add_argument('--dropoutRate', default=0.2, type=float,
                    help='Rate of dropout to be used within the fully connected layers.')
parser.add_argument('--weights', default=0.5, type=float, help='Unbalanced positive class weight (for CE loss).')
parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers.')
parser.add_argument('--gpu', default=0, type=int, help='Gpu device selection.')
parser.add_argument('--patience', default=20, type=int, help='Gpu device selection.')

# --- NEW ARGUMENTS ---
parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'focal'],
                    help='Loss function to use (ce or focal).')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss.')
parser.add_argument('--focal_alpha', type=float, default=None,
                    help='Alpha parameter (class weight) for Focal Loss. If None, uses --weights argument if not 0.5.')
parser.add_argument('--disable_weighted_sampling', action='store_true', help='Disable weighted sampling by subtype during training.')

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

    # --- Code that runs ONCE after the batch loop completes ---
    if total_samples == 0:
        mean_loss = 0.0
    else:
        mean_loss = running_loss / total_samples

    # Slide Debug Writing
    slide_path = os.path.join(args.output, "slide_debug_last_infer.tsv")
    with open(slide_path, "w") as f:
        print("slide_id\tn_tiles\tmean_prob\ttarget_label", file=f)
        for sid, info in slide_outputs.items():
            probs_arr = np.array(info["probs"])
            mean_prob = np.mean(probs_arr)
            target_label = info["target"].cpu().numpy().tolist()
            n_tiles = len(probs_arr)
            print(f"{sid}\t{n_tiles}\t{mean_prob:.4f}\t{target_label}", file=f)

    if not probs_list:
        all_probs = np.array([])
    else:
        all_probs = torch.cat(probs_list, dim=0).numpy()

    # The return values here should match the expectation in main()
    # Ensure they are correctly formatted for the groupTopKtiles call.
    return all_probs, mean_loss


def train(run, loader, model, criterion, criterion_supcon, optimizer, lambda_reg=0.2, device="cpu"):
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

    for i, (input, target, slide_ids) in tqdm.tqdm(enumerate(loader), total=len(loader), desc="[TRAINING]"):
        input = input.to(device, non_blocking= True )
        target = target.to(device, non_blocking= True )  # Target shape is (B, 2), e.g., [[0.0, 1.0], [0.3, 0.7]]
        slide_ids = slide_ids.to(device, non_blocking= True )
        batch_size = input.size(0)
        total_samples += batch_size

        # 1. Forward Pass
        # projected_features shape is (B, 128)
        logits, _, projected_features = model(input)

        # 2. Main Classification Loss (runs on ALL tiles)
        # This loss handles soft labels (e.g., [0.3, 0.7]) correctly.
        loss_cls = criterion(logits, target)

        # 3. Supervised Contrastive Loss (runs ONLY on "definitive" tiles)

        definitive_mask = (target[:, 0] == 0.0) | (target[:, 0] == 1.0)

        # Get the hard labels (0 or 1) for all tiles in the batch
        # Shape: (B,)
        hard_labels = torch.argmax(target, dim=1)

        # Filter: select only the features and labels that correspond
        # to our "definitive" tiles.
        features_for_supcon = projected_features[definitive_mask]
        labels_for_supcon = hard_labels[definitive_mask]

        inst_loss = torch.tensor(0.0, device=device)
        # SupCon loss requires at least 2 samples to compare.
        if labels_for_supcon.shape[0] > 1:
            inst_loss = criterion_supcon(features_for_supcon, labels_for_supcon)
        # --- END OF NEW LOGIC ---

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
        with torch.no_grad():
            probs = F.softmax(logits, dim=1)[:, 1]

        for sid, t, p in zip(slide_ids.cpu().numpy(), target.cpu(), probs.cpu()):
            sid_key = int(sid)
            if sid_key not in slide_outputs:
                slide_outputs[sid_key] = {"probs": [], "target": t}
            slide_outputs[sid_key]["probs"].append(p.item())

    # --- Code that runs ONCE after the batch loop completes ---
    if total_samples == 0:
        epoch_loss = 0.0
        epoch_inst_loss = 0.0
    else:
        epoch_loss = running_loss / total_samples
        epoch_inst_loss = running_inst_loss / total_samples

    # Corrected DEBUG SECTION (Now outside the loop and writing correctly)
    train_slide_path = os.path.join(args.output, f"train_slide_debug_epoch{run}.tsv")
    with open(train_slide_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["slide_id", "n_tiles", "mean_prob", "target_label"])
        for sid, info in slide_outputs.items():
            probs_arr = np.array(info["probs"])
            mean_prob = np.mean(probs_arr)
            # .tolist() is safer for CSV writing
            target_label = info["target"].cpu().numpy().tolist()
            n_tiles = len(probs_arr)
            writer.writerow([sid, n_tiles, f"{mean_prob:.4f}", target_label])

    # --- MODIFIED FINAL PRINT STATEMENT ---
    print(
        f"[EPOCH {run}] Mean train loss: {epoch_loss:.6f} | "
        f"Mean Inst. Loss: {epoch_inst_loss:.6f}"
    )
    # Return both losses, assuming your calling function handles them
    return epoch_loss, epoch_inst_loss

def compute_inst_regulation(instance_probs, instance_features, slide_labels, slide_ids, k=10, device="cpu"):
    """
    Instance-level regularization (MSE) using 128D projected features.

    Parameters:
        instance_probs:     [tensor] 1D probabilities (for top-k selection).
        instance_features:  [tensor] 128D features (for MSE calculation).
        slide_labels:       [tensor] Soft/hard labels for the batch.
        slide_ids:          [tensor] Slide IDs for the batch.
    """

    unique_slides = torch.unique(slide_ids)
    slide_losses = []
    slide_count = 0

    for slide_id in unique_slides:
        mask = (slide_ids == slide_id)
        n = mask.sum().item()
        if n == 0:
            continue

        slide_probs = instance_probs[mask]
        slide_feature_vectors = instance_features[mask]  # <-- Get 128D features
        slide_label = slide_labels[mask][0]  # Soft label for the slide

        # Determine the hard label score (0 or 1)
        if slide_label.numel() == 2:
            hrd_score = slide_label[1].item()  # Assuming class 1 is the positive class
        else:
            hrd_score = slide_label.item()

        # Apply loss only to slides with HARD labels (0.0 or 1.0)
        if hrd_score == 0.0 or hrd_score == 1.0:
            # 1. Use PROBABILITIES to find the top-k tiles (Selection)
            k_eff = min(k, slide_probs.numel())
            topk_idx = torch.topk(slide_probs, k=k_eff, dim=0)[1]

            # 2. Select the corresponding 128D FEATURES for loss calculation
            topk_features = slide_feature_vectors[topk_idx]  # Shape: (k_eff, 128)

            # 3. Create the 128D Pseudo-Target Vector
            # Target is a tensor of shape (k_eff, 128) where every element is hrd_score
            pseudo_target_vector = torch.full_like(topk_features, fill_value=hrd_score)

            # 4. Compute Loss (MSE on 128D vectors)
            # F.mse_loss averages over all elements (k_eff * 128)
            slide_loss = F.mse_loss(topk_features, pseudo_target_vector, reduction='mean')

            slide_losses.append(slide_loss)
            slide_count += 1

    if slide_count == 0:
        # Return a zero tensor that still requires a gradient for stability
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Compute the mean loss across all eligible slides
    total_loss = torch.mean(torch.stack(slide_losses))

    return total_loss


# ================= TRAIN LOOP ==================
best_val_loss = np.inf

def main():
    # 1. Initiate params + output
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cudnn.benchmark = True

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



    # 6. Dataset start
    train_dset = ut.MILdataset(args.train_lib, trans, disable_weighted_sampling=args.disable_weighted_sampling)
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
        train_dset.modelState(1)
        train_dset.setTransforms(infer_trans)
        infer_loader = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=pin_memory)
        probs, loss = inference(infer_loader, model, criterion, enable_dropout_flag=True)
        if probs.size == 0:
            print("\n[FATAL ERROR] Inference returned an empty probability array. Cannot group tiles.")
            # return
            #
        if not np.issubdtype(probs.dtype, np.number):
            print(f"\n[FATAL ERROR] Probabilities dtype is non-numeric: {probs.dtype}. Cannot sort.")

        # topk = ut.groupTopKtiles(np.array(train_dset.slideIDX), probs,
        #                          args.k)  # get top-k tiles for training + prediction
        # ii. start with training based on top instances
        # train_dset.maketraindata(topk)
        train_dset.maket_data(probs, args.k)
        train_loader_new = torch.utils.data.DataLoader(
            train_dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=pin_memory)
        train_dset.modelState(2)
        train_dset.setTransforms(trans)

        train_loss, train_inst_loss = train(epoch + 1, train_loader_new, model,
                                            criterion, criterion_supcon, optimizer, # <-- Pass new criterion
                                            lambda_reg=0.2,
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


            maxs = ut.groupTopKtilesProbabilities(np.array(val_dset.slideIDX), probs, len(val_dset.targets))

            true_labels_tensors = val_dset.targets
            true_labels_1d = np.array([torch.argmax(t).item() for t in true_labels_tensors])

            pred_binary = np.array([1 if x >= 0.5 else 0 for x in maxs])
            auc = roc_auc_score(true_labels_1d, maxs)
            accuracy = accuracy_score(true_labels_1d, pred_binary)
            f1 = f1_score(true_labels_1d, pred_binary)

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
