import argparse
import os
import numpy as np
import torch
import wandb
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from muscles.datasets.dataloader import load_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists, count_param_numbers
from utils.learning import AverageMeter, decay_lr_exponentially

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/xuqianxu/muscles/datasets',
                        help='Path to the dataset directory.')
    parser.add_argument('--folder_names', type=str, nargs='+',
                        default=["BMLmovi", "BMLrub", "KIT", "TotalCapture"],
                        help='List of folder names in the dataset directory.')
    parser.add_argument("--config", type=str, default="/home/xuqianxu/MMTransformer/configs/h36m/M2M.yaml",
                        help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH', help='Checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                        help='New checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="Checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts

def train_one_epoch(args, model, train_loader, optimizer, device, loss_meter):
    model.train()
    for sample in tqdm(train_loader, desc="Training"):
        # Assume sample is a tuple: (pose, muscle, demo)
        pose, muscle, demo, mask = sample
        pose = pose.to(device)         # [batch, seq_len, pose_dim]
        muscle = muscle.to(device)     # [batch, seq_len, muscle_dim]
        demo = demo.to(device)         # [batch, demo_dim]
        
        optimizer.zero_grad()
        # Forward pass: model expects pose sequence and demo info
        pred = model(pose, demo, mask=mask)    # Output: (seq_len, batch, muscle_dim)
        
        # Compute MSE loss between predicted and ground-truth muscle activations
        loss = torch.nn.functional.mse_loss(pred, muscle)
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), pose.size(0))
    return loss_meter.avg

def evaluate(args, model, test_loader, device):
    model.eval()
    losses = AverageMeter()
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Evaluating"):
            pose, muscle, demo = sample
            pose = pose.to(device)
            muscle = muscle.to(device)
            demo = demo.to(device)
            pose_seq = pose.permute(1, 0, 2)
            pred = model(pose_seq, demo)
            pred = pred.permute(1, 0, 2)
            loss = torch.nn.functional.mse_loss(pred, muscle)
            losses.update(loss.item(), pose.size(0))
    print("Evaluation MSE Loss:", losses.avg)
    return losses.avg

def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model, min_loss, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'min_loss': min_loss,
        'wandb_id': wandb_id,
    }, checkpoint_path)

def train(args, opts):
    #print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)
    
    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': max((opts.num_cpus - 1) // 3, 1),
        'persistent_workers': True
    }

    train_loader = load_data(opts.data_dir, opts.folder_names, mode='train')
    test_loader = load_data(opts.data_dir, opts.folder_names, mode='test')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Instantiate your adapted model (MotionToMuscleTransformer)
    model = load_model(args)
    #if torch.cuda.is_available():
        #model = torch.nn.DataParallel(model)
    model.to(device)
    
    n_params = count_param_numbers(model)
    print(f"[INFO] Number of parameters: {n_params:,}")
    
    lr = args.learning_rate
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=args.weight_decay)
    lr_decay = args.lr_decay
    epoch_start = 0
    min_loss = float('inf')
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else (wandb.util.generate_id() if opts.use_wandb else None)
    
    # Resume from checkpoint if available
    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint,
                                       opts.checkpoint_file if opts.checkpoint_file else "latest_epoch.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['model'], strict=True)
            if opts.resume:
                lr = checkpoint['lr']
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_loss = checkpoint['min_loss']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint not found. Starting from scratch.")
            opts.resume = False

    # Initialize Weights & Biases if requested
    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id, project='Motion2Muscle', resume="must",
                           settings=wandb.Settings(start_method='fork'))
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(id=wandb_id, name=opts.wandb_name, project='Motion2Muscle',
                           settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args.__dict__)
    
    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth')
    
    loss_meter = AverageMeter()
    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(args, model, test_loader, device)
            exit()
            
        print(f"[INFO] Epoch {epoch}")
        train_loss = train_one_epoch(args, model, train_loader, optimizer, device, loss_meter)
        eval_loss = evaluate(args, model, test_loader, device)
        
        if eval_loss < min_loss:
            min_loss = eval_loss
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_loss, wandb_id)
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_loss, wandb_id)
        
        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
            }, step=epoch + 1)
            
        lr = decay_lr_exponentially(lr, lr_decay, optimizer)
    
    if opts.use_wandb:
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(checkpoint_path_latest)
        artifact.add_file(checkpoint_path_best)
        wandb.log_artifact(artifact)

def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)
    train(args, opts)

if __name__ == '__main__':
    main()
