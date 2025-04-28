import argparse
import os
import numpy as np
import torch
import wandb
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import csv 
import os
from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from muscles.datasets.dataloader import load_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists, count_param_numbers
from utils.learning import AverageMeter, decay_lr_exponentially
import visual.visualize as visualize
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/xuqianxu/muscles/datasets',
                        help='Path to the dataset directory.')
    parser.add_argument('--folder_names', type=str, nargs='+',
                        default=["BMLmovi", "BMLrub", "KIT", "TotalCapture"],
                        help='List of folder names in the dataset directory.')
    parser.add_argument("--config", type=str, default="/home/xuqianxu/MMTransformer/configs/M2M.yaml",
                        help="Path to the config file.")
    parser.add_argument('--log_dir', type=str, default='/home/xuqianxu/MMTransformer/logs',)
    parser.add_argument('-c', '--checkpoint', default='/home/xuqianxu/MMTransformer/ckp_trans', type=str, metavar='PATH', help='Checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='/home/xuqianxu/MMTransformer/ckp_trans',
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

def train_one_epoch(args, model, train_loader, optimizer, device, loss_meter, att):
    model.train()
    pbar = tqdm(train_loader, desc="Training", dynamic_ncols=True)

    for pose, muscle, demo in pbar:
        pose = pose.to(device)     # [B, T, J, C]
        muscle = muscle.to(device) # [B, T, D]
        demo = demo.to(device)     # [B, 3]

        # Normalize each batch (optionally per-sample too)
        #muscle_mean = muscle.mean(dim=(1, 2), keepdim=True)  # mean over T and D
        #muscle_std = muscle.std(dim=(1, 2), keepdim=True)
        #muscle = (muscle - muscle_mean) / (muscle_std + 1e-6)

        optimizer.zero_grad()

        pred = model(pose, demo) #if att else model(pose) # Output: [B, T, D]
        loss = torch.nn.SmoothL1Loss()(pred, muscle)

        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), pose.size(0))

        # Show avg loss on tqdm
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    return loss_meter.avg

def evaluate(opts, model, test_loader, device, att):
    model.eval()
    losses_mse = AverageMeter()
    losses_mae = AverageMeter()
    #muscle_index_to_plot = [0]  # Index of the muscle to visualize
    if opts.eval_only:
        has_plotted = False
        male, female = visualize.build_body()
    with torch.no_grad():
        for sample in tqdm(test_loader, desc="Evaluating"):
            if opts.eval_only:
                pose, muscle, demo, sample_id = sample
            else:
                pose, muscle, demo = sample
            #draw_to_batch(pose)
            # Save GIFs to a folder
            pose = pose.to(device)
            muscle = muscle.to(device)
            #muscle_mean = muscle.mean(dim=(1, 2), keepdim=True)  # mean over T and D
            #muscle_std = muscle.std(dim=(1, 2), keepdim=True)
            #muscle = (muscle - muscle_mean) / (muscle_std + 1e-6)
            demo = demo.to(device)
            pred = model(pose, demo) #if att else model(pose)
            loss_mse = torch.nn.MSELoss()(pred, muscle)
            loss_mae = torch.nn.functional.l1_loss(pred, muscle)
            losses_mse.update(loss_mse.item(), pose.size(0))
            losses_mae.update(loss_mae.item(), pose.size(0))
            if opts.eval_only and not has_plotted:
                visualize.visualize_pose_and_muscle(sample_id, male, female, pred, muscle)
                #visualize.visualize_pose(sample_id, male, female)
                #visualize.visualize_muscle(pred, muscle, muscle_index_to_plot)
                has_plotted = True
    print("Evaluation MSE Loss:", losses_mse.avg)  
    print("Evaluation MAE Loss:", losses_mae.avg)
    return losses_mse.avg, losses_mae.avg

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
    att=True if args.model_name =='MotionAGFormer' else False
    do_logging = True 
    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': max((opts.num_cpus - 1) // 3, 1),
        'persistent_workers': True
    }

    train_loader = load_data(opts, mode='train')
    test_loader = load_data(opts, mode='test')
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
                                       opts.checkpoint_file if opts.checkpoint_file else "best_epoch.pth")
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
    
    checkpoint_path_latest = os.path.join(opts.new_checkpoint, f'{args.model_name}_latest_epoch.pth')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, f'{args.model_name}_best_epoch.pth')
    
    loss_meter = AverageMeter()
    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(opts, model, test_loader, device, att)
            exit()
            
        print(f"[INFO] Epoch {epoch}")
        train_loss = train_one_epoch(args, model, train_loader, optimizer, device, loss_meter, att)
        eval_loss_mse, eval_loss_mae = evaluate(opts, model, test_loader, device, att)
        if do_logging:  # optional flag
            log_dir = os.path.join(opts.log_dir, f'new_{args.model_name}_wo.csv')
            log_exists = os.path.exists(log_dir)
            with open(log_dir, "a", newline='') as f:
                writer = csv.writer(f)
                if not log_exists:
                    writer.writerow(["Epoch", "Eval Loss MSE", "Eval Loss MAE"])
                writer.writerow([epoch, eval_loss_mse, eval_loss_mae])
        
        if eval_loss_mse < min_loss:
            min_loss = eval_loss_mse
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model, min_loss, wandb_id)
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model, min_loss, wandb_id)
        
        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train_loss': train_loss,
                'eval_loss': eval_loss_mse,
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
