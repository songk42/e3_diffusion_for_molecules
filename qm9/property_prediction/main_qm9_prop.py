import sys, os, glob, re
sys.path.append(os.path.abspath(os.path.join('../../')))
from qm9.property_prediction.models_property import EGNN, Naive, NumNodes
import torch
from torch import nn, optim
import argparse
from qm9.property_prediction import prop_utils
import json
from qm9 import dataset, utils
import pickle
import wandb

loss_l1 = nn.L1Loss()


def train(model, epoch, loader, mean, mad, property, device, partition='train', optimizer=None, lr_scheduler=None, log_interval=20, debug_break=False, scaler=None):
    if partition == 'train':
        lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()

        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, 1).to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32)

        nodes = nodes.view(batch_size * n_nodes, -1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[property].to(device, torch.float32)

        if partition == 'train':
            with torch.cuda.amp.autocast():
                pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                             n_nodes=n_nodes)
                loss = loss_l1(pred, (label - mean) / mad)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.cuda.amp.autocast():
                pred = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                             n_nodes=n_nodes)
            loss = loss_l1(mad * pred + mean, label)

        res['loss'] += loss.item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(loss.item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % log_interval == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
        if debug_break:
            break
    return res['loss'] / res['counter']


def test(model, epoch, loader, mean, mad, property, device, log_interval, debug_break=False, scaler=None):
    return train(model, epoch, loader, mean, mad, property, device, partition='test', log_interval=log_interval, debug_break=debug_break, scaler=scaler)


def get_model(args):
    if args.model_name == 'egnn':
        model = EGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=args.nf, device=args.device, n_layers=args.n_layers,
                     coords_weight=1.0,
                     attention=args.attention, node_attr=args.node_attr)
    elif args.model_name == 'naive':
        model = Naive(device=args.device)
    elif args.model_name == 'numnodes':
        model = NumNodes(device=args.device)
    else:
        raise Exception("Wrong model name %s" % args.model_name)


    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='QM9 Example')
    parser.add_argument('--exp_name', type=str, default='debug', metavar='N',
                        help='experiment_name')
    parser.add_argument('--batch_size', type=int, default=96, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before logging test')
    parser.add_argument('--outf', type=str, default='outputs', metavar='N',
                        help='folder to output vae')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--nf', type=int, default=128, metavar='N',
                        help='learning rate')
    parser.add_argument('--attention', type=int, default=1, metavar='N',
                        help='attention in the ae model')
    parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                        help='number of layers for the autoencoder')
    parser.add_argument('--property', type=str, default='alpha', metavar='N',
                        help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve | relative_atomic_energy')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='number of workers for the dataloader')
    parser.add_argument('--filter_n_atoms', type=int, default=None,
                        help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
    parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                        help='maximum power to take into one-hot features')
    parser.add_argument('--dataset', type=str, default="qm9_first_half", metavar='N',
                        help='qm9_first_half')
    parser.add_argument('--datadir', type=str, default="../../qm9/temp", metavar='N',
                        help='qm9_first_half')
    parser.add_argument('--remove_h', action='store_true')
    parser.add_argument('--include_charges', type=eval, default=True, help='include atom charge or not')
    parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                        help='node_attr or not')
    parser.add_argument('--weight_decay', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--save_path', type=float, default=1e-16, metavar='N',
                        help='weight decay')
    parser.add_argument('--model_name', type=str, default='egnn', metavar='N',
                        help='egnn | naive | numnodes')
    parser.add_argument('--save_model', type=eval, default=True)
    parser.add_argument('--checkpoint_interval', type=int, default=0,
                        help='Save checkpoint every N epochs. 0 to disable. (default: 0)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from, or "latest" to auto-resume from the most recent checkpoint in outf/exp_name')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    dtype = torch.float32
    args.device = device
    print(args)

    wandb.init(project="qm9_property_prediction", name=args.exp_name, config=vars(args))

    res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    prop_utils.makedir(args.outf)
    prop_utils.makedir(args.outf + "/" + args.exp_name)

    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    args.dataset = "qm9_second_half"
    dataloaders_aux, _ = dataset.retrieve_dataloaders(args)
    dataloaders["test"] = dataloaders_aux["train"]

    # For relative_atomic_energy we fit a per-element linear baseline on the
    # train split (matching Symphony's RelativeAtomicEnergyConditioner) and
    # add it as a precomputed column on every split. After this, the property
    # is treated like any other dataset column.
    if args.property == 'relative_atomic_energy':
        relenergy_weights = utils.fit_relenergy_baseline(dataloaders['train'].dataset)
        for split in ('train', 'valid', 'test'):
            if split in dataloaders:
                utils.add_relative_atomic_energy(dataloaders[split].dataset, relenergy_weights)

    property_norms = utils.compute_mean_mad_from_dataloader(dataloaders['valid'], [args.property])
    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    start_epoch = 0
    if args.resume == 'latest':
        exp_dir = args.outf + "/" + args.exp_name
        ckpts = glob.glob(exp_dir + "/checkpoint_epoch_*.npy")
        if ckpts:
            args.resume = max(ckpts, key=lambda p: int(re.search(r'checkpoint_epoch_(\d+)', p).group(1)))
            start_epoch = int(re.search(r'checkpoint_epoch_(\d+)', args.resume).group(1))
            print("Resuming from checkpoint", args.resume)
        else:
            print("No epoch checkpoints found in %s, starting from scratch" % exp_dir)
            args.resume = None

    model = get_model(args)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print("Loaded checkpoint from %s" % args.resume)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.cuda)

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(model, epoch, dataloaders['train'], mean, mad, args.property, device, partition='train', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval, scaler=scaler)
        if epoch % args.test_interval == 0:
            val_loss = train(model, epoch, dataloaders['valid'], mean, mad, args.property, device, partition='valid', optimizer=optimizer, lr_scheduler=lr_scheduler, log_interval=args.log_interval, scaler=scaler)
            test_loss = test(model, epoch, dataloaders['test'], mean, mad, args.property, device, log_interval=args.log_interval, scaler=scaler)
            res['epochs'].append(epoch)
            res['losess'].append(test_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
                if args.save_model:
                    torch.save(model.state_dict(), args.outf + "/" + args.exp_name + "/best_checkpoint.npy")
                    with open(args.outf + "/" + args.exp_name + "/args.pickle", 'wb') as f:
                        pickle.dump(args, f)
            print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
            print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))
            wandb.log({'val_loss': val_loss, 'test_loss': test_loss, 'best_val': res['best_val'], 'best_test': res['best_test']}, step=epoch)

        wandb.log({'train_loss': train_loss, 'lr': lr_scheduler.get_last_lr()[0]}, step=epoch)

        if args.save_model and args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            torch.save(model.state_dict(), args.outf + "/" + args.exp_name + "/checkpoint_epoch_%d.npy" % epoch)

        json_object = json.dumps(res, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)

