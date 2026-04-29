import argparse
import csv
import re
from os.path import join
import torch
from torch import tensor
import os
import glob
import pickle
import ase.io
from qm9.property_prediction import prop_utils
from qm9.models import get_model
from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.sampling import sample
from qm9.property_prediction.main_qm9_prop import test
from qm9.property_prediction import main_qm9_prop
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import qm9.visualizer as vis
import tqdm

# Symphony conditioning normalization (QM9 train split, idx 0..50000).
# Source: ~/symphony-torch/scripts/sample_sweep_alpha.sh, sample_sweep.sh.
# unit_factor converts Symphony's stored units to EDM's QM9 units (eV -> Hartree
# for gap; identity otherwise).
SYMPHONY_NORM = {
    'alpha': {'mean': 72.9519, 'std': 9.0731, 'unit_factor': 1.0},
    'gap':   {'mean': 6.8997323607, 'std': 1.2955285899, 'unit_factor': 0.0367493},
    'relative_atomic_energy': {'mean': 0.0, 'std': 0.060994, 'unit_factor': 1.0},
}

_TENSOR_RE = re.compile(r'tensor\(\[\[(.*?)\]\]\)')


def _parse_symphony_conditioning_line(line):
    """Extract the list of floats from a Symphony xyz comment line of the form
    `# Conditioning <...>: tensor([[v0, v1, ...]])`. Returns None if the line
    has no tensor literal."""
    m = _TENSOR_RE.search(line)
    if m is None:
        return None
    return [float(x) for x in m.group(1).split(',')]


def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier


def get_args_gen(dir_path):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_gen = pickle.load(f)
    assert args_gen.dataset == 'qm9_second_half'

    # Add missing args!
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'
    return args_gen


def get_generator(dir_path, dataloaders, device, args_gen, property_norms):
    dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)
    args_gen.nn_cutoff = None
    model, nodes_dist, prop_dist = get_model(args_gen, device, dataset_info, dataloaders['train'])
    fn = 'generative_model_ema.npy' if args_gen.ema_decay > 0 else 'generative_model.npy'
    model_state_dict = torch.load(join(dir_path, fn), map_location='cpu')
    model.load_state_dict(model_state_dict)

    # The following function be computes the normalization parameters using the 'valid' partition

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    return model.to(device), nodes_dist, prop_dist, dataset_info


def get_dataloader(args_gen):
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args_gen)
    return dataloaders


class XYZDataloader:
    def __init__(self, args_gen, xyz_dir, device, unknown_labels=False,
                 batch_size=1, iterations=200, prop_dist=None,
                 target_property=None, cond_keys=None):
        """
        Dataloader for XYZ files that follows the same API as DiffusionDataloader.

        Args:
            device: Device to load data on
            unknown_labels: Whether labels are unknown (kept for API consistency)
            batch_size: Batch size
            iterations: Number of iterations before raising StopIteration
            xyz_dir: Directory containing XYZ files to load
            target_property: name of the property to extract from each xyz file
                and expose as `data[target_property]`. Defaults to
                `prop_dist.properties[0]`.
            cond_keys: ordered list of property names matching the elements of
                the Symphony conditioning tensor in the xyz comment line. Used
                to pick `target_property` out of multi-element conditioning
                tensors. Defaults to `[target_property]` (single-element).
        """
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unknown_labels = unknown_labels
        self.prop_dist = prop_dist
        self.dataset_info = get_dataset_info(args_gen.dataset, args_gen.remove_h)

        # XYZ specific parameters
        self.xyz_dir = xyz_dir
        self.xyz_files = self._get_xyz_files()
        self.current_idx = 0
        self.i = 0

        if target_property is None:
            target_property = (prop_dist.properties[0]
                               if prop_dist is not None and hasattr(prop_dist, 'properties')
                               else 'dummy_property')
        self.target_property = target_property
        self.cond_keys = list(cond_keys) if cond_keys else [target_property]
        if self.target_property not in self.cond_keys:
            raise ValueError(
                f"target_property {self.target_property!r} not in cond_keys "
                f"{self.cond_keys!r}")
    
    
    def _get_xyz_files(self):
        """Get all XYZ files in the directory."""
        if not os.path.exists(self.xyz_dir):
            raise FileNotFoundError(f"XYZ directory {self.xyz_dir} not found")
        
        xyz_files = glob.glob(os.path.join(self.xyz_dir, "*.xyz"))
        if not xyz_files:
            raise FileNotFoundError(f"No XYZ files found in {self.xyz_dir}")
        
        return xyz_files
    
    def _load_xyz_batch(self):
        """Load a batch of XYZ files."""
        batch_data = []
        
        for _ in range(self.batch_size):
            if self.current_idx >= len(self.xyz_files):
                self.current_idx = 0  # Cycle through files if we run out
            
            xyz_file = self.xyz_files[self.current_idx]
            try:
                batch_data.append(self._process_xyz_file(xyz_file))
            except:
                print("Could not read", xyz_file)
                pass
            self.current_idx += 1
        
        return self._collate_batch(batch_data)
    
    def _process_xyz_file(self, xyz_file):
        """Process a single XYZ file into the required format."""
        # Read molecule using ASE
        molecule = ase.io.read(xyz_file)
        
        # Extract positions and atom types
        positions = torch.tensor(molecule.get_positions(), dtype=torch.float32)
        atom_types = molecule.get_chemical_symbols()
        
        # Create one-hot encoding for atom types
        n_nodes = len(atom_types)
        one_hot = torch.zeros((n_nodes, len(self.dataset_info['atom_types'])))
        
        for i, atom_type in enumerate(atom_types):
            # Handle atom types not in the dataset info
            if atom_type not in self.dataset_info['atom_encoder']:
                print(f"Warning: Atom type {atom_type} not in dataset_info, defaulting to first atom type")
                atom_idx = 0
            else:
                atom_idx = self.dataset_info['atom_encoder'][atom_type]
            one_hot[i, atom_idx] = 1
        
        # Create node mask
        node_mask = torch.ones(self.dataset_info['max_n_nodes'], dtype=torch.bool)
        node_mask[n_nodes:] = False
        
        # Pad positions and one_hot to max_n_nodes
        padded_positions = torch.zeros((self.dataset_info['max_n_nodes'], 3), dtype=torch.float32)
        padded_positions[:n_nodes] = positions
        
        padded_one_hot = torch.zeros((self.dataset_info['max_n_nodes'], len(self.dataset_info['atom_types'])))
        padded_one_hot[:n_nodes] = one_hot
        
        # Property value lives on line 2 of the xyz file as a Symphony
        # conditioning comment, e.g.
        #   # Conditioning <...>: tensor([[v0, v1, ...]])
        target_key = self.target_property
        prop_value = None
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        if len(lines) > 1 and lines[1].strip():
            vals = _parse_symphony_conditioning_line(lines[1])
            if vals is not None:
                if len(vals) != len(self.cond_keys):
                    raise ValueError(
                        f"{xyz_file}: tensor has {len(vals)} elements but "
                        f"cond_keys has {len(self.cond_keys)} ({self.cond_keys})")
                prop_value = vals[self.cond_keys.index(target_key)]
            else:
                # legacy fallback: last token is just a float
                prop_value = float(lines[1].split()[-1])
        if prop_value is None and self.prop_dist is not None:
            prop_value = self.prop_dist.normalizer[target_key]['mean']
        if prop_value is None:
            prop_value = 0.0

        if target_key in SYMPHONY_NORM:
            norm = SYMPHONY_NORM[target_key]
            prop_value = (prop_value * norm['std'] + norm['mean']) * norm.get('unit_factor', 1.0)
        prop_value = torch.tensor(prop_value, dtype=torch.float32)

        data = {
            'positions': padded_positions,
            'one_hot': padded_one_hot,
            'node_mask': node_mask,
            'n_nodes': n_nodes,
            'filename': os.path.basename(xyz_file),
            target_key: prop_value,
        }
        return data
    
    def _collate_batch(self, batch_data):
        """Collate individual examples into a batch."""
        batch_size = len(batch_data)
        
        # Get first example to determine shape
        first = batch_data[0]
        max_n_nodes = self.dataset_info['max_n_nodes']
        
        # Initialize tensors for the batch
        positions = torch.zeros((batch_size, max_n_nodes, 3), dtype=torch.float32)
        one_hot = torch.zeros((batch_size, max_n_nodes, len(self.dataset_info['atom_types'])))
        node_mask = torch.zeros((batch_size, max_n_nodes), dtype=torch.bool)
        
        # Get property key
        prop_key = self.target_property
        prop_values = torch.zeros((batch_size, 1), dtype=torch.float32)

        # Fill tensors
        for i, data in enumerate(batch_data):
            positions[i, :data['n_nodes']] = data['positions'][:data['n_nodes']]
            one_hot[i, :data['n_nodes']] = data['one_hot'][:data['n_nodes']]
            node_mask[i] = data['node_mask']
            prop_values[i] = data[prop_key]
        
        # Create edge mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask = edge_mask * diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)
        
        # Return batch in the expected format
        data = {
            'positions': positions.to(self.device),
            'atom_mask': node_mask.to(self.device),
            'edge_mask': edge_mask.to(self.device),
            'one_hot': one_hot.to(self.device),
            'filenames': [d['filename'] for d in batch_data],
            prop_key: prop_values.to(self.device)
        }
        return data
    
    def __iter__(self):
        """Return self as iterator."""
        return self
    
    def sample(self):
        """Sample a batch of data from XYZ files."""
        return self._load_xyz_batch()
    
    def __next__(self):
        """Get next batch."""
        if self.i < self.iterations:
            self.i += 1
            return self.sample()
        else:
            self.i = 0
            raise StopIteration
    
    def __len__(self):
        """Return length of iterator."""
        return self.iterations
    

class DiffusionDataloader:
    def __init__(self, args_gen, model, nodes_dist, prop_dist, device, unkown_labels=False,
                 batch_size=1, iterations=200, context=None):
        self.args_gen = args_gen
        self.model = model
        self.nodes_dist = nodes_dist
        self.prop_dist = prop_dist
        self.batch_size = batch_size
        self.iterations = iterations
        self.device = device
        self.unkown_labels = unkown_labels
        # TODO this assumes context is a scalar
        self.context = context
        self.dataset_info = get_dataset_info(self.args_gen.dataset, self.args_gen.remove_h)
        self.i = 0
        self.samples = []

        for key in self.args_gen.conditioning:
            print(f"Prop dist for {key}:", self.prop_dist.normalizer[key])

    def __iter__(self):
        return self

    def sample(self):
        nodesxsample = self.nodes_dist.sample(self.batch_size)
        if self.context:
            context = torch.ones((self.batch_size, 1), device=self.device) * self.context
        else:
            context = self.prop_dist.sample_batch(nodesxsample).to(self.device)
        one_hot, charges, x, node_mask = sample(self.args_gen, self.device, self.model,
                                                self.dataset_info, self.prop_dist, nodesxsample=nodesxsample,
                                                context=context)

        node_mask = node_mask.squeeze(2)
        context = context.squeeze(1)

        # edge_mask
        bs, n_nodes = node_mask.size()
        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.to(self.device)
        edge_mask *= diag_mask
        edge_mask = edge_mask.view(bs * n_nodes * n_nodes, 1)

        prop_key = self.prop_dist.properties[0]
        if self.unkown_labels:
            context[:] = self.prop_dist.normalizer[prop_key]['mean']
        else:
            context = context * self.prop_dist.normalizer[prop_key]['mad'] + self.prop_dist.normalizer[prop_key]['mean']
        data = {
            'positions': x.detach(),
            'atom_mask': node_mask.detach(),
            'edge_mask': edge_mask.detach(),
            'one_hot': one_hot.detach(),
            prop_key: context.detach()
        }

        # for key in data:
        #     print(f"{key}: {data[key].shape}")

        return data

    def __next__(self):
        if self.i < self.iterations:
            if len(self.samples) > self.i:
                out = self.samples[self.i]
            else:
                out = self.sample()
                self.samples.append(out)
            self.i += 1
            return out
        else:
            self.i = 0
            raise StopIteration

    def __len__(self):
        return self.iterations

    def save_samples(self, dataset_info):
        for i, sample in enumerate(self.samples):
            vis.save_xyz_file(
                'outputs/%s/prop%s/' % (self.args_gen.exp_name, self.context), sample["one_hot"], None, 
                sample["positions"], dataset_info, i * self.batch_size, name='conditional', node_mask=sample["atom_mask"])


@torch.no_grad()
def collect_predictions(classifier, loader, mean, mad, property, device):
    """Run the classifier over `loader` and return per-sample predictions.

    Returns dict with keys 'filenames', 'targets', 'preds', and 'loss'
    (all denormalized into the original property units)."""
    classifier.eval()
    filenames, targets, preds = [], [], []
    total_loss = 0.0
    total_count = 0
    for data in loader:
        batch_size, n_nodes, _ = data['positions'].size()
        atom_positions = data['positions'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        atom_mask = data['atom_mask'].view(batch_size * n_nodes, -1).to(device, torch.float32)
        edge_mask = data['edge_mask'].view(batch_size * n_nodes * n_nodes, 1).to(device, torch.float32)
        nodes = data['one_hot'].to(device, torch.float32).view(batch_size * n_nodes, -1)
        edges = prop_utils.get_adj_matrix(n_nodes, batch_size, device)
        label = data[property].to(device, torch.float32)

        pred = classifier(h0=nodes, x=atom_positions, edges=edges, edge_attr=None,
                          node_mask=atom_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        pred_denorm = mad * pred + mean

        total_loss += torch.nn.functional.l1_loss(pred_denorm, label.view_as(pred_denorm),
                                                  reduction='sum').item()
        total_count += pred_denorm.numel()

        filenames.extend(data.get('filenames', [''] * batch_size))
        targets.extend(label.view(-1).cpu().tolist())
        preds.extend(pred_denorm.view(-1).cpu().tolist())

    return {
        'filenames': filenames,
        'targets': targets,
        'preds': preds,
        'loss': total_loss / max(total_count, 1),
    }


def main_quantitative(args):
    # Get classifier
    #if args.task == "numnodes":
    #    class_dir = args.classifiers_path[:-6] + "numnodes_%s" % args.property
    #else:
    class_dir = args.classifiers_path
    classifier = get_classifier(class_dir).to(args.device)

    # Get generator and dataloader used to train the generator and evalute the classifier
    args_gen = get_args_gen(args.generators_path)

    # Careful with this -->
    if not hasattr(args_gen, 'diffusion_noise_precision'):
        args_gen.normalization_factor = 1e-4
    if not hasattr(args_gen, 'normalization_factor'):
        args_gen.normalization_factor = 1
    if not hasattr(args_gen, 'aggregation_method'):
        args_gen.aggregation_method = 'sum'

    dataloaders = get_dataloader(args_gen)
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path, dataloaders,
                                                    args.device, args_gen, property_norms)

    # Create a dataloader with the generator

    # The classifier's target property may differ from the generator's
    # conditioning (e.g. evaluating a relative_atomic_energy classifier on
    # samples from a gap-conditioned generator). Compute mean/mad for it on
    # the same split the classifier was trained on. For relative_atomic_energy
    # we fit Symphony's per-element linear baseline on train and apply it to
    # all splits as a precomputed column.
    from qm9 import utils as qm9_utils
    if args.property == 'relative_atomic_energy':
        relenergy_weights = qm9_utils.fit_relenergy_baseline(dataloaders['train'].dataset)
        for split in ('train', 'valid', 'test'):
            if split in dataloaders:
                qm9_utils.add_relative_atomic_energy(dataloaders[split].dataset, relenergy_weights)
    if args.property not in property_norms:
        extra = compute_mean_mad(dataloaders, [args.property], args_gen.dataset)
        property_norms[args.property] = extra[args.property]

    mean, mad = property_norms[args.property]['mean'], property_norms[args.property]['mad']

    if args.task == 'edm':
        diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist,
                                                   args.device, batch_size=args.batch_size,
                                                   iterations=args.iterations, context=args.context)
        print(f"EDM: We evaluate the classifier on {args.iterations * args.batch_size} generated samples")
        loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
        print("Loss classifier on Generated samples: %.4f" % loss)
        print("Saving samples")
        diffusion_dataloader.save_samples(dataset_info)
    elif args.task == 'xyz':
        xyz_dirs = args.xyz_dir if isinstance(args.xyz_dir, list) else [args.xyz_dir]
        for xyz_dir in xyz_dirs:
            print(f"=== XYZ dir: {xyz_dir} ===")
            xyz_dataloader = XYZDataloader(
                args_gen=args_gen, xyz_dir=xyz_dir, device=args.device,
                batch_size=args.batch_size, iterations=args.iterations,
                prop_dist=prop_dist, target_property=args.property,
                cond_keys=(args.cond_keys or [args.property]))
            print(f"XYZ: We evaluate the classifier on {len(xyz_dataloader.xyz_files)} generated samples")
            results = collect_predictions(classifier, xyz_dataloader, mean, mad, args.property, args.device)
            print("Loss classifier on Generated samples: %.4f" % results['loss'])

            csv_path = os.path.join(xyz_dir, f'predictions_{args.property}.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'target', 'pred'])
                for fn, tgt, pred in zip(results['filenames'], results['targets'], results['preds']):
                    writer.writerow([fn, tgt, pred])
            print(f"Saved predictions to {csv_path}")
    elif args.task == 'qm9_second_half':
        print("qm9_second_half: We evaluate the classifier on QM9")
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on qm9_second_half: %.4f" % loss)
    elif args.task == 'naive':
        print("Naive: We evaluate the classifier on QM9")
        length = dataloaders['train'].dataset.data[args.property].size(0)
        idxs = torch.randperm(length)
        dataloaders['train'].dataset.data[args.property] = dataloaders['train'].dataset.data[args.property][idxs]
        loss = test(classifier, 0, dataloaders['train'], mean, mad, args.property, args.device, args.log_interval,
                    args.debug_break)
        print("Loss classifier on naive: %.4f" % loss)
    #elif args.task == 'numnodes':
    #    print("Numnodes: We evaluate the numnodes classifier on EDM samples")
    #    diffusion_dataloader = DiffusionDataloader(args_gen, model, nodes_dist, prop_dist, device,
    #                                               batch_size=args.batch_size, iterations=args.iterations)
    #    loss = test(classifier, 0, diffusion_dataloader, mean, mad, args.property, args.device, 1, args.debug_break)
    #    print("Loss numnodes classifier on EDM generated samples: %.4f" % loss)


def save_and_sample_conditional(
    args,
    device, 
    model, 
    prop_dist, 
    dataset_info, 
    epoch=0, 
    id_from=0, 
    visualize=False, 
    n_nodes=19, 
    n_frames=100,
):
    one_hot, charges, x, node_mask = sample_sweep_conditional(
        args, 
        device, 
        model, 
        dataset_info, 
        prop_dist, 
        n_nodes=n_nodes, 
        n_frames=n_frames,
    )

    vis.save_xyz_file(
        'outputs/%s/analysis/run%s/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    if visualize:
        vis.visualize_chain("outputs/%s/analysis/run%s/" % (args.exp_name, epoch), dataset_info,
                        wandb=None, mode='conditional', spheres_3d=True)

    return one_hot, charges, x


def reorganize_sweep_outputs(output_dir):
    """Reorganizes outputs by frame #, as opposed to sweep #"""
    new_dir = os.path.join(output_dir, "xyzs_by_frame")

    for run_dir in os.listdir(os.path.join(output_dir, "analysis")):
        run_dir_full = os.path.join(output_dir, "analysis", run_dir)
        if not os.path.isdir(run_dir_full):
            continue
        for output_file in os.listdir(run_dir_full):
            filename = os.path.splitext(output_file)[0]
            prop = filename.split('_')[1]
            prop_dir = os.path.join(new_dir, prop)
            if not os.path.isdir(prop_dir):
                os.makedirs(prop_dir, exist_ok=False)

            with open(os.path.join(run_dir_full, output_file), 'r') as f:
                content = f.read()
            with open(os.path.join(prop_dir, f"{run_dir}.xyz"), 'w') as f:
                f.write(content)


def main_qualitative(args):
    args_gen = get_args_gen(args.generators_path)
    dataloaders = get_dataloader(args_gen)
    print("Getting property info...")
    property_norms = compute_mean_mad(dataloaders, args_gen.conditioning, args_gen.dataset)
    print("Getting generator...")
    model, nodes_dist, prop_dist, dataset_info = get_generator(args.generators_path,
                                                               dataloaders, args.device, args_gen,
                                                               property_norms)
    for key in prop_dist.distributions:
        min_val, max_val = prop_dist.distributions[key][args.n_nodes]['params']
        print(f"Property {key}: Evaluating at {args.n_frames} values in range [{min_val}, {max_val}]")

    for i in tqdm.tqdm(range(args.n_sweeps), desc="Sampling sweep"):
        save_and_sample_conditional(
            args_gen, 
            device, 
            model, 
            prop_dist, 
            dataset_info, 
            epoch=i, 
            id_from=0,
            n_nodes=args.n_nodes,
            n_frames=args.n_frames,
        )
    
    reorganize_sweep_outputs(f"outputs/{args_gen.exp_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='debug_alpha')
    parser.add_argument('--generators_path', type=str, default='outputs/exp_cond_alpha_pretrained')
    parser.add_argument('--classifiers_path', type=str, default='qm9/property_prediction/outputs/exp_class_alpha_pretrained')
    parser.add_argument('--property', type=str, default='alpha',
                        help="'alpha', 'homo', 'lumo', 'gap', 'mu', 'Cv'")
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--debug_break', type=eval, default=False,
                        help='break point or not')
    parser.add_argument('--log_interval', type=int, default=5,
                        help='break point or not')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='break point or not')
    parser.add_argument('--iterations', type=int, default=20,
                        help='break point or not')
    parser.add_argument('--task', type=str, default='qualitative',
                        help='naive, edm, qm9_second_half, qualitative')
    parser.add_argument('--n_sweeps', type=int, default=10,
                        help='number of sweeps for the qualitative conditional experiment')
    parser.add_argument('--n_nodes', type=int, default=19,
                        help='number of nodes in each generated molecule')
    parser.add_argument('--n_frames', type=int, default=100,
                        help='number of values of the given property to evaluate at')
    parser.add_argument('--context', type=float, default=None,
                        help='value of property to condition on')
    parser.add_argument(
        "--xyz_dir", type=str, nargs='+', default=[],
        help="One or more directories containing XYZ files."
    )
    parser.add_argument(
        "--cond_keys", type=str, nargs='+', default=None,
        help=("Order of property names matching elements of the Symphony "
              "conditioning tensor in xyz files (e.g. 'gap "
              "relative_atomic_energy'). Defaults to [--property].")
    )
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.task == 'qualitative':
        main_qualitative(args)
    else:
        main_quantitative(args)
