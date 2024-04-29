from os.path import join as join
import urllib.request

import ase
import numpy as np
import pandas as pd
import torch
from sh import gunzip

import logging, os, urllib

from qm9.data.prepare.process import process_xyz_files, process_xyz_tmqm
from qm9.data.prepare.utils import download_data, clone_url, is_int, cleanup_file

tmqm_url = "https://github.com/bbskjelstad/tmqm.git"

def download_dataset_tmqm(datadir, dataname, splits=None, calculate_thermo=True, exclude=True, cleanup=True):
    """
    Downloads the TMQM dataset.
    """

    # Define directory for which data will be output.
    tmqmdir = join(*[datadir, dataname])

    if os.path.exists(tmqmdir):
        logging.info('Using pre-downloaded data')
        return

    # Important to avoid a race condition
    os.makedirs(tmqmdir, exist_ok=True)

    logging.info('Downloading and processing TMQM dataset. Output will be in directory: {}.'.format(tmqmdir))

    tmqm_repo_dir = tmqmdir
    tmqm_data_dir = os.path.join(tmqm_repo_dir, 'tmqm/data')

    clone_url(tmqm_url, tmqm_repo_dir)
    tmqm_xyzs_path = os.path.join(tmqmdir, "tmqm/xyz")
    if not os.path.exists(tmqm_xyzs_path):
        os.makedirs(tmqm_xyzs_path)
    for i in range(1, 3):
        gz_path = os.path.join(tmqm_data_dir, f"tmQM_X{i}.xyz.gz")
        logging.info(f"Unzipping {gz_path}...")
        gunzip(gz_path)

        mol_file = os.path.join(tmqm_data_dir, f"tmQM_X{i}.xyz")
        with open(mol_file, "r") as f:
            all_xyzs = f.read().split("\n\n")
            for xyz_n, xyz in enumerate(all_xyzs):
                if xyz == "":
                    continue
                xyz_lines = xyz.split("\n")
                assert len(xyz_lines) == int(xyz_lines[0]) + 2
                with open(os.path.join(tmqm_xyzs_path, f"X{i}_{xyz_n}.xyz"), "w") as f:
                    f.write(xyz)

    # Process GDB9 dataset, and return dictionary of splits
    tmqm_data = {}
    tmqm_props = pd.read_csv(os.path.join(tmqm_data_dir, 'tmQM_y.csv'), sep=';')
    tmqm_props.drop(columns='CSD_code', inplace=True)
    if splits is None:
        splits = {'train': np.arange(75000), 'valid': np.arange(75000, 80000), 'test': np.arange(80000, 86665)}
    for split, split_idx in splits.items():
        tmqm_data[split] = process_xyz_files(
            tmqm_xyzs_path, process_xyz_tmqm, file_idx_list=split_idx, stack=True, prop_df=tmqm_props)

    # # Subtract thermochemical energy if desired.
    # if calculate_thermo:
    #     # Download thermochemical energy from GDB9 dataset, and then process it into a dictionary
    #     therm_energy = get_thermo_dict(tmqm_data_dir, cleanup)

    #     # For each of train/validation/test split, add the thermochemical energy
    #     for split_idx, split_data in tmqm_data.items():
    #         tmqm_data[split_idx] = add_thermo_targets(split_data, therm_energy)

    # Save processed GDB9 data into train/validation/test splits
    logging.info('Saving processed data:')
    for split, data in tmqm_data.items():
        savedir = join(tmqmdir, split+'.npz')
        np.savez_compressed(savedir, **data)

    logging.info('Processing/saving complete!')


def get_thermo_dict(gdb9dir, cleanup=True):
    """
    Get dictionary of thermochemical energy to subtract off from
    properties of molecules.

    Probably would be easier just to just precompute this and enter it explicitly.
    """
    # Download thermochemical energy
    logging.info('Downloading thermochemical energy.')
    gdb9_url_thermo = 'https://springernature.figshare.com/ndownloader/files/3195395'
    gdb9_txt_thermo = join(gdb9dir, 'atomref.txt')

    urllib.request.urlretrieve(gdb9_url_thermo, filename=gdb9_txt_thermo)

    # Loop over file of thermochemical energies
    therm_targets = ['zpve', 'U0', 'U', 'H', 'G', 'Cv']

    # Dictionary that
    id2charge = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}

    # Loop over file of thermochemical energies
    therm_energy = {target: {} for target in therm_targets}
    with open(gdb9_txt_thermo) as f:
        for line in f:
            # If line starts with an element, convert the rest to a list of energies.
            split = line.split()

            # Check charge corresponds to an atom
            if len(split) == 0 or split[0] not in id2charge.keys():
                continue

            # Loop over learning targets with defined thermochemical energy
            for therm_target, split_therm in zip(therm_targets, split[1:]):
                therm_energy[therm_target][id2charge[split[0]]
                                           ] = float(split_therm)

    # Cleanup file when finished.
    cleanup_file(gdb9_txt_thermo, cleanup)

    return therm_energy


def add_thermo_targets(data, therm_energy_dict):
    """
    Adds a new molecular property, which is the thermochemical energy.

    Parameters
    ----------
    data : ?????
        QM9 dataset split.
    therm_energy : dict
        Dictionary of thermochemical energies for relevant properties found using :get_thermo_dict:
    """
    # Get the charge and number of charges
    charge_counts = get_unique_charges(data['charges'])

    # Now, loop over the targets with defined thermochemical energy
    for target, target_therm in therm_energy_dict.items():
        thermo = np.zeros(len(data[target]))

        # Loop over each charge, and multiplicity of the charge
        for z, num_z in charge_counts.items():
            if z == 0:
                continue
            # Now add the thermochemical energy per atomic charge * the number of atoms of that type
            thermo += target_therm[z] * num_z

        # Now add the thermochemical energy as a property
        data[target + '_thermo'] = thermo

    return data


def get_unique_charges(charges):
    """
    Get count of each charge for each molecule.
    """
    # Create a dictionary of charges
    charge_counts = {z: np.zeros(len(charges), dtype=int)
                     for z in np.unique(charges)}

    # Loop over molecules, for each molecule get the unique charges
    for idx, mol_charges in enumerate(charges):
        # For each molecule, get the unique charge and multiplicity
        for z, num_z in zip(*np.unique(mol_charges, return_counts=True)):
            # Store the multiplicity of each charge in charge_counts
            charge_counts[z][idx] = num_z

    return charge_counts
