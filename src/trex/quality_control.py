from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import re
import seaborn as sns
from tinyalign import hamming_distance


def load_reads(data_dir: pathlib.Path):
    """Loads saved reads into a DataFrame"""
    READ_DIR = data_dir / 'reads.txt'
    return pd.read_csv(READ_DIR, delimiter='\t')


def load_molecules(data_dir: pathlib.Path):
    """Loads saved molecules before correcting into a DataFrame."""
    MOLS_DIR = data_dir / 'molecules.txt'
    return pd.read_csv(MOLS_DIR, delimiter='\t')


def load_molecules_corrected(data_dir: pathlib.Path):
    """Loads saved molecules after correcting into a DataFrame."""
    MOLS_DIR = data_dir / 'molecules_corrected.txt'
    return pd.read_csv(MOLS_DIR, delimiter='\t')


def read_quality(reads: pd.DataFrame) -> plt.Axes:
    """Plot histogram of how many time a molecule was read."""
    count_reads = reads.groupby(['#cell_id', 'umi']).agg('count')

    ax = sns.histplot(data=count_reads, x='clone_id', discrete=True, log=True)
    plt.xlabel('Number of reads')
    plt.title('Number of reads per molecule')

    txt = 'This plot shows how many times a molecule has been read. \n' \
          'The more reads per molecule, the better.'
    plt.text(0, -0.3, txt, transform=ax.transAxes, size=12)
    return ax


def length_read(molecules: pd.DataFrame) -> plt.Axes:
    """Plot histogram of how many bases were adequately read per barcode."""
    molecules['stripped_barcode'] = molecules.clone_id.apply(
        lambda x: re.sub("[-0]", "", x))

    ax = sns.histplot(molecules.stripped_barcode.apply(len), discrete=True,
                      log=True)
    plt.xlabel('Number of detected bases')
    plt.title('Length of computed molecules')
    txt = 'This plot shows how many bases have been read per molecule. \n' \
          'Ideally all 30 bases have been read. If very few bases are read, \n'\
          'we can not be sure how to complete the missing bases.'
    plt.text(0, -0.3, txt, transform=ax.transAxes, size=12)
    return ax


def molecules_per_cell(molecules: pd.DataFrame) -> plt.Axes:
    """Plot histogram of how many molecules were detected per cell."""
    count_reads = molecules.groupby(['#cell_id']).umi.agg('count')

    ax = sns.histplot(count_reads.values, discrete=True, log=True)
    plt.xlabel('Molecules per cell')
    plt.title('Number of molecules')
    txt = 'This plot shows how many molecules were found per cell. \n' \
          'Cell that have a single molecule might be filtered out later. \n' \
          'Cells that have too many might be result of non-removed \n doublets.'
    plt.text(0, -0.3, txt, transform=ax.transAxes, size=12)
    return ax


def molecules_per_barcode(molecules: pd.DataFrame) -> plt.Axes:
    """Plot histogram of how many molecules were detected per viral barcode."""
    count_reads = molecules.groupby(['clone_id']).umi.agg('count')

    ax = sns.histplot(count_reads.values, discrete=True, log=True)
    plt.xlabel('Molecules per barcode')
    plt.title('Number of molecules')
    txt = 'This plot shows how many molecules were found per barcode. \n' \
          'Barcodes that appear a few times might not be found in more \n' \
          'cells. Barcodes that have too many might be result of \n' \
          'contamination or alignment problems or big clones.'
    plt.text(0, -0.3, txt, transform=ax.transAxes, size=12)
    return ax


def unique_barcodes_per_cell(molecules: pd.DataFrame) -> plt.Axes:
    """Plot histogram of how many unique barcodes were detected per cell."""
    count_reads = molecules.groupby('#cell_id').clone_id.unique().apply(len)

    ax = sns.histplot(count_reads.values, discrete=True, log=True)
    plt.xlabel('Unique barcodes per cell')
    plt.title('Number of unique barcodes')
    txt = 'This plot shows how many unique barcodes were detected per cell.\n' \
          'Cells with many unique barcodes show either lots of infection \n' \
          'events or possible unfiltered doublets.'
    plt.text(0, -0.3, txt, transform=ax.transAxes, size=12)
    return ax


def hamming_distance_histogram(molecules: pd.DataFrame,
                               ignore_incomplete=True) -> plt.Axes:
    """Plot histogram of Hamming distance between barcodes. ignore_incomplete is
     set to True by default and it removes incomplete barcodes."""
    if ignore_incomplete:
        molecules = molecules[~molecules.clone_id.str.contains('-|0')]
    this_clone_ids = molecules.clone_id.unique()
    hamming_distances = np.empty([len(this_clone_ids)] * 2)

    def my_iter(barcode_list):
        for inds in combinations(np.arange(len(barcode_list)), 2):
            yield inds, barcode_list[inds[0]], barcode_list[inds[1]]

    # Hamming distance function
    def is_similar(args):
        inds, s, t = args
        if not ignore_incomplete:
            bad_chars = {'-', '0'}
            if bad_chars & set(s) or bad_chars & set(t):
                # Remove suffix and/or prefix where sequences do not overlap
                s = s.lstrip("-0")
                t = t[-len(s):]
                t = t.lstrip("-0")
                s = s[-len(t):]
                s = s.rstrip("-0")
                t = t[: len(s)]
                t = t.rstrip("-0")
                s = s[: len(t)]

        return inds, hamming_distance(s, t)

    for ind, val in map(is_similar, my_iter(this_clone_ids)):
        hamming_distances[ind[0], ind[1]] = val

    vals = hamming_distances[np.triu_indices_from(hamming_distances, 1)]
    ax = sns.histplot(vals, discrete=True, log=True)

    ax.set_title('Hamming Distance Histogram')
    ax.set_xlabel('Hamming Distance')

    return ax
