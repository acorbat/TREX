import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import re
import seaborn as sns


def load_reads(data_dir: pathlib.Path):
    """Loads saved reads into a DataFrame"""
    READ_DIR = data_dir / 'reads.txt'
    return pd.read_csv(READ_DIR, delimiter='\t')


def load_molecules(data_dir: pathlib.Path):
    """Loads saved reads into a DataFrame."""
    MOLS_DIR = data_dir / 'molecules.txt'
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

