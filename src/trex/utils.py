import logging
import pandas as pd
from typing import List

from ..molecule import Molecule


class NiceFormatter(logging.Formatter):
    """
    Do not prefix "INFO:" to info-level log messages (but do it for all other
    levels).

    Based on http://stackoverflow.com/a/9218261/715090 .
    """

    def format(self, record):
        if record.levelno != logging.INFO:
            record.msg = "{}: {}".format(record.levelname, record.msg)
        return super().format(record)


def molecule_list_2_df(mol_list: List[Molecule]) -> pd.DataFrame:
    """Cast list of Molecules to pandas DataFrame"""
    mol_dict = {'cell_id': [], 'umi': [], 'barcode': [], 'read_count': []}

    for mol in mol_list:
        mol_dict['cell_id'].append(mol.cell_id)
        mol_dict['umi'].append(mol.umi)
        mol_dict['barcode'].append(mol.clone_id)
        mol_dict['read_count'].append(getattr(mol, 'read_count', -1))

    return pd.DataFrame(mol_dict)


def df_2_molecule_list(df: pd.DataFrame) -> List[Molecule]:
    """Cast pandas DataFrame to list of Molecules"""
    return [Molecule(umi=mol.umi, cell_id=mol.cell_id, clone_id=mol.barcode,
                     read_count=mol.get('read_count', -1)
                     ) for r, mol in df.iterrows()]
