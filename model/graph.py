import torch
import numpy as np
from pathlib import Path
from typing import Union
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from torch_geometric.data import Data, InMemoryDataset
from common.utils import tensor2irreps


def contains_none(obj):
    if obj is None:
        return True

    if isinstance(obj, dict):
        return any(contains_none(v) for v in obj.values())

    if isinstance(obj, (list, tuple, set)):
        return any(contains_none(v) for v in obj)

    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            return any(contains_none(v) for v in obj.tolist())
        else:
            return False

    if torch.is_tensor(obj):
        return False

    return False


class CrystalGraphDataset(InMemoryDataset):
    def __init__(
        self,
        target: str, 
        dict_data: dict,
        path_save: Path,
        datasize: int=9999,
        rmax: Union[int, float] = 10,
        lmax: int = 2,
        dtype=torch.float32,
        dict_atom_init: Union[dict, None] = None,
        transform=None,
        data_augmentation: bool = False,
    ):
        super().__init__(transform=transform)
        self.target = target
        self.rmax = rmax
        self.dtype = dtype
        self.lmax = lmax
        self.data_augmentation_flag = data_augmentation
        self.dict_atom_init = dict_atom_init

        self.matkeys = []
        self.data_list = []
        for matname, data in dict_data.items():
            assert "structure" in data
            assert isinstance(data["structure"], Structure)
            assert "prop" in data
            
            data_i = self.get_graph(
                matname=matname,
                dict_prop=data,
            )

            if not data_i:
                continue
            
            has_none = contains_none(data_i.to_dict())
            if has_none:
                print(matname)
                continue

            self.data_list.append(data_i)

            if len(self.data_list) == datasize:
                break

        self.data, self.slices = self.collate(self.data_list)
        self.save(self.data_list, path_save)

    
    def get_graph(
        self,
        matname: str,
        dict_prop: dict,
    ):
        structure = dict_prop['structure']

        attrs_node, attrs_edge = [], []
        indexes_edge, rijs_relative = [], []
        for i, site in enumerate(structure):
            if site.specie.Z not in self.dict_atom_init:
                return False
            attrs_node.append(self.get_node_attr(site.specie.Z))

            neis = structure.get_neighbors(site, r=self.rmax)
            for nei in neis:
                xij = nei.coords - site.coords
                dij = np.linalg.norm(xij)
                xij_rel = xij / (dij + 1e-8)

                rijs_relative.append(xij_rel.tolist())
                attrs_edge.append([dij])
                indexes_edge.append([i, nei.index])

        attrs_node = torch.tensor(attrs_node, dtype=self.dtype)
        attrs_edge = torch.tensor(attrs_edge, dtype=self.dtype)
        indexes_edge = torch.tensor(indexes_edge, dtype=torch.long)
        rijs_relative = torch.tensor(rijs_relative, dtype=self.dtype)
        eqv_atom_idx = SpacegroupAnalyzer(structure).get_symmetry_dataset()["equivalent_atoms"]
        unq_eqv_atom_idx = sorted(list(set(eqv_atom_idx)))
        pooling_mask = [(atm_idx in unq_eqv_atom_idx) for atm_idx in range(structure.num_sites)]
        # print("pooling mask")
        # print(pooling_mask)

        data = Data(
            matname=matname,
            structure=structure,
            prop=tensor2irreps(dict_prop["prop"].reshape(-1, 3, 3), target=self.target),
            attrs_node=attrs_node,
            indexes_edge=indexes_edge.t().contiguous(),
            attrs_edge=attrs_edge,
            rijs_relative=rijs_relative,
            pooling_mask=pooling_mask,
        )
        return data


    def get_node_attr(self, Z: int):
        if self.dict_atom_init is None:
            return [Z]
        else:
            return self.dict_atom_init[Z]
