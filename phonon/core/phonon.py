import numpy as np
from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from ase.optimize import BFGS, FIRE
from ase.constraints import FixSymmetry
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
import seekpath
import matplotlib.pyplot as plt
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_FORCE_CONSTANTS
from pymatgen.core import Structure, Lattice
import pandas as pd
import os
import torch
from pathlib import Path


EVTOJ=1.60217733E-19    # 1 eV in Joule
AMTOKG=1.6605402E-27    # 1 atomic mass unit ("proton mass") in kg
RYTOEV=13.605826        # 1 Ry in Ev
AUTOA=0.529177249       # 1 a.u. of length (Bohr) in Angstroem

FELECT = 2*RYTOEV*AUTOA
EDEPS = 4 * np.pi * FELECT
SI_CONV = np.sqrt(EVTOJ / AMTOKG) * 1E10    # 1e10 is angstrom to meture

EDEPS = 180.953062045845

META_TOKEN = "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"

POTENTIALS = ['sevennet-mf-ompa', "esen-uma-s-1-omat", "orbv3",'mace']


def reshape_hessian(hessian: np.ndarray) -> np.ndarray:
    """
    4次元のHessian行列 (n, n, 3, 3) を2次元行列 (n*3, n*3) に変換します。
    """
    n = hessian.shape[0]
    new_hes = np.empty((n * 3, n * 3))
    for i in range(n):
        for j in range(n):
            new_hes[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = hessian[i, j, :, :]
    return new_hes


class Phonon(object):
    def __init__(
        self, 
        st, 
        dir_save_phonon: Path,
    ):
        self.phonopy = Phonopy(
            st,
            supercell_matrix=[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ],
        )
        
        self.dir_save_phonon = dir_save_phonon
    
    def get_hessian(self):
        self.hessian = self.phonopy.force_constants # (N,N,3,3)
        self.dynmat = self.phonopy.get_dynamical_matrix_at_q([0, 0, 0])

        np.save(f"{self.dir_save_phonon}/hessian.npy", self.hessian)
        np.save(f"{self.dir_save_phonon}/dynmat.npy", self.dynmat)
        # if len(self.hessian.shape) == 4:
        #     np.savetxt(fname=f"{self.dir_save_phonon}/hessian.txt", X=reshape_hessian(self.hessian))
        # else:
        #     np.savetxt(fname=f"{self.dir_save_phonon}/hessian.txt", X=self.hessian)
        # np.savetxt(fname=f"{self.dir_save_phonon}/dynmat.txt", X=self.dynmat)

    def get_freqs(self):
        self.freqs = self.phonopy.get_frequencies([0, 0, 0])
        np.savetxt(f"{self.dir_save_phonon}/freqs.txt", self.freqs)
        print(f"{self.dir_save_phonon}/freqs.txt")
    
    def recip_from_cell(self, cell):
        """Real-space cell (3x3, row vectors) -> reciprocal basis (3x3, columns)"""
        A = np.array(cell, dtype=float)
        return 2 * np.pi * np.linalg.inv(A).T
    
    def symmetrize(self, mat):
        return 0.5*(mat + mat.T)

    def pymatgen_to_ase(self, st:Structure):
        st_ase = AseAtomsAdaptor.get_atoms(st)
        return st_ase

    def ase_to_pymatgen(self, st_ase:Atoms):
        st = AseAtomsAdaptor.get_structure(st_ase)
        return st
    
    def phonopy_atoms_to_structure(self, p_atoms):
        lattice = Lattice(p_atoms.cell)
        return Structure(
            lattice,
            species=p_atoms.symbols,
            coords=p_atoms.positions,
            coords_are_cartesian=True,
        )


class Phonon_IP(Phonon):
    def __init__(self,
        model_name:str,
        st_pymat: Structure, 
        dir_save_phonon: Path, 
    ):
        self.st_ase = self.pymatgen_to_ase(st_pymat)
        self.st_phonopy = PhonopyAtoms(
            symbols=self.st_ase.get_chemical_symbols(),
            positions=self.st_ase.get_positions(),
            cell=self.st_ase.get_cell(), 
        )
        super().__init__(
            st=self.st_phonopy,
            dir_save_phonon=dir_save_phonon,
        )
        
        if model_name not in POTENTIALS:
            raise ValueError(f"x must be one of {POTENTIALS}")

        # set ASE-calculator as calc
        if model_name == "esen-uma-s-1-omat":
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
            calc = FAIRChemCalculator(predictor, task_name="omat")

        elif model_name == "orbv3":
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
            device="cuda"
            orbff = pretrained.orb_v3_conservative_inf_omat(
                device=device,
                precision="float32-high",
            )
            calc = ORBCalculator(orbff, device=device)

        elif model_name == 'sevennet-mf-ompa':
            from sevenn.calculator import SevenNetCalculator
            calc = SevenNetCalculator(model='7net-mf-ompa', modal='mpa', devide="cuda:2")
        
        elif model_name == 'mace':
            from mace.calculators import mace_mp
            calc = mace_mp(model="medium-mpa-0", dispersion=False, default_dtype="float32", device='cuda')

        else:
            calc = None

        self.calc = calc
        self.pot = model_name

    def optimize_structure(self, tol = 0.001, max_step=1000):
        self.st_ase.calc = self.calc
        self.st_ase.set_constraint(FixSymmetry(self.st_ase))
        opt = BFGS(self.st_ase)
        convergence = opt.run(fmax=tol,steps=max_step)

        if convergence:
            return self.st_ase
        else:
            opt = BFGS(self.st_ase, maxstep=0.05) # 0.2 in default
            convergence = opt.run(fmax=tol, steps=max_step)

            if convergence:
                return self.st_ase
            else:
                opt = FIRE(self.st_ase)  # 0.2 in default
                convergence = opt.run(fmax=tol, steps=max_step)

                if convergence:
                    return self.st_ase
                else:
                    opt = BFGS(self.st_ase, maxstep=0.05)  # 0.2 in default
                    convergence = opt.run(fmax=tol, steps=max_step)

                    if convergence:
                        return self.st_ase
                    else:
                        return False

    def calc_hessian(
        self, 
        disp = 0.01, 
    ):
        self.phonopy.generate_displacements(distance=disp)
        print("generate_displacements done")

        sets_of_forces = []
        supercells = self.phonopy.supercells_with_displacements

        for cell in supercells:
            st_ase_cell = Atoms(
                symbols=cell.symbols,
                positions=cell.positions,
                cell=cell.cell,
                pbc = True,
            )

            st_ase_cell.calc = self.calc
            forces = st_ase_cell.get_forces()
            sets_of_forces.append(forces)
        print("get_forces done")

        sets_of_forces = np.array(sets_of_forces)
        self.phonopy.forces = sets_of_forces

        self.phonopy.produce_force_constants()
        print("produce_force_constants done")

        self.get_hessian()


class Phonon_VASP(Phonon):
    def __init__(
        self, 
        poscar, 
        dir_save_phonon,
    ):
        uc = read_vasp(poscar)
        super().__init__(
            st=uc, 
            dir_save_phonon=dir_save_phonon, 
        )
        print("Phonon VASP")
    
    def calc_hessian(
        self, 
        fc_file, 
    ):
        fc = parse_FORCE_CONSTANTS(filename=fc_file)
        self.phonopy.force_constants = fc
        
        self.get_hessian()
