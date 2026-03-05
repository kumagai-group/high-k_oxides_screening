import numpy as np

from pathlib import Path
from pymatgen.core.lattice import Lattice
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from typing import Union, Literal

EV2J=1.60217733E-19 # 1 eV in Joule
AM2KG=1.6605402E-27 # 1 atomic mass unit ("proton mass") in kg
RY2EV=13.605826 # 1 Ry in Ev
AU2A=0.529177249 # 1 a.u. of length (Bohr) in Angstroem

FELECT = 2*RY2EV*AU2A
EDEPS = 4 * np.pi * FELECT
SI_CONV = np.sqrt(EV2J / AM2KG) * 1E10 # 1e10 is angstrom to metre

""" 周波数²に変換するための係数 """
# 変換係数: sqrt(eV / (Å²·amu)) → Hz
CONV_FACTOR = np.sqrt(EV2J / AM2KG) * 1E10  # [Hz]
# → 周波数²にしたいので、CONV_FACTOR² が必要
CONV_FACTOR_SQ = CONV_FACTOR ** 2  # 単位変換: eV/Å²/amu → (Hz)²


def mlp_to_thz(mlp_eigenval: float):
    # 固有値（単位：eV/Å²/amu）→ (Hz)² → (THz)² → THz
    freq_squared_Hz2 = mlp_eigenval * (1.60218e-19) / (1.66054e-27 * (1e-10)**2) / (4 * np.pi**2)

    freq_squared_THz2 = freq_squared_Hz2 * 1e-24

    return w2_to_w(freq_squared_THz2)

def w2_to_w(w2: float):
    # (Hz)² → (THz)² → THz
    # 周波数の符号は変換しない！
    return np.sign(w2) * np.sqrt(abs(w2))

def symmetrize(hessian):
    return 0.5 * (hessian + hessian.T)

def acoustic_score(eigvec, num_atoms, tol=1e-3):
    """各原子の変位方向がほぼ等しいかどうかを確認"""
    eigvec = eigvec.reshape((num_atoms, 3))  # (N, 3)
    for i in range(eigvec.shape[0]):
        direction = eigvec[i]
        direction_norm = np.linalg.norm(direction)
        if direction_norm > tol:
            direction /= direction_norm
            break

    scores = []
    for vec in eigvec:
        norm = np.linalg.norm(vec)
        if norm < tol:
            continue  # 零変位の原子は無視
        scores.append(np.linalg.norm((vec/norm) - direction))

    return np.mean(scores)

def inv_with_cutlowfreq(mat, num_atoms):
    threshold = 1e-3  # in VASP

    # numpy version
    w, v = np.linalg.eigh(mat)  # w:eigenvalues sorted in ascending order, v:eigenvectors
    # print("w: \n", w)

    # 音響モードを検出（全原子が同方向に動いているかで判定）
    # acoustic_scores = [acoustic_score(v[:, i], num_atoms) for i in range(v.shape[1])]
    # acoustic_indices = np.argsort(acoustic_scores)[:3]  # 最も音響的な3つのモードを選択
    # print(np.sort(acoustic_scores)[:3])
    # 該当する固有値をゼロに設定
    # w[acoustic_indices] = 0.0

    w[0:3] = 0  # acoustic mode
    w_inv = np.where(w > threshold, 1 / w, 0)  # take inverse of w, w lower than 1e-3 is reduced to zero
    A_pinv = (v * w_inv) @ v.T  # calculate pseudo inverse matrix, V D^(-1) V^T

    return A_pinv

def calc_epsilon_ion_fromhessian(
        hessian: np.ndarray, 
        born: np.ndarray, 
        omega, 
        coeff = None, 
        mode:Union[Literal['vasp'], 
        Literal['einsum']]='vasp', 
        # dynmat: np.ndarray = None, 
    ):
    """
    Compute the ionic contribution to the dielectric tensor.

    :param hessian: (n_mode,n_mode)-ndarray, Hessian matrix (in UNIT_INTERNAL)
    :param born: (n_atom,3,3)-ndarray, Born effective charge (in |e|)
    :param OMEGA: float, Volume of lattice in real space
    :param coeff: float, adjust coeficient for units
    :param mode: vasp: for loop, einsum:einsum
    """

    if coeff is None:
        coeff = EDEPS

    hessian = symmetrize(hessian)
    # print("hessian: \n", hessian)
    # print("symmetrize(hessian): \n", symmetrize(hessian))
    # hessian = -hessian
    inv_hessian = inv_with_cutlowfreq(hessian, num_atoms=int(hessian.shape[0]/3))

    # print("inv_hessian: \n", inv_hessian)

    if mode == 'vasp': # for loop mode
        epsilon = np.zeros((3,3))
        DOF = inv_hessian.shape[0]
        for n in range(DOF):
            for np_ in range(DOF):
                for alpha in range(3):
                    for beta in range(3):
                        epsilon[alpha, beta] += inv_hessian[n, np_] * born[n // 3, alpha, n % 3] * born[
                            np_ // 3, beta, np_ % 3]
    else: # "einsum mode"
        n_atoms = int(inv_hessian.shape[0] / 3)
        inv_hessian = np.reshape(inv_hessian, (n_atoms, 3, n_atoms, 3))  # (ith_atom, xyz, jth_atom, xyz)
        # print(inv_hessian.shape, born.shape)
        epsilon = np.einsum('ikjl,iak,jbl->ab', inv_hessian, born, born)
        # 'ikjl,iak,jlb->ab', 'jkil,ika,jbl->ab' .... generate a little bit deviated values

    # print("epsilon: \n", epsilon)
    # print("epsilon * coeff / omega: \n", epsilon * coeff / omega)
    return epsilon * coeff / omega

def calc_freqs_fromdynmat(
        dynmat: np.ndarray, 
):
    w, _ = np.linalg.eigh(dynmat)

    max_freq = w[-1]
    min_freq = w[0]
    third_freq = w[2]
    fourth_freq = w[3] if len(w) > 3 else np.nan

    return (max_freq, min_freq, third_freq, fourth_freq)

