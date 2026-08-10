import argparse
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from niarb import nn, io, parsing, optimize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict_dir", type=Path)
    parser.add_argument("indices", type=str)
    args = parser.parse_args()
    
    indices = parsing.indices(args.indices)
    for path in io.iterdir(args.state_dict_dir, pattern="*.pt", indices=indices):
        print(path.stem)
        print_spectral_summary(path)


def print_spectral_summary(state_dict_path):
    state_dict = torch.load(state_dict_path, weights_only=True, map_location="cpu")
    model = nn.V1(
        variables=["cell_type", "space", "ori", "osi"],
        # cell_types=["PYR", "PV", "SST"],
        cell_types=["PYR", "PV", "SST", "VIP"],
        # tau=[1.0, 0.5, 1.0],
        tau=[1.0, 0.5, 1.0, 1.0],
        # f="SSN",
        # f=nn.Match(cases={"SST": nn.SSN(p=4)}, default=nn.Identity()),
        # f=nn.Match(cases={"PYR": nn.Identity()}, default=nn.Ricciardi(scale=0.967)),
        f=nn.Match(cases={"PYR": nn.Rectified(), "PV": nn.Ricciardi(scale=1.0, tau=0.01)}, default=nn.Ricciardi(scale=1.0)),
        osi_func=0.25,
        # osi_prob=("Beta", 0.88, 1.07),
        osi_prob=("Beta", 0.98, 1.28),
        sigma_symmetry="pre",
        vf_symmetry=False,
        mode="numerical",
    )
    nn.load_state_dict(model, state_dict, strict=False)
    # print(state_dict['gW'])
    # print(state_dict['kappa'])
    # print(model.state_dict())

    # df = pd.read_pickle(f"runs/eigvals_weight/{state_dict_path.stem}.pkl")
    # with torch.inference_mode():
    #     _, eigvals = model.spectral_summary(return_eigvals=True)
    # plt.scatter(df.real, df.imag, s=2, label="numerical")
    # plt.scatter(eigvals.real, eigvals.imag, s=2, label="analytical")
    # plt.legend()
    # plt.gca().axvline(1)
    # plt.gca().axhline(0)
    # plt.savefig(f"figures/eigvals_weight/{state_dict_path.stem}.pdf")
    # plt.close()

    # print(f"PYR-SST: {optimize.RelativeParamCon(param='kappa', cell_types_0=['SST', 'PYR'], cell_types_1=['PYR', 'SST'], frac=1.0, is_equality=True, eps=0.0)(model)}")
    # print(f"SST->E con: {optimize.LinearResponseCon(False, 'PYR', 'SST', eps=0)(model)}")
    # print(f"SST->E space con: {optimize.LinearResponseSpaceCon(d=2, r=torch.arange(10.0, 91.0, 10.0), cell_types=['PYR'], perturbed_cell_type='SST', eps=0, positive=False)(model)}")
    # print(f"VIP->E con: {optimize.LinearResponseCon(True, 'PYR', 'VIP', eps=0)(model)}")
    # print(f"VIP->E space con: {optimize.LinearResponseSpaceCon(d=2, r=torch.arange(10.0, 91.0, 10.0), cell_types=['PYR'], perturbed_cell_type='VIP', eps=0, positive=True)(model)}")
    # print(f"SST->PYR: {optimize.LinearResponseCon(True, 'PYR', 'SST', eps=0)(model)}")
    # print(f"SST->PV: {optimize.LinearResponseCon(True, 'PV', 'SST', eps=0)(model)}")
    # print(f"SST->SST: {optimize.LinearResponseCon(True, 'SST', 'SST', eps=0)(model)}")
    # print(f"SST->VIP: {optimize.LinearResponseCon(True, 'VIP', 'SST', eps=0)(model)}")
    # print(f"E->PV space ori con: {optimize.LinearResponseSpace2dOriCon(a=500.0, b=500.0, mode='diff', min_r=15.0, dr=5.0, eps=0.0, cell_types=['PV'])(model)}")
    # print(f"E->PV space ori con: {optimize.LinearResponseSpace2dOriCon(a=500.0, b=500.0, mode='diff', min_r=15.0, dr=5.0, eps=-0.1, cell_types=['PV'], positive=True)(model)}")
    # print(f"E->PV space ori con: {optimize.LinearResponseSpace2dOriCon(a=500.0, b=500.0, mode='diff', min_r=15.0, dr=5.0, eps=-0.1, cell_types=['PV'], positive=False)(model)}")
    print(f"SST->PYR space ori con: {optimize.LinearResponseSpace2dOriCon(a=500.0, b=500.0, mode='diff', min_r=15.0, dr=5.0, eps=0.1, cell_types=['PYR'], perturbed_cell_type='SST', positive=False)(model)}")
    # print(f"E->PV ori con: {optimize.LinearResponseOriAnalyticCon(True, 'PV', 'PYR', eps=0.0)(model)}")
    # print(f"E->PV ori con: {optimize.LinearResponseOriAnalyticCon(True, 'PV', 'PYR', eps=-0.2)(model)}")
    # print(f"E->PV ori con: {optimize.LinearResponseOriAnalyticCon(False, 'PV', 'PYR', eps=-0.2)(model)}")
    # print(f"Space con: {optimize.LinearResponseSpaceCon(eps=0.0, d=2, r=np.arange(10, 300, 10.0), cell_types=['SST'])(model)}")
    # print(optimize.LinearResponseOriCon(70.0, cell_types=["SST"])(model))
    # print(optimize.LinearResponseSpace2dOriCon(a=500.0, b=500.0, theta=50.0, min_r=15.0, dr=5.0, eps=0.0, cell_types=["SST"])(model))
    # print(f"Ori con (50): {optimize.LinearResponseSpace2dOriCon(a=500.0, b=500.0, theta=50.0, min_r=15.0, dr=5.0, eps=0.0, cell_types=['SST'])(model)}")
    # print(f"Ori con (60): {optimize.LinearResponseSpace2dOriCon(a=500.0, b=500.0, theta=60.0, min_r=15.0, dr=5.0, eps=0.0, cell_types=['SST'])(model)}")
    # print(f"Spectral norm: {model.spectral_norm()}, {model.spectral_norm(H=False, cell_types=['PYR'])}")
    # print(f"Spectral norm (PV): {model.spectral_norm(H=False, cell_types=['PV'])}")
    # print(f"Spectral norm (SST): {model.spectral_norm(H=False, cell_types=['SST'])}")
    # print(f"Full network: {model.spectral_summary(kind='J', kmax=1000.0, ksteps=100000)}")
    # print(f"Full network: {optimize.StabilityCon(eps=0)(model)}")
    # print(f"Full network (SST double): {optimize.StabilityCon(eps=0, rel_vf=[1.0, 1.0, 2.0 ** (1 / 3)])(model)}")
    # print(f"Full network (SST double): {optimize.StabilityCon(eps=0, rel_vf=[1.0, 1.0, 2.0 ** (1 / 3), 1.0])(model)}")
    # print(f"Paradoxical PV: {optimize.ParadoxicalCon(eps=0)(model)}")
    # print(f"Paradoxical SST: {optimize.ParadoxicalCon(eps=0, cell_type='SST')(model)}")
    # print(f"Full network (SST = 0): {model.spectral_summary(kind='J', dh=torch.tensor([0.0, 0.0, -0.48]))}")
    # print(f"Full network (modified): {model.spectral_summary(kind='J', dh=torch.tensor([-0.1, -0.2, 0.1]))}")
    # print(f"Full network (SST = 4): {model.spectral_summary(kind='J', dh=torch.tensor([0.0, 0.0, 0.48]))}")
    # print(f"Full network (SST = 16): {model.spectral_summary(kind='J', dh=torch.tensor([0.0, 0.0, 1.44]))}")
    # print(f"E-PV subnetwork: {model.spectral_summary(kind='J', cell_types=['PYR', 'PV'])}")
    # print(f"E subnetwork: {model.spectral_summary(kind='J', cell_types=['PYR'])}")
    # print(f"E-SST subetnwork: {model.spectral_summary(kind='J', cell_types=['PYR', 'SST'])}")

    # Mee = optimize.DeterminantCon(eps=0, exclude=('PYR', 'PYR'))(model)
    # Mpp = optimize.DeterminantCon(eps=0, exclude=('PV', 'PV'))(model)
    # Mss = optimize.DeterminantCon(eps=0, exclude=('SST', 'SST'))(model)
    # x = model.W().trace()
    # y = Mee + Mpp + Mss
    # z = optimize.DeterminantCon(eps=0)(model)
    # discriminant = x**2 * y**2 - 4 * y**3 - 4 * x**3 * z - 27 * z**2 + 18 * x * y * z
    # print(f"Tr(W): {x}")
    # print(f"Det(W): {z}")
    # print(f"Mee: {Mee}")
    # print(f"Mpp: {Mpp}")
    # print(f"Mss: {Mss}")
    # print(f"Mee + Mpp + Mss: {y}")
    # print(f"Discriminant: {discriminant}")
    # print(f"Mep: {optimize.DeterminantCon(eps=0, exclude=('PYR', 'PV'))(model)}")
    # print(f"Mes: {optimize.DeterminantCon(eps=0, exclude=('PYR', 'SST'))(model)}")


if __name__ == "__main__":
    main()

