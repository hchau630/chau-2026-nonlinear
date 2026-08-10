import argparse
from pathlib import Path

import torch

from niarb import parsing, io
from niarb.cell_type import CellType
from niarb.tensors import categorical
from niarb.optimize import elementwise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", type=str)
    parser.add_argument("--conf", type=Path, default="fit.toml")
    parser.add_argument("--dhp", type=float, default=10.0)
    parser.add_argument("--cell-type", type=str, default="SST")
    parser.add_argument("--fits", type=Path, default="fits")
    parser.add_argument("--out", "-o", type=Path, default="modified_fits")
    args = parser.parse_args()
    
    indices = parsing.indices(args.indices)
    conf = io.load_config(args.conf)
    f = conf["validation_pipeline"]["model"].f
    cell_types = conf["validation_pipeline"]["model"].cell_types
    i = cell_types.index(CellType[args.cell_type])
    ct = categorical.tensor(i, categories=tuple([ct.name for ct in cell_types]))

    for filename in io.iterdir(args.fits, pattern="*.pt", indices=indices):
        state_dict = torch.load(filename, weights_only=True)
        print(f"Original state_dict:\n{state_dict}")
        vf = state_dict["vf"]
        ct = ct.to(vf.device)
        dhp = torch.tensor(args.dhp, device=vf.device)
        vf0 = torch.ones((), device=vf.device)
        if vf.ndim != 1:
            raise ValueError()
        dh = elementwise.newton(lambda x: f(x, ct) - (f(vf[i], ct) + dhp), vf0) - vf[i]
        state_dict["dh"] = dh
        print(f"Modified state_dict:\n{state_dict}")
        torch.save(state_dict, args.out / filename.name)


if __name__ == "__main__":
    main()
