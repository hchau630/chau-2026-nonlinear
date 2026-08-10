import argparse
from pathlib import Path

import torch

from niarb import parsing, io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", type=str)
    parser.add_argument("--conf", type=Path, default="fit.toml")
    parser.add_argument("--fits", type=Path, default="fits")
    parser.add_argument("--out", "-o", type=Path, default="modified_fits")
    args = parser.parse_args()
    
    indices = parsing.indices(args.indices)
    conf = io.load_config(args.conf)
    f = conf["validation_pipeline"]["model"].f
    for filename in io.iterdir(args.fits, pattern="*.pt", indices=indices):
        state_dict = torch.load(filename, weights_only=True)
        print(f"Original state_dict:\n{state_dict}")
        dh, vf = state_dict["dh"], state_dict["vf"]
        if vf.ndim > 0:
            vf = vf[0]
        state_dict["dh"] = f(vf + dh) - f(vf)
        print(f"Modified state_dict:\n{state_dict}")
        torch.save(state_dict, args.out / filename.name)


if __name__ == "__main__":
    main()
