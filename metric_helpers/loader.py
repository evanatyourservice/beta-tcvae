import torch
import lib.dist as dist
import lib.flows as flows
import lib.utils
import warnings
import argparse
from torch.serialization import add_safe_globals
import os

from lib.datasets import setup_data_loaders
from models import VAE

add_safe_globals([argparse.Namespace])

def load_model_and_dataset(checkpt_filename):
    print(f'Loading model from {checkpt_filename}')
    if not os.path.exists(checkpt_filename):
        raise FileNotFoundError(f"No checkpoint found at {checkpt_filename}")
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        checkpt = torch.load(
            checkpt_filename,
            map_location='cpu',
            weights_only=False
        )
    
    if 'state_dict' not in checkpt or 'args' not in checkpt:
        raise ValueError(f"Checkpoint is missing required keys. Found keys: {checkpt.keys()}")
        
    args = checkpt['args']
    state_dict = checkpt['state_dict']

    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            fixed_state_dict[k[10:]] = v
        else:
            fixed_state_dict[k] = v

    dist_classes = {
        'normal': (dist.Normal(), dist.Normal()),
        'laplace': (dist.Laplace(), dist.Laplace()),
        'flow': (flows.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32), dist.Normal())
    }
    prior_dist, q_dist = dist_classes.get(getattr(args, 'dist', 'normal'))

    vae = VAE(
        z_dim=args.latent_dim if not hasattr(args, 'ncon') else args.ncon,
        use_cuda=False,
        prior_dist=prior_dist,
        q_dist=q_dist,
        include_mutinfo=not args.exclude_mutinfo,
        tcvae=args.tcvae,
        conv=args.conv,
        mss=args.mss
    )
    
    load_result = vae.load_state_dict(fixed_state_dict, strict=True)
    print(f"Loaded checkpoint with result: {load_result}")
    
    vae.cuda()
    vae.eval()

    vae = torch.compile(vae)

    loader = setup_data_loaders(args)
    return vae, loader, args
