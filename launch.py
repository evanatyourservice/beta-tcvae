"""
Launch script for distributed VAE training.

Usage:
    python launch.py          # Run with default kron optimizer
    python launch.py adam     # Run with adam optimizer
"""

import os
import sys
import subprocess
import torch

def main():
    # Default settings
    num_gpus = torch.cuda.device_count()
    master_port = '12355'
    
    # Set default arguments
    default_args = [
        '--dataset', 'shapes',
        '--beta', '6',
        '--tcvae',
        '--conv',
    ]
    
    # Allow optimizer to be specified, default to kron
    optimizer = 'adam'
    if len(sys.argv) > 1:
        optimizer = sys.argv[1]
    
    # Build torchrun command
    cmd = [
        'torchrun',
        f'--nproc_per_node={num_gpus}',
        f'--master_port={master_port}',
        'vae_quant.py',
    ] + default_args + ['--optimizer', optimizer]
    
    # Set required environment variables
    env = os.environ.copy()
    env['MASTER_ADDR'] = 'localhost'
    env['MASTER_PORT'] = master_port
    
    # Run the command
    process = subprocess.run(cmd, env=env)
    return process.returncode

if __name__ == '__main__':
    sys.exit(main()) 