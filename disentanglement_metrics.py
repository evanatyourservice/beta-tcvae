"""
Disentanglement metrics for VAE models.
Computes mutual information gap (MIG) metrics for shapes and faces datasets.
"""

# Standard library imports
import math
import os

# Third-party imports
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

# Local imports
import lib.utils as utils
from metric_helpers.loader import load_model_and_dataset
from metric_helpers.mi_metric import compute_metric_shapes, compute_metric_faces

# Constants
BATCH_SIZE = 1000
NUM_WORKERS = 1
IMAGE_SIZE = 64


def estimate_entropies(qz_samples, qz_params, q_dist, n_samples=10000, weights=None):
    """
    Compute entropy estimates for the latent distributions.
    
    Computes:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    
    Args:
        qz_samples (torch.Tensor): Samples from q(z|x), shape (K, N)
        qz_params (torch.Tensor): Parameters of q(z|x), shape (N, K, nparams)
        q_dist: Distribution object with log_density method
        n_samples (int): Number of samples to use
        weights (torch.Tensor, optional): Sample weights
        
    Returns:
        torch.Tensor: Estimated entropies for each latent dimension
    """
    with torch.no_grad():
        if weights is None:
            qz_samples = qz_samples.index_select(
                1, 
                Variable(torch.randperm(qz_samples.size(1))[:n_samples].cuda())
            )
        else:
            sample_inds = torch.multinomial(weights, n_samples, replacement=True)
            qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    weights = -math.log(N) if weights is None else torch.log(weights.view(N, 1, 1) / weights.sum())
    entropies = torch.zeros(K).cuda()

    pbar = tqdm(total=S)
    k = 0
    while k < S:
        batch_size = min(10, S - k)
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size]
        )
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        entropies += - utils.logsumexp(logqz_i + weights, dim=0, keepdim=False).data.sum(1)
        pbar.update(batch_size)
    pbar.close()

    return entropies / S


class DisentanglementMetrics:
    """Class to handle computation of disentanglement metrics."""
    
    def __init__(self, vae, dataset):
        self.vae = vae
        self.dataset = dataset
        self.dataset_loader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False
        )
        
    def _compute_qz_params(self):
        """Compute q(z|x) distributions for the entire dataset."""
        N = len(self.dataset_loader.dataset)
        K = self.vae.z_dim
        nparams = self.vae.q_dist.nparams
        
        self.vae.eval()
        qz_params = torch.Tensor(N, K, nparams)
        
        n = 0
        for xs in self.dataset_loader:
            batch_size = xs.size(0)
            xs = Variable(xs.view(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE).cuda())
            qz_params[n:n + batch_size] = self.vae.encoder.forward(xs).view(
                batch_size, self.vae.z_dim, nparams
            ).data
            n += batch_size
            
        return qz_params

    def compute_shapes_metric(self):
        """Compute disentanglement metric for shapes dataset."""
        with torch.no_grad():
            qz_params = self._compute_qz_params()
            qz_params = Variable(qz_params.view(3, 6, 40, 32, 32, self.vae.z_dim, self.vae.q_dist.nparams).cuda())
            qz_samples = self.vae.q_dist.sample(params=qz_params)
            
            print('Estimating marginal entropies.')
            marginal_entropies = estimate_entropies(
                qz_samples.view(-1, self.vae.z_dim).transpose(0, 1),
                qz_params.view(-1, self.vae.z_dim, self.vae.q_dist.nparams),
                self.vae.q_dist
            ).cpu()
            
            cond_entropies = self._compute_shapes_conditional_entropies(qz_samples, qz_params)
            metric = compute_metric_shapes(marginal_entropies, cond_entropies)
            
            return metric, marginal_entropies, cond_entropies

    def compute_faces_metric(self):
        """Compute disentanglement metric for faces dataset."""
        with torch.no_grad():
            qz_params = self._compute_qz_params()
            qz_params = Variable(qz_params.view(50, 21, 11, 11, self.vae.z_dim, self.vae.q_dist.nparams).cuda())
            qz_samples = self.vae.q_dist.sample(params=qz_params)
            
            print('Estimating marginal entropies.')
            marginal_entropies = estimate_entropies(
                qz_samples.view(-1, self.vae.z_dim).transpose(0, 1),
                qz_params.view(-1, self.vae.z_dim, self.vae.q_dist.nparams),
                self.vae.q_dist
            ).cpu()
            
            cond_entropies = self._compute_faces_conditional_entropies(qz_samples, qz_params)
            metric = compute_metric_faces(marginal_entropies, cond_entropies)
            
            return metric, marginal_entropies, cond_entropies

    def _compute_shapes_conditional_entropies(self, qz_samples, qz_params):
        """Helper method to compute conditional entropies for shapes dataset."""
        N = len(self.dataset_loader.dataset)
        K = self.vae.z_dim
        cond_entropies = torch.zeros(4, K)
        
        factors = [
            ('scale', 6),
            ('orientation', 40),
            ('pos x', 32),
            ('pos y', 32)
        ]
        
        for idx, (name, size) in enumerate(factors):
            print(f'Estimating conditional entropies for {name}.')
            for i in range(size):
                slices = [slice(None)] * 5
                slices[idx + 1] = i
                qz_samples_factor = qz_samples[tuple(slices)].contiguous()
                qz_params_factor = qz_params[tuple(slices)].contiguous()
                
                cond_entropies_i = estimate_entropies(
                    qz_samples_factor.view(N // size, K).transpose(0, 1),
                    qz_params_factor.view(N // size, K, self.vae.q_dist.nparams),
                    self.vae.q_dist
                )
                cond_entropies[idx] += cond_entropies_i.cpu() / size
                
        return cond_entropies

    def _compute_faces_conditional_entropies(self, qz_samples, qz_params):
        """Helper method to compute conditional entropies for faces dataset."""
        N = len(self.dataset_loader.dataset)
        K = self.vae.z_dim
        cond_entropies = torch.zeros(3, K)
        
        factors = [
            ('azimuth', 21),
            ('elevation', 11),
            ('lighting', 11)
        ]
        
        for idx, (name, size) in enumerate(factors):
            print(f'Estimating conditional entropies for {name}.')
            for i in range(size):
                slices = [slice(None)] * 4
                slices[idx + 1] = i
                qz_samples_factor = qz_samples[tuple(slices)].contiguous()
                qz_params_factor = qz_params[tuple(slices)].contiguous()
                
                cond_entropies_i = estimate_entropies(
                    qz_samples_factor.view(N // size, K).transpose(0, 1),
                    qz_params_factor.view(N // size, K, self.vae.q_dist.nparams),
                    self.vae.q_dist
                )
                cond_entropies[idx] += cond_entropies_i.cpu() / size
                
        return cond_entropies


def main():
    """Main function to run the disentanglement metrics computation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpt', required=True, help='Path to model checkpoint')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--save', type=str, default='.', help='Save directory')
    args = parser.parse_args()

    if args.gpu != 0:
        torch.cuda.set_device(args.gpu)
        
    vae, dataset_loader, cpargs = load_model_and_dataset(args.checkpt)
    metrics = DisentanglementMetrics(vae, dataset_loader.dataset)
    
    compute_func = (metrics.compute_shapes_metric if cpargs.dataset == 'shapes' 
                   else metrics.compute_faces_metric)
    
    metric, marginal_entropies, cond_entropies = compute_func()
    
    # Save results
    torch.save({
        'metric': metric,
        'marginal_entropies': marginal_entropies,
        'cond_entropies': cond_entropies,
    }, os.path.join(args.save, 'disentanglement_metric.pth'))
    
    print('MIG: {:.2f}'.format(metric))


if __name__ == '__main__':
    import argparse
    main()
