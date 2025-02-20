"""ELBO decomposition for VAE models."""

import os
import math
from numbers import Number
import torch
from torch.autograd import Variable
from tqdm import tqdm
from metric_helpers.loader import load_model_and_dataset

NUM_SAMPLES = 10000
BATCH_SIZE = 10
IMAGE_SIZE = 64


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of value.exp().sum(dim, keepdim).log()"""
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + (math.log(sum_exp) if isinstance(sum_exp, Number) else torch.log(sum_exp))


class ELBODecomposition:
    def __init__(self, vae, dataset_loader):
        self.vae = vae
        self.dataset_loader = dataset_loader
        self.N = len(dataset_loader.dataset)
        self.K = vae.z_dim
        self.S = 1
        self.nparams = vae.q_dist.nparams

    def estimate_entropies(self, qz_samples, qz_params):
        with torch.no_grad():
            qz_samples = qz_samples.index_select(
                1, 
                Variable(torch.randperm(qz_samples.size(1))[:NUM_SAMPLES].cuda())
            )

            K, S = qz_samples.size()
            N, _, nparams = qz_params.size()
            assert nparams == self.vae.q_dist.nparams
            assert K == qz_params.size(1)

            marginal_entropies = torch.zeros(K).cuda()
            joint_entropy = torch.zeros(1).cuda()

            pbar = tqdm(total=S, desc="Estimating entropies")
            k = 0
            while k < S:
                batch_size = min(BATCH_SIZE, S - k)
                logqz_i = self.vae.q_dist.log_density(
                    qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
                    qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size]
                )
                k += batch_size

                marginal_entropies += (math.log(N) - logsumexp(logqz_i, dim=0, keepdim=False).data).sum(1)
                logqz = logqz_i.sum(1)
                joint_entropy += (math.log(N) - logsumexp(logqz, dim=0, keepdim=False).data).sum(0)
                pbar.update(batch_size)
            pbar.close()

            return marginal_entropies / S, joint_entropy / S

    def compute_qz_params(self):
        print('Computing q(z|x) distributions.')
        qz_params = torch.Tensor(self.N, self.K, self.nparams)
        n = 0
        logpx = 0
        
        for xs in self.dataset_loader:
            batch_size = xs.size(0)
            xs = Variable(xs.view(batch_size, -1, IMAGE_SIZE, IMAGE_SIZE).cuda())
            z_params = self.vae.encoder.forward(xs).view(batch_size, self.K, self.nparams)
            qz_params[n:n + batch_size] = z_params.data
            n += batch_size

            for _ in range(self.S):
                z = self.vae.q_dist.sample(params=z_params)
                x_params = self.vae.decoder.forward(z)
                logpx += self.vae.x_dist.log_density(xs, params=x_params).view(batch_size, -1).data.sum()
                
        return Variable(qz_params.cuda()), logpx / (self.N * self.S)

    def compute_decomposition(self):
        with torch.no_grad():
            qz_params, logpx = self.compute_qz_params()

            print('Sampling from q(z).')
            qz_params_expanded = qz_params.view(self.N, self.K, 1, self.nparams).expand(
                self.N, self.K, self.S, self.nparams
            )
            qz_samples = self.vae.q_dist.sample(params=qz_params_expanded)
            qz_samples = qz_samples.transpose(0, 1).contiguous().view(self.K, self.N * self.S)

            print('Estimating entropies.')
            marginal_entropies, joint_entropy = self.estimate_entropies(qz_samples, qz_params)

            if hasattr(self.vae.q_dist, 'NLL'):
                nlogqz_condx = self.vae.q_dist.NLL(qz_params).mean(0)
            else:
                nlogqz_condx = -self.vae.q_dist.log_density(
                    qz_samples,
                    qz_params_expanded.transpose(0, 1).contiguous().view(self.K, self.N * self.S)
                ).mean(1)

            if hasattr(self.vae.prior_dist, 'NLL'):
                pz_params = self.vae._get_prior_params(self.N * self.K).contiguous().view(self.N, self.K, -1)
                nlogpz = self.vae.prior_dist.NLL(pz_params, qz_params).mean(0)
            else:
                nlogpz = -self.vae.prior_dist.log_density(qz_samples.transpose(0, 1)).mean(0)

            nlogqz_condx = nlogqz_condx.data
            nlogpz = nlogpz.data

            dependence = (-joint_entropy + marginal_entropies.sum())[0]
            information = (-nlogqz_condx.sum() + joint_entropy)[0]
            dimwise_kl = (-marginal_entropies + nlogpz).sum()
            analytical_cond_kl = (-nlogqz_condx + nlogpz).sum()

            print(f'Dependence (TC): {dependence:.4f}')
            print(f'Information (MI): {information:.4f}')
            print(f'Dimension-wise KL: {dimwise_kl:.4f}')
            print(f'Analytical KL: {analytical_cond_kl:.4f}')
            print(f'Estimated ELBO: {logpx - analytical_cond_kl:.4f}')

            return (logpx, dependence, information, dimwise_kl, 
                   analytical_cond_kl, marginal_entropies, joint_entropy)


def elbo_decomposition(vae, dataset_loader):
    """Compatibility wrapper for the ELBODecomposition class."""
    decomp = ELBODecomposition(vae, dataset_loader)
    return decomp.compute_decomposition()


def main():
    parser = argparse.ArgumentParser(description='Compute ELBO decomposition for VAE')
    parser.add_argument('-checkpt', required=True, help='Path to model checkpoint')
    parser.add_argument('-save', type=str, default='.', help='Save directory')
    parser.add_argument('-gpu', type=int, default=0, help='GPU device ID')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    vae, dataset_loader = load_model_and_dataset(args.checkpt)
    
    decomp = ELBODecomposition(vae, dataset_loader)
    results = decomp.compute_decomposition()
    
    torch.save({
        'logpx': results[0],
        'dependence': results[1],
        'information': results[2],
        'dimwise_kl': results[3],
        'analytical_cond_kl': results[4],
        'marginal_entropies': results[5],
        'joint_entropy': results[6]
    }, os.path.join(args.save, 'elbo_decomposition.pth'))


if __name__ == '__main__':
    import argparse
    main()
