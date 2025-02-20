import os
import time
from datetime import datetime
import multiprocessing
import wandb
import torch
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from kron_torch import DistributedKron as Kron
import torch.distributed as dist_backend

from lib.datasets import setup_data_loaders
import lib.dist as dist
from lib.flows import FactorialNormalizingFlow
from models import VAE
from elbo_decomposition import elbo_decomposition
from lib.utils import RunningAverageMeter, isnan, save_checkpoint
from plot_latent_vs_true import LatentAnalyzer


def plot_to_image(matrix, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    im = ax.imshow(matrix.cpu().numpy())
    plt.colorbar(im)
    ax.set_title(title)
    plt.close()
    return fig


def display_samples(model, x, step):
    with torch.no_grad():
        sample_mu = model.model_sample(batch_size=20).sigmoid()
        sample_grid = torchvision.utils.make_grid(
            sample_mu.view(-1, 1, 64, 64), nrow=5, normalize=True, value_range=(0, 1)
        )
        wandb.log({"samples": wandb.Image(sample_grid)}, step=step)

        test_imgs = x[:10]
        _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
        comparison = torch.cat([
            test_imgs.view(-1, 1, 64, 64),
            reco_imgs.sigmoid().view(-1, 1, 64, 64)
        ])
        recon_grid = torchvision.utils.make_grid(comparison, nrow=10, normalize=True)
        wandb.log({"reconstructions": wandb.Image(recon_grid)}, step=step)

        zs = zs[0:3]
        batch_size, z_dim = zs.size()
        xs = []
        delta = torch.autograd.Variable(torch.linspace(-2, 2, 7)).type_as(zs)
        for i in range(z_dim):
            vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)
            vec[:, i] = 1
            vec = vec * delta[:, None]
            zs_delta = zs.clone().view(batch_size, 1, z_dim)
            zs_delta[:, :, i] = 0
            zs_walk = zs_delta + vec[None]
            xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
            xs.append(xs_walk)

        latent_walk_imgs = torch.cat(xs, 0)
        latent_grid = torchvision.utils.make_grid(
            latent_walk_imgs.view(-1, 1, 64, 64), nrow=7, normalize=True
        )
        wandb.log({"latent_walks": wandb.Image(latent_grid)}, step=step)

        zs, z_params = model.encode(x)
        z_means = z_params[:,:,0].mean(0)
        z_stds = z_params[:,:,1].exp().mean(0)
        
        wandb.log({
            f"z_means/dim_{dim}": z_means[dim] for dim in range(model.z_dim)
        }, step=step)
        wandb.log({
            f"z_stds/dim_{dim}": z_stds[dim] for dim in range(model.z_dim)
        }, step=step)
        wandb.log({
            f"z_distributions/dim_{dim}": wandb.Histogram(zs[:,dim].cpu().numpy())
            for dim in range(min(model.z_dim, 10))
        }, step=step)

        z_corr = torch.corrcoef(zs.T)
        wandb.log({
            "latent_correlations": wandb.Image(plot_to_image(z_corr, "Latent Correlations"))
        }, step=step)


def anneal_kl(args, vae, iteration):
    warmup_iter = 7000 if args.dataset == 'shapes' else 2500
    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)
    else:
        vae.lamb = 0
    if args.beta_anneal:
        vae.beta = min(args.beta, args.beta / warmup_iter * iteration)
    else:
        vae.beta = args.beta


def get_gradient_norm(parameters):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def main():
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='shapes', choices=['shapes', 'faces'])
    parser.add_argument('-dist', default='normal', choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int)
    parser.add_argument('-b', '--batch-size', default=2048, type=int)
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'kron'])
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float)
    parser.add_argument('-z', '--latent-dim', default=10, type=int)
    parser.add_argument('--beta', default=1, type=float)
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    parser.add_argument('--mss', action='store_true')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--wandb-project', default='vae-disentanglement')
    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=200, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # Initialize distributed training
    if not dist_backend.is_initialized():
        dist_backend.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=1,
            rank=args.local_rank
        )

    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    train_loader = setup_data_loaders(args)

    prior_dist = {
        'normal': dist.Normal(),
        'laplace': dist.Laplace(),
        'flow': FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
    }[args.dist]
    q_dist = dist.Normal() if args.dist != 'laplace' else dist.Laplace()

    vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
              include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, 
              conv=args.conv, mss=args.mss)
    vae = torch.compile(vae)

    optimizer = {
        'adam': optim.Adam(vae.parameters(), lr=args.learning_rate, fused=True),
        'kron': Kron(vae.parameters(), lr=args.learning_rate / 3.0, weight_decay=0.1)
    }[args.optimizer]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{timestamp}"
    args.save = run_name

    wandb.init(
        project=args.wandb_project,
        config=args.__dict__,
        name=run_name,
        tags=[args.dataset, args.dist, 'conv' if args.conv else 'mlp'],
        group=args.dataset
    )

    elbo_running_mean = RunningAverageMeter()
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0
    
    while iteration < num_iterations:
        for x in train_loader:
            iteration += 1
            batch_time = time.time()
            vae.train()
            anneal_kl(args, vae, iteration)
            
            optimizer.zero_grad()
            x = Variable(x.cuda())
            (obj, elbo), (logpx, logpz, logqz_condx) = vae.elbo(x, dataset_size)
            
            if isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
                
            loss = -obj.mean()
            loss.backward()
            elbo_running_mean.update(elbo.mean().item())
            optimizer.step()

            if iteration % args.log_freq == 0:
                print(f'[iteration {iteration:03d}] time: {time.time() - batch_time:.2f} '
                      f'beta {vae.beta:.2f} lambda {vae.lamb:.2f} '
                      f'training ELBO: {elbo_running_mean.val:.4f} ({elbo_running_mean.avg:.4f})')

                metrics = {
                    'elbo': elbo_running_mean.val,
                    'elbo_avg': elbo_running_mean.avg,
                    'beta': vae.beta,
                    'lambda': vae.lamb,
                    'progress': iteration / num_iterations,
                    'time_per_iter': time.time() - batch_time,
                    'loss': loss.item(),
                    'logpx': logpx.mean().item(),
                    'logpz': logpz.mean().item(),
                    'logqz': logqz_condx.mean().item(),
                    'grad_norm': get_gradient_norm(vae.parameters()),
                    'memory_allocated': torch.cuda.memory_allocated() / 1024**2,
                    'memory_reserved': torch.cuda.memory_reserved() / 1024**2
                }

                for name, param in vae.named_parameters():
                    if param.requires_grad:
                        metrics.update({
                            f"param_norm/{name}": param.norm().item(),
                            f"grad_norm/{name}": param.grad.norm().item() if param.grad is not None else 0,
                        })

                wandb.log(metrics, step=iteration)

                vae.eval()
                display_samples(vae, x, iteration)
                save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args
                }, args.save, 0)
                
                plot_path = os.path.join(args.save, f'gt_vs_latent_{iteration:05d}.png')
                analyzer = LatentAnalyzer(vae, train_loader.dataset)
                plot_func = getattr(analyzer, f'plot_{args.dataset}_comparison')
                plot_func(plot_path)
                wandb.log({"gt_vs_latent": wandb.Image(plot_path)})

    vae.eval()
    save_checkpoint({'state_dict': vae.state_dict(), 'args': args}, args.save, 0)
    
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
    results = elbo_decomposition(vae, dataset_loader)
    
    torch.save({
        'logpx': results[0],
        'dependence': results[1],
        'information': results[2],
        'dimwise_kl': results[3],
        'analytical_cond_kl': results[4],
        'marginal_entropies': results[5],
        'joint_entropy': results[6]
    }, os.path.join(args.save, 'elbo_decomposition.pth'))

    wandb.log({
        'final_logpx': results[0],
        'final_dependence': results[1],
        'final_information': results[2],
        'final_analytical_cond_kl': results[4],
        'final_joint_entropy': results[6],
        'final_dimwise_kl': wandb.Histogram(results[3].cpu().numpy()),
        'final_marginal_entropies': wandb.Histogram(results[5].cpu().numpy()),
    })

    final_plot_path = os.path.join(args.save, 'gt_vs_latent.png')
    analyzer = LatentAnalyzer(vae, dataset_loader.dataset)
    plot_func = getattr(analyzer, f'plot_{args.dataset}_comparison')
    plot_func(final_plot_path)
    wandb.log({"final_gt_vs_latent": wandb.Image(final_plot_path)})

    wandb.finish()
    return vae


if __name__ == '__main__':
    import argparse
    model = main()
