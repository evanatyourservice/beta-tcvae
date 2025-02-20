"""
Visualization tool for comparing learned latent variables against ground truth factors.
Generates plots showing the relationship between VAE latent dimensions and true generative factors.
"""

# Standard library imports
import os

# Third-party imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import brewer2mpl

# Local imports
from metric_helpers.loader import load_model_and_dataset
from elbo_decomposition import elbo_decomposition, ELBODecomposition

# Constants
BATCH_SIZE = 1000
VAR_THRESHOLD = 1e-2
IMAGE_SIZE = 64

# Set up color scheme
bmap = brewer2mpl.get_map('Set1', 'qualitative', 3)
COLORS = bmap.mpl_colors
plt.style.use('ggplot')


class LatentAnalyzer:
    """Class to analyze and visualize latent space representations."""
    
    def __init__(self, vae, dataset):
        self.vae = vae
        self.dataset = dataset
        self.dataset_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,
            shuffle=False,
            pin_memory=True
        )
        
    def _compute_latent_params(self):
        """Compute latent parameters for the entire dataset."""
        N = len(self.dataset_loader.dataset)
        K = self.vae.z_dim
        nparams = self.vae.q_dist.nparams
        
        self.vae.eval()
        qz_params = torch.Tensor(N, K, nparams)
        
        with torch.no_grad():
            n = 0
            for xs in self.dataset_loader:
                batch_size = xs.size(0)
                xs = Variable(xs.view(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE).cuda())
                qz_params[n:n + batch_size] = self.vae.encoder.forward(xs).view(
                    batch_size, self.vae.z_dim, nparams
                ).data
                n += batch_size
                
        return qz_params

    def _get_active_units(self, qz_means, N, K):
        """Identify active latent units based on variance threshold."""
        var = torch.std(qz_means.contiguous().view(N, K), dim=0).pow(2)
        active_units = torch.arange(0, K)[var > VAR_THRESHOLD].long()
        
        print('Active units: ' + ','.join(map(str, active_units.tolist())))
        n_active = len(active_units)
        print(f'Number of active units: {n_active}/{self.vae.z_dim}')
        
        return active_units if n_active > 0 else torch.arange(0, K).long()

    def plot_shapes_comparison(self, save_path, z_inds=None):
        """Generate comparison plots for shapes dataset."""
        with torch.no_grad():
            # Compute latent parameters
            qz_params = self._compute_latent_params()
            qz_params = qz_params.view(3, 6, 40, 32, 32, self.vae.z_dim, self.vae.q_dist.nparams)
            
            # Get active units
            qz_means = qz_params[:, :, :, :, :, :, 0]
            N = len(self.dataset_loader.dataset)
            active_units = self._get_active_units(qz_means, N, self.vae.z_dim)
            z_inds = z_inds if z_inds is not None else active_units
            
            # Compute mean values across different dimensions
            mean_scale = qz_means.mean(2).mean(2).mean(2)      # (shape, scale, latent)
            mean_rotation = qz_means.mean(1).mean(2).mean(2)   # (shape, rotation, latent)
            mean_pos = qz_means.mean(0).mean(0).mean(0)        # (pos_x, pos_y, latent)
            
            self._create_shapes_plot(mean_scale, mean_rotation, mean_pos, z_inds, save_path)

    def plot_faces_comparison(self, save_path, z_inds=None):
        """Generate comparison plots for faces dataset."""
        with torch.no_grad():
            # Compute latent parameters
            qz_params = self._compute_latent_params()
            qz_params = qz_params.view(50, 21, 11, 11, self.vae.z_dim, self.vae.q_dist.nparams)
            
            # Get active units
            qz_means = qz_params[:, :, :, :, :, 0]
            N = len(self.dataset_loader.dataset)
            active_units = self._get_active_units(qz_means, N, self.vae.z_dim)
            z_inds = z_inds if z_inds is not None else active_units
            
            # Compute mean values across different dimensions
            mean_pose_az = qz_means.mean(3).mean(2).mean(0)    # (pose_az, latent)
            mean_pose_el = qz_means.mean(3).mean(1).mean(0)    # (pose_el, latent)
            mean_light_az = qz_means.mean(2).mean(1).mean(0)   # (light_az, latent)
            
            self._create_faces_plot(mean_pose_az, mean_pose_el, mean_light_az, z_inds, save_path)

    def _create_shapes_plot(self, mean_scale, mean_rotation, mean_pos, z_inds, save_path):
        """Create visualization plot for shapes dataset."""
        fig = plt.figure(figsize=(3, len(z_inds)))
        gs = gridspec.GridSpec(len(z_inds), 3)
        gs.update(wspace=0, hspace=0)

        # Plot position
        vmin_pos = torch.min(mean_pos)
        vmax_pos = torch.max(mean_pos)
        for i, j in enumerate(z_inds):
            ax = fig.add_subplot(gs[i * 3])
            ax.imshow(mean_pos[:, :, j].numpy(), cmap=plt.get_cmap('coolwarm'), 
                     vmin=vmin_pos, vmax=vmax_pos)
            self._format_subplot(ax, i, j, z_inds, 'pos')

        # Plot scale
        self._plot_factor(fig, gs, mean_scale, z_inds, 1, 'scale')
        
        # Plot rotation
        self._plot_factor(fig, gs, mean_rotation, z_inds, 2, 'rotation')

        fig.text(0.5, 0.03, 'Ground Truth', ha='center')
        fig.text(0.01, 0.5, 'Learned Latent Variables ', va='center', rotation='vertical')
        plt.savefig(save_path)
        plt.close()

    def _create_faces_plot(self, mean_pose_az, mean_pose_el, mean_light_az, z_inds, save_path):
        """Create visualization plot for faces dataset."""
        fig = plt.figure(figsize=(len(z_inds), 3))
        gs = gridspec.GridSpec(3, len(z_inds))
        gs.update(wspace=0, hspace=0)

        # Plot azimuth
        self._plot_faces_factor(fig, gs, mean_pose_az, z_inds, 0, 'azimuth')
        
        # Plot elevation
        self._plot_faces_factor(fig, gs, mean_pose_el, z_inds, 1, 'elevation')
        
        # Plot lighting
        self._plot_faces_factor(fig, gs, mean_light_az, z_inds, 2, 'lighting')

        plt.suptitle('GT Factors vs. Latent Variables')
        plt.savefig(save_path)
        plt.close()

    def _plot_factor(self, fig, gs, mean_data, z_inds, offset, label):
        """Helper method to plot shape factors."""
        vmin = torch.min(mean_data)
        vmax = torch.max(mean_data)
        for i, j in enumerate(z_inds):
            ax = fig.add_subplot(gs[offset + i * 3])
            for shape_idx in range(3):
                ax.plot(mean_data[shape_idx, :, j].numpy(), color=COLORS[shape_idx])
            self._format_subplot(ax, i, j, z_inds, label, vmin, vmax)

    def _plot_faces_factor(self, fig, gs, mean_data, z_inds, row, label):
        """Helper method to plot face factors."""
        vmin = torch.min(mean_data)
        vmax = torch.max(mean_data)
        for i, j in enumerate(z_inds):
            ax = fig.add_subplot(gs[row * len(z_inds) + i])
            ax.plot(mean_data[:, j].numpy())
            self._format_subplot(ax, i, j, z_inds, label, vmin, vmax, row == 0)

    @staticmethod
    def _format_subplot(ax, i, j, z_inds, label, vmin=None, vmax=None, show_ylabel=True):
        """Helper method to format subplot axes."""
        if vmin is not None and vmax is not None:
            ax.set_ylim([vmin, vmax])
        ax.set_xticks([])
        ax.set_yticks([])
        if show_ylabel:
            ax.set_ylabel(rf'$z_{j}$')
        if i == len(z_inds) - 1:
            ax.set_xlabel(label)
        
        # Set aspect ratio
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))


def main():
    """Main function to run the latent space analysis."""
    parser = argparse.ArgumentParser(description='Analyze VAE latent space')
    parser.add_argument('-checkpt', required=True, help='Path to model checkpoint')
    parser.add_argument('-zs', type=str, default=None, help='Comma-separated list of latent indices')
    parser.add_argument('-gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('-save', type=str, default='latent_vs_gt.pdf', help='Save path for plot')
    parser.add_argument('-elbo_decomp', action='store_true', help='Compute ELBO decomposition')
    args = parser.parse_args()

    # Setup
    z_inds = list(map(int, args.zs.split(','))) if args.zs is not None else None
    torch.cuda.set_device(args.gpu)
    
    # Load model and dataset
    vae, dataset_loader, cpargs = load_model_and_dataset(args.checkpt)
    
    # Optional ELBO decomposition
    if args.elbo_decomp:
        elbo_decomposition(vae, dataset_loader)
    
    # Create analyzer and generate plots
    analyzer = LatentAnalyzer(vae, dataset_loader.dataset)
    plot_func = getattr(analyzer, f'plot_{cpargs.dataset}_comparison')
    plot_func(args.save, z_inds)


if __name__ == '__main__':
    import argparse
    main()
