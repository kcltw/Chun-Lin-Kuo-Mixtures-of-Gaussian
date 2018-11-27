import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim
from MLP_Layer import MLPLayer
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn
from mog import data_generator
import math

# torch.manual_seed(123)
c = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch MNIST WAE-GAN_server1')
parser.add_argument('-batch_size', type=int, default= 100, metavar='N',help='input batch size for training (default: 100)')
parser.add_argument('-sample_size', type=int, default=100, metavar='N',help='input batch size for training (default: 100)')
parser.add_argument('-iterations', type=int, default=30000, help='number of epochs to train (default: 100)')
parser.add_argument('-interval', type=int, default=5000)
parser.add_argument('-lr', type=float, default= 1e-3, help='learning rate (default: 0.0001)')
parser.add_argument('-dim_h', type=int, default=16 , help='hidden dimension (default: 128)')
parser.add_argument('-n_z', type=int, default= 32, help='hidden dimension of z (default: 8)')
parser.add_argument('-LAMBDA', type=float, default=1, help='regularization coef MMD term (default: 10)')
parser.add_argument('-sigma_z', type=float, default=1, help='variance of hidden dimension (default: 1)')
parser.add_argument('-sigma_prior', type=float, default=torch.tensor(np.exp(-3)).to(device))
parser.add_argument('-n_mc', type=int, default=5)
parser.add_argument('-n_input', type=int, default=2)
parser.add_argument('-optim_betas', type=tuple, default=(0.5, 0.999))
args = parser.parse_args()

plt.style.use('ggplot')
dset = data_generator()
dset.uniform_distribution()
sample_dir = './wae_gaussian_z32_h16/'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def plot(points, title):
    plt.scatter(points[:, 0], points[:, 1], s=10, c='b', alpha=0.5)
    plt.scatter(dset.centers[:, 0], dset.centers[:, 1], s= 100, c='g', alpha=0.5)
    plt.title(title)
    plt.ylim(-5, 5)
    plt.xlim(-5, 5)
    plt.savefig( sample_dir + title + '.png')
    plt.clf()
    # plt.show()
    # plt.close()


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.dim_h = args.dim_h
        self.n_z = args.n_z
        self.input = args.n_input

        self.l1 = MLPLayer(self.input, self.dim_h, args.sigma_prior)
        self.l1_act = nn.Tanh()
        self.l2 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l2_act = nn.Tanh()
        self.l3 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l3_act = nn.Tanh()
        self.l4 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l4_act = nn.Tanh()
        self.l5 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l5_act = nn.Tanh()
        self.l6 = MLPLayer(self.dim_h, self.n_z, args.sigma_prior)

    def forward(self, x):
        output = self.l1_act(self.l1(x))
        output = self.l2_act(self.l2(output))
        output = self.l3_act(self.l3(output))
        output = self.l4_act(self.l4(output))
        output = self.l5_act(self.l5(output))
        output = self.l6(output)
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw  + self.l4.lpw + self.l5.lpw + self.l6.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw  + self.l4.lqw + self.l5.lqw + self.l6.lqw
        return lpw, lqw


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.output = args.n_input
        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.l1 = MLPLayer(self.n_z, self.dim_h, args.sigma_prior)
        self.l1_act = nn.Tanh()
        self.l2 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l2_act = nn.Tanh()
        self.l3 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l3_act = nn.Tanh()
        self.l4 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l4_act = nn.Tanh()
        self.l5 = MLPLayer(self.dim_h, self.dim_h, args.sigma_prior)
        self.l5_act = nn.Tanh()
        self.l6 = MLPLayer(self.dim_h, self.output, args.sigma_prior)


    def forward(self, z):
        output = self.l1_act(self.l1(z))
        output = self.l2_act(self.l2(output))
        output = self.l3_act(self.l3(output))
        output = self.l4_act(self.l4(output))
        output = self.l5_act(self.l5(output))
        output = self.l6(output)
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw  + self.l3.lpw + self.l4.lpw + self.l5.lpw + self.l6.lpw
        lqw = self.l1.lqw + self.l2.lqw  + self.l3.lqw + self.l4.lqw + self.l5.lqw + self.l6.lqw
        return lpw, lqw


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()

        self.dim_h = args.dim_h
        self.n_z = args.n_z

        self.main = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h),
            # nn.BatchNorm1d(self.dim_h),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h, self.dim_h),
            # nn.BatchNorm1d(self.dim_h),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h, self.dim_h),
            # nn.BatchNorm1d(self.dim_h),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h, self.dim_h),
            # nn.BatchNorm1d(self.dim_h),
            nn.LeakyReLU(0.2),
            nn.Linear(self.dim_h, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.main(x)
        return x


def forward_pass_samples(x, real_labels):
    enc_kl, dec_kl, enc_scores, sam_scores = torch.zeros(args.n_mc), torch.zeros(args.n_mc), torch.zeros(
        args.n_mc), torch.zeros(args.n_mc)
    enc_log_likelihoods, dec_log_likelihoods = torch.zeros(args.n_mc), torch.zeros(args.n_mc)
    for i in range(args.n_mc):
        z_enc = encoder(x)
        x_rec = decoder(z_enc)
        rec_loss = mse_sum(x_rec, x)
        d_enc = discriminator(z_enc)
        # div_loss =  -args.LAMBDA * (torch.log(d_enc)).sum()
        div_loss = bcewl_sum(d_enc, real_labels)
        # print("rec_loss",rec_loss.item())
        # print("div_loss",div_loss.item())
        enc_log_likelihood = rec_loss * 10 + div_loss
        dec_log_likelihood = rec_loss * 10 + div_loss

        enc_log_pw, enc_log_qw = encoder.get_lpw_lqw()
        dec_log_pw, dec_log_qw = decoder.get_lpw_lqw()

        enc_kl[i] = enc_log_qw - enc_log_pw
        dec_kl[i] = dec_log_qw - dec_log_pw
        enc_log_likelihoods[i] = enc_log_likelihood
        dec_log_likelihoods[i] = dec_log_likelihood
        enc_scores[i] = d_enc.mean()
        # sam_scores[i] = d_sam.mean()

    return enc_kl.mean(), dec_kl.mean(), enc_log_likelihoods.mean(), dec_log_likelihoods.mean(), enc_scores.mean()  # , sam_scores.mean()


encoder, decoder, discriminator = Encoder(args).to(device), Decoder(args).to(device), Discriminator(args).to(device)
mse = nn.MSELoss()
mse_sum = nn.MSELoss(reduction= 'sum')
bcewl = nn.BCEWithLogitsLoss()
bcewl_sum = nn.BCEWithLogitsLoss(reduction= 'sum')
bce = nn.BCELoss()

def criterion(kl, log_likelihood):
    # print("kl" , kl.item())
    # print("likelihood", log_likelihood.item())
    return kl / 1000 + log_likelihood



def dec_sample():
    with torch.no_grad():
        z_sam = (torch.randn(args.sample_size, args.n_z) * args.sigma_z).to(device)
        x_rec = decoder(z_sam)
        return x_rec.cpu().numpy()

def plot_samples(samples):
    xmax = 5
    rows = 3
    cols = math.ceil(len(samples)/rows)
    bg_color = seaborn.color_palette('Greens', n_colors=256)[0]
    #plt.figure(figsize=(3 * cols,   * cols))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(rows, cols , 1)
        else:
            plt.subplot(rows, cols, i + 1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(samps[:, 0], samps[:, 1], shaded=True, cmap='Greens', n_levels=20, clip=[[-xmax, xmax]] * 2)
        ax2.set_facecolor(bg_color)
        plt.xticks([])
        plt.yticks([])
        if i % 2 == 0 :
            plt.title('iteration %d' % (i/2 * args.interval))
        else :
            plt.title('ground truth')

    plt.gcf().tight_layout()
    plt.savefig(sample_dir + "iterations.png")





# Optimizers
enc_optim = optim.Adam(encoder.parameters(), lr=args.lr) # betas = args.optim_betas
dec_optim = optim.Adam(decoder.parameters(), lr=args.lr)
dis_optim = optim.Adam(discriminator.parameters(), lr= args.lr * 0.1)

# enc_sch = StepLR(enc_optim, step_size= 9000, gamma= 0.1)
# dec_sch = StepLR(dec_optim, step_size= 9000, gamma= 0.1)
# dis_sch = StepLR(dis_optim, step_size= 9000, gamma= 0.1)




#samples = []
for it in range(args.iterations):
    # enc_sch.step()
    # dec_sch.step()
    # dis_sch.step()
    x = torch.from_numpy(dset.sample(args.batch_size)).to(device)
    real_labels = torch.ones(args.batch_size, 1).to(device)
    fake_labels = torch.zeros(args.batch_size, 1).to(device)

    # ======== Train Generator ======== #
    free_params(decoder)
    free_params(encoder)
    frozen_params(discriminator)

    enc_kl, dec_kl, enc_log_likelihood, dec_log_likelihood, enc_scores = forward_pass_samples(x, real_labels)
    enc_loss = criterion(enc_kl, enc_log_likelihood)
    dec_loss = criterion(dec_kl, dec_log_likelihood)
    # enc_loss = enc_criterion_reW(enc_kl, (it + 1), enc_log_likelihood)
    # dec_loss = dec_criterion_reW(dec_kl, (it + 1), dec_log_likelihood)

    encoder.zero_grad()
    decoder.zero_grad()
    enc_loss.backward(retain_graph=True)
    enc_optim.step()

    encoder.zero_grad()
    decoder.zero_grad()
    dec_loss.backward(retain_graph=True)
    dec_optim.step()

    # ======== Train Discriminator ======== #
    frozen_params(decoder)
    frozen_params(encoder)
    free_params(discriminator)


    z_sam = (torch.randn(args.batch_size, args.n_z) * args.sigma_z).to(device)  # images.size()[0] -> 100
    d_sam = discriminator(z_sam)
    d_loss_real = bcewl(d_sam, real_labels)

    z_enc = encoder(x)
    d_enc = discriminator(z_enc)
    d_loss_fake = bcewl(d_enc, fake_labels)
    # dis_loss =  (-torch.log(d_sam).mean() - torch.log(1 - d_enc).mean())
    # dis_loss = args.LAMBDA * (-torch.log(d_sam + c).sum() + torch.log(d_enc + c).sum())
    dis_loss = d_loss_real + d_loss_fake
    # print("dis_loss",dis_loss.item())
    encoder.zero_grad()
    decoder.zero_grad()
    discriminator.zero_grad()
    dis_loss.backward()
    dis_optim.step()

    if (it + 1) % args.interval == 0:
        x_rec = dec_sample()
        # samples.append(x_rec)
        # samples.append(x.cpu().numpy())
        plot(x_rec, title =' Iteration {}'.format(it + 1))
        print("iteration [{}/{}], enc_Loss: {:.4f} ,dec_Loss: {:.4f}, dis_Loss: {:.4f}, enc_kl: {:.4f}, dec_kl: {:.4f},"
              .format(it + 1, args.iterations, enc_log_likelihood.item(), dec_log_likelihood.item(), dis_loss.item(), enc_kl.item(), dec_kl.item()))


#plot_samples(samples)