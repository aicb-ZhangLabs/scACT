import argparse


def get_common_parser():
    parser = argparse.ArgumentParser()
    # task setting
    parser.add_argument("--task_type", type=str, choices=['train', 'eval', 'embed'], required=True)
    parser.add_argument("--train_type", type=str, choices=['rna', 'atac', 'joint', 'joint_pseudo'], required=True)
    parser.add_argument("--generator_type", type=str, choices=['vae', 'ae'], required=True)
    parser.add_argument("--git_info", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)  # 128

    # training parameter setting
    parser.add_argument("--lr_g", type=float, default=1e-4,  # 1e-4
                        help="adam: learning rate for G")
    parser.add_argument("--lr_d", type=float, default=1e-4,  # 1e-4
                        help="adam: learning rate for D")
    parser.add_argument("--lr_ae", type=float, default=1e-4,  # 1e-4
                        help="adam: learning rate for AE")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--atac_lambda", type=float, default=1, help="reg for ATAC cyclic loss")
    parser.add_argument("--lamda1", type=float, default=1.0,
                        help="reg for cycle consistency")
    parser.add_argument("--lamda3", type=float, default=1.0,  # 1.0
                        help="reg for adversarial loss")
    parser.add_argument("--lamda_rna_kl", type=float, default=1.0,
                        help="reg for imt kld")
    parser.add_argument("--lamda_atac_kl", type=float, default=1.0,
                        help="reg for txt kld")
    parser.add_argument("--lamda_atac_recon", type=float, default=1.0,
                        help="reg for txt reconstruction")
    parser.add_argument("--gan_type", type=str, default='naive',
                        choices=['naive', 'wasserstein'])
    parser.add_argument("--grad_clip_value", type=float, default=5.,
                        help="gradient clipping")
    parser.add_argument("--weight_clip_value", type=float, default=0.05,
                        help="weight clipping")
    parser.add_argument('--update_p_freq', type=int, default=10)
    parser.add_argument('--update_d_freq', type=int, default=5)
    parser.add_argument('--tol', type=int, default=1e-3)
    parser.add_argument('--save_ratio', type=float, default=0.75)
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--eval_freq', type=int, default=50)
    parser.add_argument('--log_freq', type=int, default=30)
    parser.add_argument('--test_freq', type=int, default=1)
    parser.add_argument("--pretrain", type=str, default='None',
                        choices=['rna', 'atac', 'load_ae', 'load_all', 'None'])
    parser.add_argument("--warmup_epochs", type=int, default=10,
                        help="warm-up step number before training (-1 means no warm-up)")
    parser.add_argument("--scheduler_patience", type=int, default=50,
                        help="learning rate scheduler patience")
    parser.add_argument("--affine_num", type=int, default=9,
                        help="cluster mapping affine num")

    # file setting
    parser.add_argument("--dataset", type=str, default='pbmc_10k')
    parser.add_argument("--data_dir", type=str, default='../data/',
                        help='must contain train_rna.npz, train_atac.npz, test_rna.npz, test_atac.npz, and the corresponding _chromosome.txt files')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--cpt_dir', type=str, default='cpt/',
                        help="dir for saved checkpoint")
    parser.add_argument('--rna_cptpath', type=str, default='cpt/',
                        help="path to load rna AE checkpoint")
    parser.add_argument('--atac_cptpath', type=str, default='cpt/',
                        help="path to load txt AE checkpoint")
    parser.add_argument('--dm2c_cptpath', type=str, default='cpt/',
                        help="path to load dm2c checkpoint")
    parser.add_argument('--embedding_label_path', type=str, default='label.txt',
                        help="path to load label path")
    parser.add_argument('--embedding_pseudo_label_path', type=str, default='pseudo_label.txt',
                        help="path to load pseudo label path")

    # device setting
    parser.add_argument("--n_cpu", type=int, default=8, help="# of cpu threads during batch generation")
    parser.add_argument("--seed", type=int, default=2018)
    return parser
