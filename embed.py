import os
import time

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import LabelEncoder

from src.args import get_common_parser
from src.model import MultimodalGAN
from src.utils import check_dir_exist, SingleModalityDataSet, plot_single_embedding_umap, plot_joint_embedding_umap


if __name__ == '__main__':
    parser = get_common_parser()
    parser.add_argument('--cptpath', type=str, default='cpt/', help="path to load dm2c checkpoint")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # reproducible
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True

    # preload the dataset
    dataset_rna = SingleModalityDataSet(os.path.join(
        args.data_dir, 'train_rna.npz'), os.path.join(args.data_dir, 'train_rna_chromosome.txt'))
    dataset_atac = SingleModalityDataSet(os.path.join(
        args.data_dir, 'train_atac.npz'), os.path.join(args.data_dir, 'train_atac_chromosome.txt'))

    config = dict()
    config['rna_input_dim'] = dataset_rna.get_features()
    config['atac_input_dim'] = dataset_atac.get_features()
    config['rna_hiddens'] = 20
    config['atac_hiddens'] = 20
    config['n_clusters'] = 10
    config['rna2atac_hiddens'] = [20, 128, 20]
    config['atac2rna_hiddens'] = [20, 128, 20]
    # if the data include corresponding filename for each sample feature
    config['has_filename'] = True
    config['batchnorm'] = True
    config['cuda'] = use_cuda
    config['device'] = device
    current_time = time.strftime(
        "%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    config['log_file'] = current_time + '.log'

    args.cpt_dir = f'{args.cpt_dir}/embedding_{current_time}'

    check_dir_exist(args.log_dir)
    check_dir_exist(args.cpt_dir)

    model = MultimodalGAN(args, config)

    if use_cuda:
        model.to_cuda()

    model.load_cpt(args.cptpath)

    if use_cuda:
        model.to_cuda()

    # prepare labels
    label_frame = pd.read_csv(args.embedding_label_path, delimiter='\t', header=None)
    labels = label_frame.iloc[:, 1].tolist()
    label_encoder = LabelEncoder()
    label_array = label_encoder.fit_transform(labels)

    model.logger.info(f"Loaded model trained on {args.train_type}")
    if args.train_type == 'joint':
        rna2rna_latent, rna2atac_latent = model.embedding(
            model.test_loader_rna, source_modal='rna')
        atac2rna_latent, atac2atac_latent = model.embedding(
            model.test_loader_atac, source_modal='atac')

        embedding_works = {
            'rna2rna_latent': rna2rna_latent,
            'rna2atac_latent': rna2atac_latent,
            'atac2rna_latent': atac2rna_latent,
            'atac2atac_latent': atac2atac_latent
        }
    elif args.train_type == 'rna':
        rna2rna_latent, _ = model.embedding(
            model.test_loader_rna, source_modal='rna')
        embedding_works = {
            'rna2rna_latent': rna2rna_latent
        }
    elif args.train_type == 'atac':
        _, atac2atac_latent = model.embedding(
            model.test_loader_atac, source_modal='atac')

        embedding_works = {
            'atac2atac_latent': atac2atac_latent
        }

    # run umap
    umap_config = {
        'random_state': args.seed,
        'n_neighbors': 30,
        'min_dist': 0.3,
        'n_components': 2,
        'metric': 'cosine'
    }
    for embedding_name, embedding in embedding_works.items():
        model.logger.info(f"Run umap learning on {embedding_name} with {umap_config}")
        plot_results = plot_single_embedding_umap(embedding_name, embedding, labels, umap_config)
        fig, _ = plot_results['plot']
        fig.tight_layout()
        fig.savefig(f"{args.cpt_dir}/{embedding_name}_umap.pdf")

        # save embedding and labels
        umap_embedded = plot_results['umap_embedding']
        np.savetxt(f"{args.cpt_dir}/{embedding_name}_embeding.txt", embedding)
        np.savetxt(f"{args.cpt_dir}/{embedding_name}_umap_embeding.txt", umap_embedded)

    # joint embedding needs extra umap figures
    if args.train_type == 'joint':
        joint_embedding_works = {
            'rna_joint': (rna2rna_latent, atac2rna_latent),
            'atac_joint': (atac2atac_latent, rna2atac_latent)
        }
        for embedding_name, (embedding1, embedding2) in joint_embedding_works.items():
            model.logger.info(f"Run umap learning on {embedding_name} with {umap_config}")

            plot_results = plot_joint_embedding_umap(embedding_name, embedding1, embedding2, labels, umap_config)
            fig, _ = plot_results['plot']
            fig.tight_layout()
            fig.savefig(f"{args.cpt_dir}/{embedding_name}_umap.pdf")

            # save embedding and labels
            umap_embedded = plot_results['umap_embedding']
            np.savetxt(f"{args.cpt_dir}/{embedding_name}_embeding.txt", embedding)
            np.savetxt(f"{args.cpt_dir}/{embedding_name}_umap_embeding.txt", umap_embedded)

