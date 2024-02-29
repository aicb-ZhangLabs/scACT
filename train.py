import os
import time

import numpy as np
import torch
import pandas as pd

from src.args import get_common_parser
from src.model import MultimodalGAN
from src.utils import SingleModalityDataSet, check_dir_exist, plot_joint_embedding_umap, plot_single_embedding_umap


def main():
    parser = get_common_parser()
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    print(f'device: {device}')

    # reproducible
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.benchmark = True

    # preload the dataset
    sample_file_path = os.path.join(args.data_dir, 'samples.txt')
    if os.path.exists(sample_file_path) == False:
        sample_file_path = None
    dataset_rna = SingleModalityDataSet(
        os.path.join(args.data_dir, 'train_rna.npz'), os.path.join(args.data_dir, 'train_rna_chromosome.txt'), file_sample = sample_file_path
    )
    dataset_atac = SingleModalityDataSet(
        os.path.join(args.data_dir, 'train_atac.npz'), os.path.join(args.data_dir, 'train_atac_chromosome.txt'), file_sample = sample_file_path
    )

    # model settings
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
    config['n_samples'] = 1
    if sample_file_path is not None:
        samples = np.loadtxt(sample_file_path, dtype = str, delimiter = '\t')
        config['n_samples'] = np.unique(samples[:, 1]).shape[0]
    config['cuda'] = use_cuda
    config['device'] = device
    current_time = time.strftime(
        "%Y-%m-%d-%H-%M-%S",
        time.localtime(time.time())
    )

    # get experiment name
    experiment_name = f'{current_time}-{args.task_type}-{args.train_type}-{args.generator_type}-{args.dataset}-ep{args.n_epochs}'

    # set cpt and log dirs
    config['log_file'] = f'{experiment_name}.log'
    check_dir_exist(args.log_dir)
    check_dir_exist(args.cpt_dir)
    args.cpt_dir = os.path.join(args.cpt_dir, experiment_name)
    os.mkdir(args.cpt_dir)

    # prepare labels
    label_frame = pd.read_csv(args.embedding_label_path, delimiter='\t', header=None)
    labels = label_frame.iloc[:, 1].tolist()

    if sample_file_path is not None:
        sample_frame = pd.read_csv(os.path.join(args.data_dir, 'samples.txt'), delimiter = '\t', header = None)
        samples = sample_frame.iloc[:, 1].tolist()

    # init model
    model = MultimodalGAN(args, config)
    if use_cuda:
        model.to_cuda()

    # pretrain the autoencoders or load weights
    if args.pretrain == 'rna':
        model.pretrain('rna')
    elif args.pretrain == 'atac':
        model.pretrain('atac')
    elif args.pretrain == 'load_all':
        model.load_cpt(args.dm2c_cptpath)
    elif args.pretrain == 'load_ae':
        model.load_pretrain_cpt(args.rna_cptpath, 'rna', only_weight=True)
        model.load_pretrain_cpt(args.atac_cptpath, 'atac', only_weight=True)
    else:
        model.logger.info(f'Without loading any pretrain module in the {args.train_type} model...')

    if args.task_type in ['train', 'eval']:
        model.logger.info(f'Start to {args.task_type} on the {args.train_type} model...')

        # warm-up
        if args.warmup_epochs > 0:
            model.logger.info(f'Starting warm-up with {args.warmup_epochs} epochs...')
            if args.train_type == 'atac':
                model.atac_warm_up()
            elif args.train_type == 'rna':
                model.rna_warm_up()
            elif args.train_type == 'joint':
                model.atac_warm_up()
                model.rna_warm_up()

        # training
        first_epoch = model.epoch + 1 if hasattr(model, 'epoch') else 0
        for epoch in range(first_epoch, args.n_epochs):
            loss_log = model.train(epoch, args.train_type)

            # log loss
            model.logger.info(f"Loss at epoch {epoch}: {' | '.join([f'{k}: {v}' for k, v in loss_log.items()])}")
            # write tensorboard
            for k, v in loss_log.items():
                model.writer.add_scalar(f'Train_{args.train_type}/{k}', v, epoch)

            # visualize umap and evaluate score
            if (epoch + 1) % args.eval_freq == 0:
                model.logger.info(f"Evaluating umap and score...")
                umap_config = {
                    'random_state': args.seed,
                    'n_neighbors': 30,
                    'min_dist': 0.3,
                    'n_components': 2,
                    'metric': 'cosine'
                }

                model.logger.info(f"Run umap learning with {umap_config}")
                with torch.no_grad():
                    if args.train_type in ['joint', 'joint_pseudo']:
                        rna2rna_latent, rna2atac_latent = model.embedding(
                            model.test_loader_rna, source_modal='rna')
                        atac2rna_latent, atac2atac_latent = model.embedding(
                            model.test_loader_atac, source_modal='atac')

                        rna_plot_results = plot_single_embedding_umap(
                            'rna2rna', rna2rna_latent, labels, umap_config
                        )
                        atac_plot_results = plot_single_embedding_umap(
                            'atac2atac', atac2atac_latent, labels, umap_config
                        )
                        rna_joint_plot_results = plot_joint_embedding_umap(
                            'rna_joint', rna2rna_latent, atac2rna_latent, labels, umap_config
                        )
                        atac_joint_plot_results = plot_joint_embedding_umap(
                            'atac_joint', atac2atac_latent, rna2atac_latent, labels, umap_config
                        )
                        figure_dict = {
                            'rna2rna': rna_plot_results['plot'][0],
                            'atac2atac': atac_plot_results['plot'][0],
                            'rna_joint': rna_joint_plot_results['plot'][0],
                            'atac_joint': atac_joint_plot_results['plot'][0]
                        }
                        score_dict = {
                            'rna2rna_silhouette': rna_plot_results['silhouette_score'],
                            'atac2atac_silhouette': atac_plot_results['silhouette_score'],
                            'rna_joint_silhouette': rna_joint_plot_results['silhouette_score'],
                            'atac_joint_silhouette': atac_joint_plot_results['silhouette_score'],
                        }
                    elif args.train_type == 'rna':
                        rna2rna_latent, _ = model.embedding(
                            model.test_loader_rna, source_modal='rna')
                        plot_results = plot_single_embedding_umap(
                            'rna2rna', rna2rna_latent, labels, umap_config
                        )
                        figure_dict = {'rna2rna': plot_results['plot'][0]}
                        score_dict = {'rna2rna_silhouette': plot_results['silhouette_score']}
                    elif args.train_type == 'atac':
                        _, atac2atac_latent = model.embedding(
                            model.test_loader_atac, source_modal='atac')
                        plot_results = plot_single_embedding_umap(
                            'atac2atac', atac2atac_latent, labels, umap_config
                        )
                        #plot_results_by_samples = plot_single_embedding_umap(
                        #    'atac2atac', atac2atac_latent, samples, umap_config
                        #)
                        figure_dict = {'atac2atac': plot_results['plot'][0]}#, 'atac2atac_sample': plot_results_by_samples['plot'][0]}
                        score_dict = {'atac2atac_silhouette': plot_results['silhouette_score']}

                    for k, v in figure_dict.items():
                        model.writer.add_figure(
                            f'Train_{args.train_type}_umap/{k}', v, epoch
                        )
                    for k, v in score_dict.items():
                        model.writer.add_scalar(
                            f'Train_{args.train_type}_score/{k}', v, epoch
                        )
                        model.logger.info(f'Evaluated {k}: {v}')

    elif args.task_type == 'embed':
        rna, atac = model.embedding(model.test_loader_atac, source_modal='atac')
        np.savetxt(f'{args.cpt_dir}/rna_latent.txt', rna, delimiter=',')


if __name__ == '__main__':
    main()
