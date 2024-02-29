# scACT: Accurate Cross-modality Translation via Cycle-consistent Training from Unpaired Single-cell Data

Single-cell sequencing technologies have transformed genomics by allowing the simultaneous profiling of various molecular modalities within individual cells. The integration of these modalities, especially cross-modality translation, provides profound insights into cellular regulatory mechanisms. Despite the existence of numerous methods for cross-modality translation, their applicability is often limited due to reliance on scarce high-quality co-assay data.

In response to this challenge, we present scACT, a deep generative model tailored for extracting cross-modality biological insights from unpaired single-cell data. scACT addresses three key challenges:

- **Alignment of Unpaired Multi-Modal Data:** Leveraging adversarial training, scACT aligns diverse single-cell modalities, allowing for meaningful comparisons and joint analysis.

- **Cross-Modality Translation without Prior Knowledge:** scACT employs cycle-consistent training to facilitate cross-modality translation without the need for prior knowledge, enhancing its versatility and applicability.

- **Interpretable Regulatory Interconnections Explorations:** Through in-silico perturbations, scACT enables the exploration of interpretable regulatory interconnections, shedding light on the underlying biological mechanisms.

## Architecture

To learn the latent space from those single-cell sequencing data, we adopted two types of variational autoencoders (VAEs), each corresponding to a different type of data with unique design specific to that data type.

Since scRNA-seq data is relatively small, and interactions across different genes are common, we used a fully connected structure in the encoder and decoder. Previously, people assume a Poisson distribution of the scRNA-seq raw counts. However, due to its characteristics, recently more and more scholars find benefits mapping it to a negative binomial distribution. Plus its highly sparsed nature, we decided to use a Zero-Inflated Negative Binomial (ZINB) Distribution in the decoder design.

On the other hand, scATAC-seq data are usually with ultra-high dimensionality and sparsity, making them even more difficult to analyze. However, interchromosomal talk is limited due to the biological nature and physical shape of DNA. Therefore, we split the scATAC-seq data by chromosomes and limit (not prohibit though) their cross-links, effectively reducing the number of parameters dramatically. We model the output of scATAC-seq as a Bernoulli distributtion as the openness of a chromatin has two states: open or closed.

## Architecture
scACT's architecture is structured to address the aforementioned challenges:

- **Adversarial Training Module:** Aligns unpaired multi-modal data, minimizing distributional discrepancies between different molecular profiles.

- **Cycle-Consistent Training Module:** Enables cross-modality translation by enforcing consistency between the translated data and the original data, ensuring meaningful and accurate translations.

- **In-Silico Perturbation Module:** Facilitates the exploration of regulatory interconnections by simulating perturbations within the generated data, allowing for the identification of key regulatory elements.

## Usage

### Obtaining the model

To utilize scACT for advancing single-cell omics data processing and analysis, refer to the following steps:

1. Clone the repository:

`git clone https://github.com/your-username/scACT.git`

2. Install dependencies: 

``cd scACT
pip install -r requirements.txt``

3. Follow the provided examples and documentation to apply scACT to your single-cell datasets.

### Training

scACT supports three training mode: scATAC-seq only, scRNA-seq only, and joint training. The single-modality only modes are useful to pretrain the model on a specific single-modality dataset to save time and improve accuracy.

To train the model, you may use the `train.py` with the following important parameters:

- `train_type` choose from `rna`, `atac`, or `joint`
- `data_dir`: the path to the dataset directory
- `lr_g`: the learning rate used to train the cross-modality transformation functions
- `lr_d`: the learning rate used to train the discriminators
- `lr_ae`: the learning rate to train the single-modality generators (the autoencoders)

## Embeddings

After training, scACT generates 4 sets of embeddings from the training dataset: RNA only, ATAC only, RNA transformed to ATAC, and ATAC transformed to RNA. You may also generate using `embed.py` to generate embeddings, supporting existing or new data. It also gives a cpt file, storing the saved model parameters. 

## Downstream analysis

The generated embedding can be used towards downstream analyses. One can simply treat it as another method of dimensionality reduction.
