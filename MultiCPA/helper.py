# Author: Kemal Inecik
# Email: k.inecik@gmail.com

import scanpy as sc
import pandas as pd


def model_importer(_id, df, model_dir, dataset_relative_to, model_created='TotalComPert'):
    import os
    import torch
    from MultiCPA.train import prepare_compert
    
    args = {
        'dataset_path': "config.dataset.dataset_args.dataset_path",  # full path to the anndata dataset
        'cell_type_key': 'config.dataset.dataset_args.cell_type_key', # necessary field for cell types. Fill it with a dummy variable if no celltypes present.
        'split_key': 'config.dataset.dataset_args.split_key',  # necessary field for train, test, ood splits.
        'perturbation_key': 'config.dataset.dataset_args.perturbation_key',  # necessary field for perturbations
        'dose_key': 'config.dataset.dataset_args.dose_key',  # necessary field for dose. Fill in with dummy variable if dose is the same.
        'checkpoint_freq': 'config.training.checkpoint_freq',  # checkoint frequencty to save intermediate results
        'max_epochs': 'config.training.num_epochs',  # maximum epochs for training
        'max_minutes': 'config.training.max_minutes',  # maximum computation time
        'patience': 'config.model.model_args.patience',  # patience for early stopping
        'loss_ae': 'config.model.model_args.loss_ae',  # loss (currently only gaussian loss is supported)
        'doser_type': 'config.model.model_args.doser_type',  # non-linearity for doser function
        'save_dir': 'config.training.save_dir',  # directory to save the model
        'decoder_activation': 'config.model.model_args.decoder_activation',  # last layer of the decoder
        'seed': 'config.seed',  # random seed
        'raw_counts_key': 'config.dataset.dataset_args.counts_key',  # necessary field for nb loss. Name of the layer storing raw gene counts.
        'is_vae': 'config.model.model_args.is_vae', # using a vae or ae model
        'protein_key': 'config.dataset.dataset_args.proteins_key', # name of the field storing the protein data in adata.obsm[proteins_key]
        'raw_protein_key': 'config.dataset.dataset_args.raw_proteins_key', # necessary field for nb loss. Name of the field storing the raw protein data in adata.obsm[protein_expression_raw]
    } #'hparams': "",  # autoencoder architecture
    
    model_experiment = df.loc[_id]
    if model_experiment["status"] == 0:
        raise NotImplementedError
    exp_id = model_experiment['result.exp_id']
    model_name = os.path.join(model_dir, f"{exp_id}_last.pt")
    
    for i in args:
        args[i] = model_experiment[args[i]]
    state, hypers, history = torch.load(model_name, map_location=torch.device('cpu'))
    args['hparams'] = hypers
    args['dataset_path'] = os.path.join(dataset_relative_to, args['dataset_path'])
    autoencoder, datasets = prepare_compert(args, state_dict=state, model=model_created)
    for p in autoencoder.parameters():  # reset requires_grad
        p.requires_grad = False
    autoencoder.eval();
    return autoencoder, datasets, state, history, args


def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added='rank_genes_groups_cov',
    return_dict=False,
):

    """
    Function that generates a list of differentially expressed genes computed
    separately for each covariate category, and using the respective control
    cells as reference.

    Usage example:

    rank_genes_groups_by_cov(
        adata,
        groupby='cov_product_dose',
        covariate_key='cell_type',
        control_group='Vehicle_0'
    )

    Parameters
    ----------
    adata : AnnData
        AnnData dataset
    groupby : str
        Obs column that defines the groups, should be
        cartesian product of covariate_perturbation_cont_var,
        it is important that this format is followed.
    control_group : str
        String that defines the control group in the groupby obs
    covariate : str
        Obs column that defines the main covariate by which we
        want to separate DEG computation (eg. cell type, species, etc.)
    n_genes : int (default: 50)
        Number of DEGs to include in the lists
    rankby_abs : bool (default: True)
        If True, rank genes by absolute values of the score, thus including
        top downregulated genes in the top N genes. If False, the ranking will
        have only upregulated genes at the top.
    key_added : str (default: 'rank_genes_groups_cov')
        Key used when adding the dictionary to adata.uns
    return_dict : str (default: False)
        Signals whether to return the dictionary or not

    Returns
    -------
    Adds the DEG dictionary to adata.uns

    If return_dict is True returns:
    gene_dict : dict
        Dictionary where groups are stored as keys, and the list of DEGs
        are the corresponding values

    """

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        print(cov_cat)
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate]==cov_cat]

        #compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes
        )

        #add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict


