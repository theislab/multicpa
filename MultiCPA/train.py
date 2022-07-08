# Author: Kemal Inecik
# Email: k.inecik@gmail.com

import os
import json
import argparse

import torch
import numpy as np
from collections import defaultdict

try:
    from data import load_dataset_splits
    from model import ComPert, TotalComPert, PoEComPert, TotalPoEComPert
except (ModuleNotFoundError, ImportError):
    from MultiCPA.data import load_dataset_splits
    from MultiCPA.model import ComPert, TotalComPert, PoEComPert, TotalPoEComPert

from sklearn.metrics import r2_score, balanced_accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import time

def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


def evaluate_disentanglement(autoencoder, dataset, nonlinear=False):
    """
    Given a ComPert model, this function measures the correlation between
    its latent space and 1) a dataset's drug vectors 2) a datasets covariate
    vectors.

    """
    if autoencoder.loss_ae == 'gauss':
        latent_basal = autoencoder.get_latent(
            dataset.genes,
            dataset.drugs,
            dataset.cell_types,
            proteins=dataset.proteins,
            return_latent_treated=False)
    elif autoencoder.loss_ae == 'nb':
        latent_basal = autoencoder.get_latent(
            dataset.raw_genes,
            dataset.drugs,
            dataset.cell_types,
            proteins=dataset.raw_proteins,
            return_latent_treated=False)
    else:
        raise ValueError("Autoencoder loss must be either 'nb' or 'gauss'.")

    latent_basal = latent_basal.detach().cpu().numpy()

    if nonlinear:
        clf = KNeighborsClassifier(
            n_neighbors=int(np.sqrt(len(latent_basal))))
    else:
        clf = LogisticRegression(solver="liblinear",
                                 multi_class="auto",
                                 max_iter=10000)

    pert_scores = cross_val_score(
        clf,
        StandardScaler().fit_transform(latent_basal), dataset.drugs_names,
        scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)

    if len(np.unique(dataset.cell_types_names)) > 1:
        cov_scores = cross_val_score(
            clf,
            StandardScaler().fit_transform(latent_basal), dataset.cell_types_names,
            scoring=make_scorer(balanced_accuracy_score), cv=5, n_jobs=-1)
        return np.mean(pert_scores), np.mean(cov_scores)
    else:
        return np.mean(pert_scores), 0


def evaluate_r2(autoencoder, dataset, genes_control, proteins_control, sample=False):
    """
    Measures different quality metrics about an ComPert `autoencoder`, when
    tasked to translate some `genes_control` into each of the drug/cell_type
    combinations described in `dataset`.

    Considered metrics are R2 score about means and variances for all genes, as
    well as R2 score about means and variances about differentially expressed
    (_de) genes.

    If protein data is available, the R2 score for the protein data is computed separately.

    For computing the R2 score, one can take the predicted mean or sample from the decoder distribution.
    """

    mean_score_genes, var_score_genes, mean_score_genes_de, var_score_genes_de, \
    mean_score_proteins, var_score_proteins = [], [], [], [], [], []
    num, dim_genes = genes_control.size(0), genes_control.size(1)

    if autoencoder.num_proteins is not None:
        dim_proteins = proteins_control.size(1)

    for pert_category in np.unique(dataset.pert_categories):
        # pert_category category contains: 'celltype_perturbation_dose' info
        de_idx = np.where(
            dataset.var_names.isin(
                np.array(dataset.de_genes[pert_category])))[0]

        idx = np.where(dataset.pert_categories == pert_category)[0]

        if len(idx) > 30:
            emb_drugs = dataset.drugs[idx][0].view(
                1, -1).repeat(num, 1).clone()
            emb_cts = dataset.cell_types[idx][0].view(
                1, -1).repeat(num, 1).clone()

            if sample:
                # sample from the decoder distribution
                gene_predictions, protein_predictions = autoencoder.sample(
                    genes_control, emb_drugs, emb_cts, proteins_control)

                mean_predict_genes = gene_predictions[:, :dim_genes]
                var_predict_genes = gene_predictions[:, dim_genes:]

                if autoencoder.num_proteins is not None:
                    mean_predict_proteins = protein_predictions[:, :dim_proteins]
                    var_predict_proteins = protein_predictions[:, dim_proteins:]
            else:
                # take the predicted means instead of sampling from the decoder distribution
                gene_predictions, protein_predictions = autoencoder.predict(
                    genes_control, emb_drugs, emb_cts, proteins_control)

                if isinstance(gene_predictions, list):
                    gene_predictions = gene_predictions[-1]

                gene_predictions = gene_predictions.detach().cpu()

                mean_predict_genes = gene_predictions[:, :dim_genes]
                if autoencoder.loss_ae == 'nb':
                    # compute variance based on dispersion
                    var_predict_genes = mean_predict_genes + (mean_predict_genes ** 2) / \
                                        gene_predictions[:, dim_genes:]
                else:
                    # take the predicted variance estimates
                    var_predict_genes = gene_predictions[:, dim_genes:]

                if autoencoder.num_proteins is not None:
                    if isinstance(protein_predictions, list):
                        protein_predictions = protein_predictions[-1]

                    protein_predictions = protein_predictions.detach().cpu()

                    mean_predict_proteins = protein_predictions[:, :dim_proteins]
                    if autoencoder.loss_ae == 'nb':
                        # compute variance based on dispersion
                        var_predict_proteins = mean_predict_proteins + (mean_predict_proteins ** 2) / \
                                            protein_predictions[:, dim_proteins:]
                    else:
                        # take the predicted variance estimates
                        var_predict_proteins = protein_predictions[:, dim_proteins:]

            # estimate metrics only for reasonably-sized drug/cell-type combos
            if autoencoder.loss_ae == 'gauss':
                y_true_genes = dataset.genes[idx, :].numpy()
            elif autoencoder.loss_ae == 'nb':
                y_true_genes = dataset.raw_genes[idx, :].numpy()
            else:
                raise ValueError("Autoencoder loss must be either 'nb' or 'gauss'.")

            # true means and variances
            yt_m_genes = y_true_genes.mean(axis=0)
            yt_v_genes = y_true_genes.var(axis=0)
            # predicted means and variances
            if sample:
                yp_m_genes = mean_predict_genes.mean(0)
                yp_v_genes = var_predict_genes.var(0)
            else:
                yp_m_genes = mean_predict_genes.mean(0)
                yp_v_genes = var_predict_genes.mean(0)

            
            mean_score_genes.append(r2_score(yt_m_genes, yp_m_genes))
            var_score_genes.append(r2_score(yt_v_genes, yp_v_genes))

            mean_score_genes_de.append(r2_score(yt_m_genes[de_idx], yp_m_genes[de_idx]))
            var_score_genes_de.append(r2_score(yt_v_genes[de_idx], yp_v_genes[de_idx]))

            if autoencoder.num_proteins is not None:
                # estimate metrics only for reasonably-sized drug/cell-type combos
                if autoencoder.loss_ae == 'gauss':
                    y_true_proteins = dataset.proteins[idx, :].numpy()
                elif autoencoder.loss_ae == 'nb':
                    y_true_proteins = dataset.raw_proteins[idx, :].numpy()

                # true means and variances
                yt_m_proteins = y_true_proteins.mean(axis=0)
                yt_v_proteins = y_true_proteins.var(axis=0)
                # predicted means and variances
                if sample:
                    yp_m_proteins = mean_predict_proteins.mean(0)
                    yp_v_proteins = var_predict_proteins.var(0)
                else:
                    yp_m_proteins = mean_predict_proteins.mean(0)
                    yp_v_proteins = var_predict_proteins.mean(0)

                if len(yt_m_proteins) > 0:
                    mean_score_proteins.append(r2_score(yt_m_proteins, yp_m_proteins))
                    var_score_proteins.append(r2_score(yt_v_proteins, yp_v_proteins))
                else:
                    mean_score_proteins.append(0)
                    var_score_proteins.append(0)


    return [np.mean(s) if len(s) else -1
            for s in [mean_score_genes, var_score_genes, mean_score_genes_de,
                      var_score_genes_de, mean_score_proteins, var_score_proteins]]


def evaluate(autoencoder, datasets):
    """
    Measure quality metrics using `evaluate()` on the training, test, and
    out-of-distributiion (ood) splits.
    """

    autoencoder.eval()
    if autoencoder.loss_ae == 'gauss':
        # use the normalized counts to evaluate the model
        with torch.no_grad():
            stats_test = evaluate_r2(
                autoencoder,
                datasets["test_treated"],
                datasets["test_control"].genes,
                datasets["test_control"].proteins)

            stats_disent_pert, stats_disent_cov = evaluate_disentanglement(
                autoencoder, datasets["test"])

            evaluation_stats = {
                "training": evaluate_r2(
                    autoencoder,
                    datasets["training_treated"],
                    datasets["training_control"].genes,
                    datasets["training_control"].proteins),
                "test": stats_test,
                "ood": evaluate_r2(
                    autoencoder,
                    datasets["ood"],
                    datasets["test_control"].genes,
                    datasets["test_control"].proteins),
                "perturbation disentanglement": stats_disent_pert,
                "optimal for perturbations": 1 / datasets['test'].num_drugs,
                "covariate disentanglement": stats_disent_cov,
                "optimal for covariates": 1 / datasets['test'].num_cell_types,
            }
    elif autoencoder.loss_ae == 'nb':
        # use the raw counts to evaluate the model
        with torch.no_grad():
            stats_test = evaluate_r2(
                autoencoder,
                datasets["test_treated"],
                datasets["test_control"].raw_genes,
                datasets["test_control"].raw_proteins)

            stats_disent_pert, stats_disent_cov = evaluate_disentanglement(
                autoencoder, datasets["test"])

            evaluation_stats = {
                "training": evaluate_r2(
                    autoencoder,
                    datasets["training_treated"],
                    datasets["training_control"].raw_genes,
                    datasets["training_control"].raw_proteins),
                "test": stats_test,
                "ood": evaluate_r2(
                    autoencoder,
                    datasets["ood"],
                    datasets["test_control"].raw_genes,
                    datasets["test_control"].raw_proteins),
                "perturbation disentanglement": stats_disent_pert,
                "optimal for perturbations": 1 / datasets['test'].num_drugs,
                "covariate disentanglement": stats_disent_cov,
                "optimal for covariates": 1 / datasets['test'].num_cell_types,
            }
    else:
        raise ValueError("Autoencoder loss must be either 'nb' or 'gauss'.")

    autoencoder.train()
    return evaluation_stats


def prepare_compert(args, model='ComPert', state_dict=None):
    """
    Instantiates autoencoder and dataset to run an experiment.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = load_dataset_splits(
        args["dataset_path"],
        args["perturbation_key"],
        args["dose_key"],
        args["cell_type_key"],
        args["split_key"],
        args["raw_counts_key"],
        args["protein_key"],
        args["raw_protein_key"],
    )

    if model == 'ComPert':
        autoencoder = ComPert(
            datasets["training"].num_genes,
            datasets["training"].num_drugs,
            datasets["training"].num_cell_types,
            num_proteins=datasets["training"].num_proteins,
            device=device,
            seed=args["seed"],
            loss_ae=args["loss_ae"],
            doser_type=args["doser_type"],
            patience=args["patience"],
            hparams=args["hparams"],
            decoder_activation=args["decoder_activation"],
            is_vae=args["is_vae"],
        )
    elif model == 'TotalComPert':
        autoencoder = TotalComPert(
            datasets["training"].num_genes,
            datasets["training"].num_drugs,
            datasets["training"].num_cell_types,
            num_proteins=datasets["training"].num_proteins,
            device=device,
            seed=args["seed"],
            loss_ae=args["loss_ae"],
            doser_type=args["doser_type"],
            patience=args["patience"],
            hparams=args["hparams"],
            decoder_activation=args["decoder_activation"],
            is_vae=args["is_vae"],
        )
    elif model == 'PoEComPert':
        autoencoder = PoEComPert(
            datasets["training"].num_genes,
            datasets["training"].num_drugs,
            datasets["training"].num_cell_types,
            num_proteins=datasets["training"].num_proteins,
            device=device,
            seed=args["seed"],
            loss_ae=args["loss_ae"],
            doser_type=args["doser_type"],
            patience=args["patience"],
            hparams=args["hparams"],
            decoder_activation=args["decoder_activation"],
            is_vae=args["is_vae"],
        )
    elif model == 'TotalPoEComPert':
        autoencoder = TotalPoEComPert(
            datasets["training"].num_genes,
            datasets["training"].num_drugs,
            datasets["training"].num_cell_types,
            num_proteins=datasets["training"].num_proteins,
            device=device,
            seed=args["seed"],
            loss_ae=args["loss_ae"],
            doser_type=args["doser_type"],
            patience=args["patience"],
            hparams=args["hparams"],
            decoder_activation=args["decoder_activation"],
            is_vae=args["is_vae"],
        )
    else:
        raise NotImplementedError("The model architecture {} is not implemented!".format(model))

    if state_dict is not None:
        autoencoder.load_state_dict(state_dict)

    return autoencoder, datasets


def train_compert(args, model='ComPert', return_model=False):
    """
    Trains a ComPert autoencoder
    """

    autoencoder, datasets = prepare_compert(args, model)

    datasets.update({
        "loader_tr": torch.utils.data.DataLoader(
            datasets["training"],
            batch_size=autoencoder.hparams["batch_size"],
            shuffle=True)
    })

    pjson({"training_args": args})
    pjson({"autoencoder_params": autoencoder.hparams})

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for genes, drugs, cell_types, proteins, raw_genes, raw_proteins in datasets["loader_tr"]:
            minibatch_training_stats = autoencoder.update(
                genes, drugs, cell_types, proteins, raw_genes, raw_proteins, epoch, args["max_epochs"])

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in autoencoder.history.keys()):
                autoencoder.history[key] = []
            autoencoder.history[key].append(epoch_training_stats[key])
        autoencoder.history['epoch'].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        autoencoder.history['elapsed_time_min'] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: patience ran out OR
        # time ran out OR max epochs achieved
        stop = ellapsed_minutes > args["max_minutes"] or \
               (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            evaluation_stats = evaluate(autoencoder, datasets)
            for key, val in evaluation_stats.items():
                if not (key in autoencoder.history.keys()):
                    autoencoder.history[key] = []
                autoencoder.history[key].append(val)
            autoencoder.history['stats_epoch'].append(epoch)

            pjson({
                "epoch": epoch,
                "training_stats": epoch_training_stats,
                "evaluation_stats": evaluation_stats,
                "ellapsed_minutes": ellapsed_minutes
            })

            torch.save(
                (autoencoder.state_dict(), args, autoencoder.history),
                os.path.join(
                    args["save_dir"],
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch)))

            pjson({"model_saved": "model_seed={}_epoch={}.pt\n".format(
                args["seed"], epoch)})
            stop = stop or autoencoder.early_stopping(
                np.mean(evaluation_stats["test"])) #or autoencoder.specific_threshold(
                    #evaluation_stats["test"][0], epoch, epoch_thr=361, score_thr=0.15)  # 0->gene
            if stop:
                pjson({"early_stop": epoch})
                break

    if return_model:
        return autoencoder, datasets


def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser(description='Drug combinations.')
    # dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--perturbation_key', type=str, default="condition")
    parser.add_argument('--dose_key', type=str, default="dose_val")
    parser.add_argument('--cell_type_key', type=str, default="cell_type")
    parser.add_argument('--split_key', type=str, default="split")
    parser.add_argument('--loss_ae', type=str, default='gauss')
    parser.add_argument('--doser_type', type=str, default='sigm')
    parser.add_argument('--decoder_activation', type=str, default='linear')

    # ComPert arguments (see set_hparams_() in MultiCPA.model.ComPert)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hparams', type=str, default="")

    # training arguments
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--max_minutes', type=int, default=300)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--checkpoint_freq', type=int, default=20)

    # output folder
    parser.add_argument('--save_dir', type=str, required=True)
    # number of trials when executing MultiCPA.sweep
    parser.add_argument('--sweep_seeds', type=int, default=200)
    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    train_compert(parse_arguments())
