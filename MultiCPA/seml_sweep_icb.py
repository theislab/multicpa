# Author: Kemal Inecik
# Email: k.inecik@gmail.com

import sys
from sacred import Experiment
from collections import defaultdict
import json
import seml
import torch
import os
import time
import numpy as np

from data import load_dataset_splits
from model import ComPert, TotalComPert, PoEComPert, TotalPoEComPert
from train import evaluate

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


def pjson(s):
    """
    Prints a string in JSON format and flushes stdout
    """
    print(json.dumps(s), flush=True)


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, init_all=True):


        if init_all:
            self.init_all()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "dataset".
    @ex.capture(prefix="dataset")
    def init_dataset(self, dataset_args: dict):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="dataset ", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        self.datasets, self.dataset = load_dataset_splits(
            **dataset_args,
            return_dataset=True
        )

    @ex.capture(prefix="model")
    def init_model(self, model_type: str, model_args: dict):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f'# DEVICE: {device}')

        if model_type == 'ComPert':
            self.autoencoder = ComPert(
                self.datasets["training"].num_genes,
                self.datasets["training"].num_drugs,
                self.datasets["training"].num_cell_types,
                num_proteins=self.datasets["training"].num_proteins,
                device=device,
                seed=self.seed,
                loss_ae=model_args["loss_ae"],
                doser_type=model_args["doser_type"],
                patience=model_args["patience"],
                hparams=model_args["hparams"],
                decoder_activation=model_args["decoder_activation"],
                is_vae=model_args["is_vae"],
            )
        elif model_type == 'TotalComPert':
            self.autoencoder = TotalComPert(
                self.datasets["training"].num_genes,
                self.datasets["training"].num_drugs,
                self.datasets["training"].num_cell_types,
                num_proteins=self.datasets["training"].num_proteins,
                device=device,
                seed=self.seed,
                loss_ae=model_args["loss_ae"],
                doser_type=model_args["doser_type"],
                patience=model_args["patience"],
                hparams=model_args["hparams"],
                decoder_activation=model_args["decoder_activation"],
                is_vae=model_args["is_vae"],
            )
        elif model_type == 'PoEComPert':
            self.autoencoder = PoEComPert(
                self.datasets["training"].num_genes,
                self.datasets["training"].num_drugs,
                self.datasets["training"].num_cell_types,
                num_proteins=self.datasets["training"].num_proteins,
                device=device,
                seed=self.seed,
                loss_ae=model_args["loss_ae"],
                doser_type=model_args["doser_type"],
                patience=model_args["patience"],
                hparams=model_args["hparams"],
                decoder_activation=model_args["decoder_activation"],
                is_vae=model_args["is_vae"],
            )
        elif model_type == 'TotalPoEComPert':
            self.autoencoder = TotalPoEComPert(
                self.datasets["training"].num_genes,
                self.datasets["training"].num_drugs,
                self.datasets["training"].num_cell_types,
                num_proteins=self.datasets["training"].num_proteins,
                device=device,
                seed=self.seed,
                loss_ae=model_args["loss_ae"],
                doser_type=model_args["doser_type"],
                patience=model_args["patience"],
                hparams=model_args["hparams"],
                decoder_activation=model_args["decoder_activation"],
                is_vae=model_args["is_vae"],
            )

    def update_datasets(self):

        self.datasets.update({
            "loader_tr": torch.utils.data.DataLoader(
                self.datasets["training"],
                batch_size=self.autoencoder.hparams["batch_size"],
                shuffle=True)
        })
        # pjson({"training_args": args})
        pjson({"autoencoder_params": self.autoencoder.hparams})

    @ex.capture
    def init_all(self, seed):
        """
        Sequentially run the sub-initializers of the experiment.
        """

        self.seed = seed
        self.init_dataset()
        self.init_model()
        self.update_datasets()

    @ex.capture(prefix="training")
    def train(
        self,
        num_epochs: int,
        max_minutes: int,
        checkpoint_freq: int,
        ignore_evaluation: bool,
        save_checkpoints: bool,
        save_dir: str,
        save_last: bool,
    ):

        print(f"CWD: {os.getcwd()}")
        print(f"Save dir: {save_dir}")
        print(f"Is path?: {os.path.exists(save_dir)}")

        exp_id = ''.join(map(str, list(np.random.randint(0, 10, 30))))

        start_time = time.time()
        for epoch in range(num_epochs):
            epoch_training_stats = defaultdict(float)

            for genes, drugs, cell_types, proteins, raw_genes, raw_proteins in self.datasets["loader_tr"]:
                minibatch_training_stats = self.autoencoder.update(
                    genes, drugs, cell_types, proteins, raw_genes, raw_proteins, epoch, num_epochs)

                for key, val in minibatch_training_stats.items():
                    epoch_training_stats[key] += val

            for key, val in epoch_training_stats.items():
                epoch_training_stats[key] = val / len(self.datasets["loader_tr"])
                if not (key in self.autoencoder.history.keys()):
                    self.autoencoder.history[key] = []
                self.autoencoder.history[key].append(epoch_training_stats[key])
            self.autoencoder.history['epoch'].append(epoch)

            ellapsed_minutes = (time.time() - start_time) / 60
            self.autoencoder.history['elapsed_time_min'] = ellapsed_minutes

            # decay learning rate if necessary
            # also check stopping condition: patience ran out OR
            # time ran out OR max epochs achieved
            stop = ellapsed_minutes > max_minutes or \
                   (epoch == num_epochs - 1)

            if (epoch % checkpoint_freq) == 0 or stop:
                evaluation_stats = {}
                if not ignore_evaluation:
                    evaluation_stats = evaluate(self.autoencoder, self.datasets)
                    for key, val in evaluation_stats.items():
                        if not (key in self.autoencoder.history.keys()):
                            self.autoencoder.history[key] = []
                        self.autoencoder.history[key].append(val)
                    self.autoencoder.history['stats_epoch'].append(epoch)

                pjson({
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes
                })

                if save_checkpoints:
                    if save_dir is None or not os.path.exists(save_dir):
                        print(os.path.exists(save_dir))
                        print(not os.path.exists(save_dir))
                        raise ValueError(
                            "Please provide a valid directory path in the 'save_dir' argument."
                        )
                    fn = os.path.join(save_dir, f"{exp_id}_{epoch}.pt")
                    torch.save(
                        (self.autoencoder.state_dict(), self.autoencoder.hparams, self.autoencoder.history),
                        fn)
                    print(f"Model saved: {fn}")

                stop = stop or self.autoencoder.early_stopping(
                    np.mean(evaluation_stats["test"])) #or self.autoencoder.specific_threshold(
                    #evaluation_stats["test"][0], epoch, epoch_thr=361, score_thr=0.05)  # 0->gene
                if stop:
                    pjson({"early_stop": epoch})
                    if save_last:
                        if save_dir is None or not os.path.exists(save_dir):
                            print(os.path.exists(save_dir))
                            print(not os.path.exists(save_dir))
                            raise ValueError(
                                "Please provide a valid directory path in the 'save_dir' argument."
                            )
                        fn = os.path.join(save_dir, f"{exp_id}_last.pt")
                        torch.save(
                            (self.autoencoder.state_dict(), self.autoencoder.hparams, self.autoencoder.history),
                            fn)
                        print(f"Model saved: {fn}")

                    break

        results = self.autoencoder.history
        # results = pd.DataFrame.from_dict(results) # not same length!
        results["total_epochs"] = epoch
        results["exp_id"] = exp_id
        return results


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
def train(experiment=None):
    if experiment is None:
        experiment = ExperimentWrapper()
    return experiment.train()
