import argparse

import numpy as np
import torch
from matplotlib.ticker import PercentFormatter, MultipleLocator
from torch.utils.data import DataLoader
from typeguard import typechecked

from datasets.datasets import DatasetType, load_dataset
from experiment import Experiment
from boundflow.layers import LinearLayer, ReLU
from boundflow.model_functions import train_model, evaluate_model
from boundflow.network import Model
from boundflow.objective import CrossEntropyLoss, HingeLoss, BinaryCrossEntropyLoss
from boundflow.perturbation import perturb_entire_dataset, perturb_first_datapoint
from util import require
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def main():
    args = parse_args()
    experiment = Experiment(args)
    logger = experiment.logger
    with torch.no_grad():

        data = load_dataset(dataset=args.dataset, pretrain_size=args.pretrain_size, train_size=None, test_size=None)

        pre_train_loader = DataLoader(
            dataset=data.pretrain_set,
            batch_size=min(len(data.pretrain_set), args.batch_size),
            shuffle=True
        )

        if args.perturbation == "all":
            perturbed_train_set = perturb_entire_dataset(data.train_set, args.epsilon, data_range=data.data_range)
        elif args.perturbation == "first":
            perturbed_train_set = perturb_first_datapoint(data.train_set, args.epsilon, data_range=data.data_range)

        if args.perturb_inference:
            val_set = perturb_entire_dataset(data.val_set, args.epsilon, data_range=data.data_range)
            test_set = perturb_entire_dataset(data.test_set, args.epsilon, data_range=data.data_range)
        else:
            val_set = data.val_set
            test_set = data.test_set

        train_loader = DataLoader(dataset=perturbed_train_set, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

        require(args.loss == "ce" or data.label_dim == 2,
                "Multi-class classification is only supported for the CE loss.")
        out_features = data.label_dim if args.loss == 'ce' else 1

        num_seeds = 10

        seeds = np.random.randint(low=0, high=1_000_000, size=num_seeds)

        results = []
        points = []
        for run in range(num_seeds):
            experiment.set_random_seed(seeds[run].item())
            if args.layers == 2:
                model = Model([
                    LinearLayer(in_features=np.prod(data.feature_dim).item(), out_features=args.hidden_nodes),
                    ReLU(),
                    LinearLayer(in_features=args.hidden_nodes, out_features=out_features),
                ])
            elif args.layers == 3:
                model = Model([
                    LinearLayer(in_features=np.prod(data.feature_dim).item(), out_features=args.hidden_nodes),
                    ReLU(),
                    LinearLayer(in_features=args.hidden_nodes, out_features=args.hidden_nodes),
                    ReLU(),
                    LinearLayer(in_features=args.hidden_nodes, out_features=out_features),
                ])
            else:
                raise ValueError(f"Invalid number of layers {args.layers}.")

            logger.info(model)

            if args.loss == "ce":
                loss_function = CrossEntropyLoss()

            elif args.loss == "hinge":
                loss_function = HingeLoss()
            elif args.loss == "bce":
                loss_function = BinaryCrossEntropyLoss()
            else:
                raise ValueError(f"Invalid loss function {args.loss}.")

            logger.info("Pre-training...")

            models = [(model.clone(), evaluate_model(model, val_loader)[2])]
            next_accuracy = 0.5

            for epoch in range(args.pretrain_epochs):
                train_model(
                    model=model,
                    epochs=1,
                    train_loader=pre_train_loader,
                    val_loader=val_loader,
                    learning_rate=args.pretrain_learning_rate,
                    loss_function=loss_function,
                    quantize_inputs=args.quantize_inputs
                )
                accuracy = evaluate_model(model, val_loader)[2]
                if accuracy >= next_accuracy:
                    models.append((model.clone(), evaluate_model(model, test_loader)[2]))
                    next_accuracy += 0.05

            models.append((model.clone(), evaluate_model(model, val_loader)[2]))
            logger.info([f"{accuracy:.1%}" for _, accuracy in models])

            final_models = []

            for model, accuracy in models:
                logger.info(f"Training model with accuracy {accuracy:.1%}...")
                best_model = model.clone()
                best_certified_accuracy = accuracy
                for epoch in range(args.epochs):
                    train_model(
                        model=model,
                        epochs=1,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        learning_rate=args.learning_rate,
                        loss_function=loss_function,
                        quantize_inputs=args.quantize_inputs
                    )
                    certified_accuracy = evaluate_model(model, val_loader)[2]
                    if certified_accuracy > best_certified_accuracy:
                        best_model = model.clone()
                        best_certified_accuracy = certified_accuracy
                    elif certified_accuracy < accuracy:
                        break
                logger.info(f"Final accuracy: {best_certified_accuracy:.1%}")
                final_models.append((best_model, accuracy, evaluate_model(best_model, test_loader)[2]))

            logger.info(f"After Pre-Training: {[f'{accuracy:.1%}' for _, accuracy, _ in final_models]}")
            logger.info(f"After Training: {[f'{certified_accuracy:.1%}' for _, _, certified_accuracy in final_models]}")
            results.append([(a, c) for _, a, c in final_models])
            points += [(a, c) for _, a, c in final_models]

        points = np.array(points)
        convex_hull = ConvexHull(points)

        # set plt figure size
        plt.figure(figsize=(4.854, 3))

        plt.fill(points[convex_hull.vertices, 0], points[convex_hull.vertices, 1], alpha=0.5, color="C0")
        plt.plot(points[:, 0], points[:, 1], 'o', color="C1", markersize=5)

        plt.plot([0.5, 1.0], [0.5, 1.0], linestyle='--', color="black")
        plt.xlabel("Pretrain-Accuracy")
        plt.ylabel("Certified-Accuracy")
        plt.xlim(0.5, 1.0)
        plt.ylim(0.5, 1.0)

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))

        plt.tight_layout()
        plt.savefig(experiment.directory / "results.pdf")
        plt.show()


@typechecked
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Certified Training")
    parser.add_argument("--batch-size", type=int, default=100, help="The batch size for training (default=100)")
    parser.add_argument("--learning-rate", type=float, default=.7,
                        help="The learning rate for training (default=0.7)")
    parser.add_argument("--epochs", type=int, default=10, help="The number of epochs to train for (default=10)")
    parser.add_argument("--pretrain-epochs", type=int, default=500,
                        help="The number of epochs for pre-training (default=500)")
    parser.add_argument("--pretrain-learning-rate", type=float, default=10.0,
                        help="The learning rate for pre-training (default=10.0)")
    parser.add_argument("--pretrain-size", type=int, default=10, help="The number of data samples for pretraining")
    parser.add_argument("--epsilon", type=float, default=0.01, help="The perturbation radius (default=0.01)")
    parser.add_argument("--hidden-nodes", type=int, default=20,
                        help="The number of neurons in the hidden layer (default=20)")
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.MOONS,
                        help="The dataset to use")
    parser.add_argument("--perturbation", choices=["first", "all"], default="all", help="Perturbation strategy")
    parser.add_argument("--quantize-inputs", action="store_true", help="Convert inputs to black and white")
    parser.add_argument("--experiment", type=str, required=True, help="The experiment name")
    parser.add_argument("--seed", type=int, default=10123310, help="The random seed")
    parser.add_argument("--loss", choices=["ce", "hinge", "bce", "mse"], default="ce", help="The loss function")
    parser.add_argument("--freeze-first-layer", action="store_true", help="Freeze the first layer after pre-training")
    parser.add_argument("--layers", type=int, choices=[2, 3], required=True, help="The number of layers to use")
    parser.add_argument("--perturb-inference", action="store_true", help="Perturb the inference set")
    return parser.parse_args()


if __name__ == "__main__":
    main()
