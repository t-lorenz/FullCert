import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from typeguard import typechecked

from boundflow.layers import LinearLayer, ReLU
from boundflow.model_functions import evaluate_model
from boundflow.model_functions import train_model
from boundflow.network import Model
from boundflow.objective import CrossEntropyLoss, BinaryCrossEntropyLoss
from boundflow.perturbation import perturb_entire_dataset
from datasets.datasets import DatasetType, load_dataset
from experiment import Experiment
from util import require


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

        perturbed_train_set = perturb_entire_dataset(data.train_set, args.epsilon, data_range=data.data_range)

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

        model = Model([
            LinearLayer(in_features=np.prod(data.feature_dim).item(), out_features=args.hidden_nodes),
            ReLU(),
            LinearLayer(in_features=args.hidden_nodes, out_features=out_features),
        ])

        logger.info(model)

        if args.loss == "ce":
            loss_function = CrossEntropyLoss()
        elif args.loss == "bce":
            loss_function = BinaryCrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function {args.loss}.")

        if args.pretrain_epochs > 0:
            logger.info("Pre-training...")

            train_model(
                model=model,
                epochs=args.pretrain_epochs,
                train_loader=pre_train_loader,
                val_loader=val_loader,
                learning_rate=args.pretrain_learning_rate,
                loss_function=loss_function
            )
            experiment.save_model(model, "pretrained_model")

        if args.freeze_first_layer:
            model.layers[0].freeze(True)
            logger.info("Freezing first layer")

        logger.info("Training...")

        max_certified_accuracy, best_model = train_model(
            model=model,
            epochs=args.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.learning_rate,
            loss_function=loss_function,
            early_stopping=True
        )
        experiment.save_model(best_model, "trained_model")

        _, _, certified_test_accuracy = evaluate_model(best_model, test_loader)

        logger.info(f"Validation certified accuracy: {max_certified_accuracy:.1%}")
        logger.info(f"Test certified accuracy: {certified_test_accuracy:.1%}")
        logger.info("Done.")


@typechecked
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FullCert")
    parser.add_argument("--batch-size", type=int, default=100, help="The batch size for training (default=100)")
    parser.add_argument("--learning-rate", type=float, default=.5,
                        help="The learning rate for training (default=0.5)")
    parser.add_argument("--epochs", type=int, default=20, help="The number of epochs to train for (default=20)")
    parser.add_argument("--pretrain-epochs", type=int, default=0,
                        help="The number of epochs for pre-training (default=0)")
    parser.add_argument("--pretrain-learning-rate", type=float, default=10.0,
                        help="The learning rate for pre-training (default=10.0)")
    parser.add_argument("--pretrain-size", type=int, default=10, help="The number of data samples for pretraining")
    parser.add_argument("--epsilon", type=float, default=0.01, help="The perturbation radius (default=0.01)")
    parser.add_argument("--hidden-nodes", type=int, default=20,
                        help="The number of neurons in the hidden layer (default=20)")
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.MOONS,
                        help="The dataset to use")
    parser.add_argument("--experiment", type=str, required=True, help="The experiment name")
    parser.add_argument("--seed", type=int, default=10123310, help="The random seed")
    parser.add_argument("--loss", choices=["ce", "bce"], default="bce", help="The loss function")
    parser.add_argument("--freeze-first-layer", action="store_true", help="Freeze the first layer after pre-training")
    parser.add_argument("--perturb-inference", action="store_true", help="Perturb the inference set")
    parser.add_argument("--train-size", type=int, default=None, help="Number of training images")
    parser.add_argument("--test-size", type=int, default=None, help="Number of test images")
    return parser.parse_args()


if __name__ == "__main__":
    main()
