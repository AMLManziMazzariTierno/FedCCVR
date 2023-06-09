import argparse
import torch

from conf import conf
from fedavg.server import Server
from fedavg.client import Client
from fedavg.models import CNN_Model, resnet20
from utils import get_data, FedTSNE


if __name__ == '__main__':
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_before_calibration', default='./save_model/model-epoch9.pth', type=str, help='path to model before calibration')
    parser.add_argument('--model_after_calibration', default='./save_model/model.pth', type=str, help='path to model after calibration')
    parser.add_argument('--random_state', default=1, type=int, help='random state for tsne')
    parser.add_argument('--save_path', default='./visualize/tsne.png', type=str, help='path to save tsne result')
    args = parser.parse_args()

    train_datasets, val_datasets, test_dataset = get_data()

    # Define the model
    model = resnet20(100)
    if torch.cuda.is_available():
        model.cuda()

    # Initialize the server
    server = Server(conf, model, test_dataset)

    print('Start TSNE...')
    server.global_model.load_state_dict(torch.load(args.model_before_calibration))
    # Get feature vectors, true labels, and labels before calibration for the test dataset
    tsne_features, tsne_true_labels, tsne_before_labels = server.get_feature_label()
    # Get labels after calibration
    server.global_model.load_state_dict(torch.load(args.model_after_calibration))
    _, _, tsne_after_labels = server.get_feature_label()
    # Perform t-SNE visualization
    fed_tsne = FedTSNE(tsne_features.detach().cpu().numpy(), random_state=args.random_state)
    fed_tsne.visualize_3(tsne_true_labels.detach().cpu().numpy(),
                         tsne_before_labels.detach().cpu().numpy(),
                         tsne_after_labels.detach().cpu().numpy(),
                        figsize=(15, 3), save_path=args.save_path)
    print('TSNE done.')
