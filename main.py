# Problem: statistical heterogeneity
# In the federated scenario, we often have to deal with non-IID training data, due to diverse user behaviors.
# This leads to unstable and slow convergence and cause suboptimal or even detremimental model performance.

# The authors want to analyze the representations learned by different layers.
# The authors of the paper found out that:
# 1. there exists a greater bias in the classifier than other layers
# 2. the classification performance can be improved by post-calibrating the classifier after federated training

# The algorithm is called "Classifier Calibration with Virtual Representations" (CCVR).
# It adjusts the classifier using virtual representations sampled from an approximated gaussian mixture model.

import json,os

from conf import conf
import torch
import numpy as np
from fedavg.server import Server
from fedavg.client import Client

from fedavg.models import resnet20, CNN_Model, MLP, weights_init_normal
from utils import get_data
import copy
import wandb


if __name__ == '__main__':

    wandb.init(project="", entity="samaml", group="fedccvr", name="fedccvr")

    train_datasets, val_datasets, test_dataset = get_data()

    ### Initialize the aggregation weight of each node
    client_weight = {}
    if conf["is_init_avg"]:
        for key in train_datasets.keys():
            client_weight[key] = 1 / len(train_datasets)

    print("Aggregation weight initialized")

    ## Save nodes
    clients = {}
    # Save node models
    clients_models = {}

    if conf['model_name'] == "mlp":
        n_input = test_dataset.shape[1] - 1
        model = MLP(n_input, 512, conf["num_classes"])
    elif conf['model_name'] == 'cnn':
        ## Target model for training
        model = CNN_Model()
    elif conf['model_name'] == 'resnet20':
        model = resnet20(100)

    model.apply(weights_init_normal)

    if torch.cuda.is_available():
        model.cuda()


    server = Server(conf, model, test_dataset)

    print("Server initialized!")

    for key in train_datasets.keys():
        clients[key] = Client(conf, server.global_model, train_datasets[key], val_datasets[key])

    print("Clients initialized!")

    # Save the model
    if not os.path.isdir(conf["model_dir"]):
        os.mkdir(conf["model_dir"])
    max_acc = 0

    # Federated learning
    for e in range(conf["global_epochs"]):
        print("Epoch %d of %d" % (e,conf["global_epochs"]))

        for key in clients.keys():
            print('training client {}...'.format(key))
            model_k = clients[key].local_train(server.global_model)
            clients_models[key] = copy.deepcopy(model_k)

        # Federated aggregation
        server.model_aggregate(clients_models, client_weight)
        # Test the global model
        acc, loss = server.model_eval()
        print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))

        # Save test accuracy to wandb
        wandb.log({"Test Accuracy": acc})

        # # Save the best model
        # if acc >= max_acc:
        #     torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"], "model-epoch{}.pth".format(e)))
        #     max_acc = acc

    # Virtual Representation Generation ####################################################
    if conf['no-iid'] == 'fed_ccvr':
        # Post-processing using VR
        client_mean = {}
        client_cov = {}
        client_length = {}

        for key in clients.keys():
            print("Local feature mean and covariance calculated")
            # Client k computes local mean and covariance
            c_mean, c_cov, c_length = clients[key].cal_distributions(server.global_model)
            client_mean[key] = c_mean
            client_cov[key] = c_cov
            client_length[key] = c_length
        print("Completed calculation of local feature means and covariances")


        # Calculate the global mean and covariance
        g_mean, g_cov = server.cal_global_gd(client_mean, client_cov, client_length)
        print("Global mean and covariance calculated")

        # Generate a set of Gc virtual features with ground truth label c from the Gaussian distribution.
        retrain_vr = []
        label = []
        eval_vr = []
        for i in range(conf['num_classes']):
            mean = np.squeeze(np.array(g_mean[i]))
            # The optimal num_vr (M_c), number of virtual features, is 2000
            vr = np.random.multivariate_normal(mean, g_cov[i], conf["retrain"]["num_vr"]*2)
            retrain_vr.extend(vr.tolist()[:conf["retrain"]["num_vr"]])
            eval_vr.extend(vr.tolist()[conf["retrain"]["num_vr"]:])
            label.extend([i]*conf["retrain"]["num_vr"])

        print("Finished generating virtual features")



        ################### Classifier Re-Training ################################ using virtual representations
        
        # We take out the classifier g from the global model, initialize its parameter as phi^,
        # and re-train the parameter to phi^ for the objective (see the paper) where l is the cross-entropy loss.
        # We then obtain the final classification model "g_phi o f_teta^" consisting  of the pre-trained feature extractor
        # and the calibrated classifier


        # Get the layers of the model to be retrained
        retrain_model = ReTrainModel()
        if torch.cuda.is_available():
            retrain_model.cuda()
        reset_name = []
        for name, _ in retrain_model.state_dict().items():
            reset_name.append(name)

        # Initialize the retraining model
        for name, param in server.global_model.state_dict().items():
            if name in reset_name:
                retrain_model.state_dict()[name].copy_(param.clone())

        # Retrain using virtual features
        retrain_model = server.retrain_vr(retrain_vr, label, eval_vr, retrain_model)
        print("Finished retraining")

        # Update the global model using the retrained layers
        for name, param in retrain_model.state_dict().items():
            server.global_model.state_dict()[name].copy_(param.clone())

        acc, loss = server.model_eval()
        wandb.log({"Final Test Accuracy": acc})
        print("After retraining global_acc: %f, global_loss: %f\n" % (acc, loss))


    torch.save(server.global_model.state_dict(), os.path.join(conf["model_dir"],conf["model_file"]))

    print("Federated training completed, the model is saved in the {0} directory!".format(conf["model_dir"]))