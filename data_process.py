import torch
import torchvision
from torchvision.utils import save_image
import torchvision.transforms as transforms
import os
import pandas as pd
from conf import conf

def save_img(loader, is_train, target_dir):
    """
    :param loader: Data loader
    :param is_train: Whether it is the training data
    :param target_dir: Directory to save the data
    :return:
    """
    # Construct the directory for saving data and the index file location
    if is_train:
        target_dir = os.path.join(target_dir, 'train')
        index_file = os.path.join(target_dir,'train.csv')
    else:
        target_dir = os.path.join(target_dir, 'test')
        index_file = os.path.join(target_dir, 'test.csv')

    os.makedirs(target_dir, exist_ok=True)

    num = 0
    # Save the filenames of the images
    index_fname  = []
    # Save the labels
    index_label = []

    for _, batch_data in enumerate(loader):
        data, label = batch_data
        for d,l in zip(data, label):

            # Construct the directory to save the image
            result_dir = os.path.join(target_dir, str(l.item()))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir,exist_ok=True)

            # Construct the filename to save the image
            file = os.path.join(result_dir, "{0}-{1}.png".format(l.item(), num))

            index_fname.append(file)
            index_label.append(l.item())

            # Save the image
            save_image(d.data, file)
            num += 1

    # Save the index
    # index = pd.DataFrame({
    #     conf["file_column"]:index_fname,
    #     conf["label_column"]:index_label
    # })
    # index.to_csv(index_file, index=False)
    index = torchvision.datasets.ImageFolder(target_dir)
    index_file = os.path.join(target_dir, f"{'train' if is_train else 'test'}.csv")
    with open(index_file, 'w') as f:
        f.write("file,label\n")
        for i in range(len(index)):
            path, label = index.imgs[i]
            f.write(f"{path},{label}\n")


def process_cifar100(data_dir, target_dir):
    """
    :param data_dir: Data directory
    :param target_dir: Target directory for processed data
    :return:
    """
    transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                            download=True, transform=transform)

    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                           download=False, transform=transform)

    train_loader =  torch.utils.data.DataLoader(trainset, batch_size=64,shuffle=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=64,shuffle=True)

    save_img(train_loader, is_train=True, target_dir=target_dir)
    save_img(test_loader,is_train=False,target_dir=target_dir)
    print("cifar100  process done !")


if __name__ == "__main__":
    process_cifar100('./data','./data/cifar10')



