import torch


DEV_IDS = [0, 7]
DEV = torch.device('cuda:{}'.format(DEV_IDS[0]) if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    print(DEV)
