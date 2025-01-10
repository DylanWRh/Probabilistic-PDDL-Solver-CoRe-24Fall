from engine_sgn import train_one_epoch, evaluate
from model.sgn import BlockStackingSGN
from data_utils.dataset import BlockStackingDataset, BlockStackingDemonstration

import os
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
from tqdm.auto import tqdm


def train_sgn(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    logger: SummaryWriter,
    lr: float = 1e-3,
    epochs: int = 100,
    val_interval: int = 10,
    save_interval: int = 10,
    save_dir: str = './checkpoints',
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    os.makedirs(f'{save_dir}/sgn-{current_time}', exist_ok=True)
    save_dir = f'{save_dir}/sgn-{current_time}'

    for epoch in range(1, epochs + 1):
        train_one_epoch(model, criterion, train_loader,
                        optimizer, device, epoch, logger)
        if epoch % val_interval == 0:
            evaluate(model, criterion, val_loader, device, epoch)
        if epoch % save_interval == 0 or epoch in [1, epochs]:
            torch.save(model.state_dict(), f'{save_dir}/model_{epoch:03d}.pth')
    return model


def fit_sgn(model, ds, device):
    lr = 1e-3
    n_epoch = 25
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    # p_bar = tqdm(range(n_epoch))
    for epoch in range(n_epoch):
        loss = train_one_epoch(model, criterion, ds, optimizer, device, epoch, None, False)
        # p_bar.set_description('current_loss: {}'.format(loss))
    return model


def main():

    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--experience_name', type=str,
                        default='sgn_{}'
                        .format(
                            datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
                        )
                        )

    # Data parameters
    parser.add_argument('--train_data', type=str,
                        default='./data/states/8blocks-3000_train.npz')
    parser.add_argument('--val_data', type=str,
                        default='./data/states/8blocks-500_val.npz')
    parser.add_argument('--batch_size', type=int, default=32)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    ds_train = BlockStackingDataset(args.train_data)
    # ds_val = BlockStackingDataset(args.val_data)
    # ds_train = BlockStackingDemonstration(args.train_data, 3000)
    ds_val = BlockStackingDataset(args.val_data)
    train_loader = torch.utils.data.DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_blocks = ds_train.n_blocks
    sgn_model = BlockStackingSGN(
        n_blocks, args.hidden_dim, args.depth).to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    logger = SummaryWriter('./logs/{}'.format(args.experience_name))
    train_sgn(
        sgn_model,
        criterion,
        train_loader,
        val_loader,
        device,
        logger,
        args.lr,
        epochs=args.epochs,
        val_interval=args.val_interval
    )


if __name__ == '__main__':
    main()
