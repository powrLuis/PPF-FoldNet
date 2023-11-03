"""Module for loading data"""
import time
import torch
from dataset import SunDataset


def get_dataloader(root, split, batch_size=1, num_patches=32, num_points_per_patch=1024,
                   num_workers=4, shuffle=True,
                   on_the_fly=True):
    """Get dataloader for training and testing"""
    dataset = SunDataset(
        root=root,
        split=split,
        num_patches=num_patches,
        num_points_per_patch=num_points_per_patch,
        on_the_fly=on_the_fly
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dataloader


if __name__ == '__main__':
    DATASET = 'sun3d'
    DATAROOT = "/data/3DMatch/whole"
    trainloader = get_dataloader(DATAROOT, split='test', batch_size=32)
    start_time = time.time()
    print(f"Totally {len(trainloader)} iter.")
    for itera, (patches, ids) in enumerate(trainloader):
        if itera % 100 == 0:
            print(f"Iter {itera}: {time.time() - start_time} s")
    print(f"On the fly: {time.time() - start_time}")
