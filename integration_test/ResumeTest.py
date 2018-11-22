from utils.datasets import UCIHAR
from utils.preprocess import Si2Ai
from torch.utils.data import DataLoader
from model.MoDeX import JY15CNN2
from trainer.Coach import Coach

if __name__ == "__main__":
    """
    Test the resuming training!!!
    """
    ucihar = UCIHAR(transform=Si2Ai(magnitude=True))
    trainloader = DataLoader(ucihar, batch_size=30, shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    sample = next(dataiter)
    print(sample['input'].shape)
    print(sample['label'])

    jy15cnn22 = JY15CNN2(6)
    # NOTICE:
    # remember to use the same type optimizer!!! otherwise state_dict of optimizer can't be restored correctly

    # NOTICE:
    # to resuming monitoring, please make sure that 'logdir' is the same as when you new a Coach
    # class to resume training!
    Coach = Coach(jy15cnn22, epochs=100, logdir=r'../logs/test_resuming_monitoring-')
    Coach.resume(trainloader, cp_file=r"../checkpoints/")
