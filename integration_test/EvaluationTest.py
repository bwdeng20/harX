from utils.datasets import UCIHAR
from utils.preprocess import Si2Ai
from torch.utils.data import DataLoader
from model.MoDeX import JY15CNN3
from trainer.Coach import Coach


if __name__ == "__main__":
    test_dir=r'D:\DataSets\Activity_Recognition\SensorBased\UCI HAR Dataset\UCI HAR Dataset\test'
    testset = UCIHAR(dir=test_dir,transform=Si2Ai())
    test_loader = DataLoader(testset,421 ,True, num_workers=2)

    net1 = JY15CNN3(6)
    net1.load_weights(r"D:\WalkingTrajectoryEstimation\Projects\checkpoints\LeakyOne\JY15CNN3")
    coach = Coach(net1, cpdir=r"../checkpoints/LeakyOne/", logdir=r'../logs/LeakyJY15CNN/')
    coach.evaluate(test_loader)

    c_mat=coach.evaluations['confusionMatrix']
    print(c_mat)
    from utils.ProcessedVisual import visual_cm
    visual_cm(coach.evaluations['confusionMatrix'],classes=testset.activity.values())
    print(c_mat.sum())
