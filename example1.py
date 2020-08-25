from torch.utils.data import DataLoader
from utils.datasets import UCIHAR
from utils.preprocess import Si2Ai
from model.MoDeX import JY15CNN3
from trainer.Coach import Coach
from utils.processedvisual import visual_cm

if __name__ == "__main__":
    # replace `train_dir` and `test_dir` to the dataset path in your machine.
    train_dir = r'D:/DataSets/Activity_Recognition/SensorBased/UCI HAR Dataset/UCI HAR Dataset/train'
    test_dir = r'D:/DataSets/Activity_Recognition/SensorBased/UCI HAR Dataset/UCI HAR Dataset/test'

    trainset = UCIHAR(dir=train_dir, transform=Si2Ai())
    train_loader = DataLoader(trainset, 32, True)

    net1 = JY15CNN3(6)
    cp_dir = r"./checkpoints/example1/"
    log_dir = r"./logs/example1/"
    coach = Coach(net1, optimizer="Adam", epochs=5, cpdir=cp_dir, logdir=log_dir)
    coach.fit(train_loader)  # coach will save a well-trained model

    testset = UCIHAR(dir=test_dir, transform=Si2Ai())
    test_loader = DataLoader(testset, 421, True)
    coach.evaluate(test_loader)
    c_mat = coach.evaluations['confusionMatrix']
    print(c_mat)

    visual_cm(coach.evaluations['confusionMatrix'], classes=testset.activity.values())
    print(c_mat.sum())
