from utils.datasets import UCIHAR
from utils.preprocess import Si2Ai, Ucihar4Convlstm
from torch.utils.data import DataLoader
from model.MoDeX import JY15CNN3, ConvLstm
from trainer.Coach import Coach

if __name__ == "__main__":
    test_dir = r'D:\DataSets\Activity_Recognition\SensorBased\UCI HAR Dataset\UCI HAR Dataset\test'
    testset = UCIHAR(dir=test_dir, transform=Si2Ai())
    test_loader = DataLoader(testset, 421, True, num_workers=2)

    net1 = JY15CNN3(6)
    net1.load_weights(r"D:\WalkingTrajectoryEstimation\harX\checkpoints\LeakyOne\JY15CNN3")
    coach = Coach(net1)
    coach.evaluate(test_loader)

    c_mat = coach.evaluations['confusionMatrix']
    print(c_mat)
    from utils.processedvisual import visual_cm

    visual_cm(coach.evaluations['confusionMatrix'], classes=testset.activity.values())
    print(c_mat.sum())
    # ----------------------------------------------------------------------------------------------------

    testset = UCIHAR(dir=test_dir, transform=Ucihar4Convlstm())
    testloader = DataLoader(testset, batch_size=30)
    net2 = ConvLstm(6, 9)
    net2.load_weights(
        r"D:\WalkingTrajectoryEstimation\harX\checkpoints\ConvLstm1\ConvLstm-trained")
    coach = Coach(net2)
    coach.evaluate(testloader)
    c_mat = coach.evaluations['confusionMatrix']
    print(c_mat)

    visual_cm(coach.evaluations['confusionMatrix'], classes=testset.activity.values())
    print(c_mat.sum())
