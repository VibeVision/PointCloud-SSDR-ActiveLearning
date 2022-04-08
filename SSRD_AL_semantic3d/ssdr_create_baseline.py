import argparse

from RandLANet import Network, log_out
from sampler2 import *

if __name__ == '__main__':
    """create seed samples and model weights"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--dataset', type=str, default='semantic3d', choices=["S3DIS", "semantic3d", "SemanticKITTI"])
    parser.add_argument('--reg_strength', default=0.012, type=float,
                        help='regularization strength for the minimal partition')
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--lr_decay', default=0.92, type=float)

    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '2'



    dataset_name = FLAGS.dataset
    reg_strength = FLAGS.reg_strength
    round_num = 1
    epoch = FLAGS.epoch
    lr_decay = FLAGS.lr_decay

    sampler_args = []
    sampler_args.append("baseline")
    sampler_args.append(str(epoch))
    sampler_args.append(str(lr_decay))

    if dataset_name == "semantic3d":
        test_area_idx = 0
        input_ = "input_0.060"
        cfg = ConfigSemantic3D
        cfg.max_epoch = epoch
        cfg.lr_decays = {i: lr_decay for i in range(0, 500)}


    round_result_file = open(os.path.join("record_round", dataset_name + "_" + str(test_area_idx) + "_" + get_sampler