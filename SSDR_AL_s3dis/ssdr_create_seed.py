import argparse

from RandLANet import Network, log_out
from sampler2 import *

if __name__ == '__main__':
    """create seed samples and model weights"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--dataset', type=str, default='S3DIS', choices=["S3DIS", "semantic3d", "SemanticKITTI"])
    parser.add_argument('--seed_percent', type=float, default=0.01, help='seed percent')
    parser.add_argument('--reg_strength', default=0.012, type=float,
                        help='regularization strength for the minimal partition')

    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '2'

    sampler_args = []
    sampler_args.append("seed")

    dataset_name = FLAGS.dataset
    seed_percent = FLAGS.seed_percent
    reg_strength = FLAGS.reg_strength
    round_num = 1

    if dataset_name == "S3DIS":
        test_area_idx = 5
        input_  = "input_0.040"
        cfg = ConfigS3DIS

    round_result_file = open(os.path.join("record_round", dataset_name + "_" + str(test_area_idx) + "_" + get_sampler_args_str(sampler_args) + "_" + str(reg_streng