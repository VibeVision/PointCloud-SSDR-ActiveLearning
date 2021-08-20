import argparse

from RandLANet import Network, log_out
from sampler2 import *

if __name__ == '__main__':
    """create seed samples and model weights"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--dataset', type=str, default='S3DIS', choices=["S3DIS", "semantic3d", "SemanticKITTI"])
    parser.add_argument('--reg_strength', default=0.008, type=float,
                        help='regularization strength for the minimal partition')

    FLAGS = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '2'

    sampler_args = []
    sampler_args.append("baseline")

    dataset_name = FLAGS.dataset
    reg_strength = FLAGS.reg_strength
    round_num = 1

    if dataset_name == "S3DIS":
        test_area_idx = 5
        input_  = "input_0.040"
        cfg = ConfigS3DIS

    round_result_file = open(os.path.join("record_round", dataset_name + "_" + str(test_area_idx) + "_" + get_sampler_args_str(sampler_args) + "_" + str(reg_strength) + '.txt'), 'a')

    with open(os.path.join("data", dataset_name, str(reg_strength), "superpoint/total.pkl"), "rb") as f:
        total_obj = pickle.load(f)
    total_sp_num = total_obj["sp_num"]

    print("total_sp_num", total_sp_num)
    Sampler = SeedSampler("data/" +dataset_name + "/" + input_, "data/" + dataset_name + "/" + str(reg_strength), total_sp_num, sampler_args)

    w = {"sp_num": 0, "p_num": 0, "p_num_list": [], "sp_id_list": [], "sub_num": 0, "sub_p_num": 0}
    sp_batch_size = total_sp_num
    Sampler.sampling(None, sp_batch_size, last_round=round_num - 1, w=w)
    label