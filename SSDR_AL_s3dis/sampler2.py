
import shutil
import shutil
import time

import numpy as np

from s3dis_dataset import *
from base_op import *
from gcn import *
from fps_gcn_cpu import GCN_FPS_sampling

def compute_region_uncertainty(pixel_uncertainty, pixel_class, class_num, sampler_args):
    if "mean" in sampler_args:
        return np.mean(pixel_uncertainty)
    elif "sum_weight" in sampler_args:
        # pixel_weights = np.exp(weights_percentage(list_class=prob_class[point_ids], class_num=class_num) - 1)
        pixel_weights = weights_percentage(list_class=pixel_class, class_num=class_num)
        return np.sum(np.multiply(pixel_weights, pixel_uncertainty))
    elif "WetSU" in sampler_args:
        d_label, _ = _dominant_label(pixel_class)
        equal_id = np.where(pixel_class == d_label, 1.0, 0.0)
        equal_num = np.sum(equal_id)
        pixel_num = len(pixel_uncertainty)

        region_uncertainty = np.sum(np.multiply(pixel_uncertainty, equal_id)) - np.sum(np.multiply(pixel_uncertainty, 1 - equal_id))
        return region_uncertainty

def compute_point_uncertainty(prob_logits, sampler_args):
    if "lc" in sampler_args:
        """
        least confidence
        An analysis of active learning strategies for sequence labeling tasks
        """
        prob_max = np.max(prob_logits, axis=-1)  # [batch_size * point_num]
        point_uncertainty = 1.0 - prob_max  # [batch_size * point_num]
    elif "entropy" in sampler_args:
        """
        entropy
        An analysis of active learning strategies for sequence labeling tasks
        """
        point_uncertainty = compute_entropy(prob_logits)  # [batch_size * point_num]
    elif "sb" in sampler_args:
        """second best / best"""
        prob_sorted = np.sort(prob_logits, axis=-1)
        point_uncertainty = prob_sorted[:, -2] / prob_sorted[:, -1]  # [batch_size * point_num]

    return point_uncertainty

def farthest_superpoint_sample(superpoint_list, superpoint_centroid_list, sample_number, trigger_idx):

    sp_num = len(superpoint_list)
    align_superpoint_list = []
    tree_list = []
    for i in range(sp_num):
        # 中心对齐
        align_superpoint = superpoint_list[i] - superpoint_centroid_list[i]
        align_superpoint_list.append(align_superpoint)
        tree_list.append(KDTree(align_superpoint))


    centroids = np.zeros([sample_number], dtype=np.int32)
    centroids[0] = trigger_idx

    distance = np.ones([sp_num]) * 1e10

    for i in range(sample_number - 1):

        current_superpoint_center = superpoint_centroid_list[centroids[i]]
        euclidean_dist = np.sum((superpoint_centroid_list - current_superpoint_center) ** 2, axis=-1)

        cd_dist = chamfer_distance(align_superpoint_list, tree_list, centroids[i])


        dist = np.add(euclidean_dist, cd_dist)

        mask = dist < distance
        distance[mask] = dist[mask]

        centroids[i + 1] = np.argmax(distance)
    return centroids

def weights_softmax(list_class, class_num):
    class_distribution = np.zeros([class_num], dtype=np.float128)
    for cls in list_class:
        class_distribution[cls] = class_distribution[cls] + 1
    class_distribution = np.true_divide(np.exp(class_distribution), np.sum(np.exp(class_distribution)))  # softmax
    weights = []
    for cls in list_class:
        weights.append(class_distribution[cls])
    return np.asarray(weights)

def weights_percentage(list_class, class_num):
    class_distribution = np.zeros([class_num])
    for c in list_class:
        class_distribution[c] = class_distribution[c] + 1
    class_distribution = class_distribution / len(list_class)
    weights = []
    for c in list_class:
        weights.append(class_distribution[c])
    return np.asarray(weights)

def _dominant_label(ary):
    ssdr = np.zeros([np.max(ary) + 1], dtype=np.int32)
    for a in ary:
        ssdr[a] = ssdr[a] + 1
    return np.argmax(ssdr), np.amax(ssdr) / len(ary)

def _dominant_2(ary):
    ary = np.array(ary)
    ssdr = np.zeros([np.max(ary) + 1], dtype=np.int32)
    for a in ary:
        ssdr[a] = ssdr[a] + 1
    label = np.argmax(ssdr)
    ids = np.where(ary == label)
    return label, ids

def _get_sub_region_from_superpoint(prob_class, point_inds):
    ssdr = [[] for _ in range(np.max(prob_class[point_inds]) + 1)]
    for pid in point_inds:
        cls = prob_class[pid]
        ssdr[cls].append(pid)
    return ssdr

def oracle_labeling(superpoint_inds, components, input_gt, pseudo_gt, cloud_name, w, sampler_args, prob_class, threshold, budget, min_size, total_obj):
    used_superpoint_inds = []

    if "dominant" in sampler_args:
        for superpoint_idx in superpoint_inds:
            if budget["click"] > 0:
                point_inds = components[superpoint_idx]
                if len(point_inds) >= min_size:
                    used_superpoint_inds.append(superpoint_idx)
                    budget["click"] = budget["click"] - 1  # click + 1

                    do_label, _ = _dominant_label(input_gt[point_inds])
                    pseudo_gt[0][point_inds] = 1.0
                    pseudo_gt[1][point_inds] = do_label * 1.0
                    total_obj["selected_class_list"].append(do_label)

                    w["sp_num"] = w["sp_num"] + 1
                    w["p_num"] = w["p_num"] + len(point_inds)

            else:
                break

    elif "NAIL" in sampler_args:
        for superpoint_idx in superpoint_inds:
            if budget["click"] > 0:
                point_inds = components[superpoint_idx]
                if len(point_inds) >= min_size:
                    ignore = True

                    used_superpoint_inds.append(superpoint_idx)
                    budget["click"] = budget["click"] - 1  # click + 1

                    do_label, do_rate = _dominant_label(input_gt[point_inds])
                    if do_rate >= threshold:
                        pseudo_gt[0][point_inds] = 1.0
                        pseudo_gt[1][point_inds] = do_label * 1.0
                        total_obj["selected_class_list"].append(do_label)

                        w["sp_num"] = w["sp_num"] + 1
                        w["p_num"] = w["p_num"] + len(point_inds)
                        ignore = False
                    else:
                        sub_region_list = _get_sub_region_from_superpoint(prob_class=prob_class, point_inds=point_inds)
                        for sub_id in range(len(sub_region_list)):
                            sub_region_pids = sub_region_list[sub_id]
                            if len(sub_region_pids) > min_size:
                                sub_do_label, sub_do_rate = _dominant_label(input_gt[sub_region_pids])
                                if sub_do_rate >= threshold:
                                    budget["click"] = budget["click"] - 1  # click + 1
                                    pseudo_gt[0][sub_region_pids] = 1.0
                                    pseudo_gt[1][sub_region_pids] = sub_do_label * 1.0
                                    total_obj["selected_class_list"].append(sub_do_label)

                                    w["sub_num"] = w["sub_num"] + 1
                                    w["sub_p_num"] = w["sub_p_num"] + len(sub_region_pids)
                                    ignore = False

                        if not ignore:
                            w["split_sp_num"] = w["split_sp_num"] + 1

                    if ignore:
                        w["ignore_sp_num"] = w["ignore_sp_num"] + 1

            else:
                break
    else:
        print("not find oracle_mode==" + get_sampler_args_str(sampler_args))
        1 / 0
    return pseudo_gt, used_superpoint_inds

def _help(input_path, data_path, total_obj, current_path, cloud_name, superpoint_inds, w, sampler_args, prob_class, threshold, budget, min_size):

    with open(os.path.join(data_path, "superpoint", cloud_name + ".superpoint"), "rb") as f:
        sp = pickle.load(f)
    components = sp["components"]
    # pseudo gt
    pseudo_gt_path = os.path.join(current_path, cloud_name + ".gt")
    with open(pseudo_gt_path, "rb") as f:
        pseudo_gt = pickle.load(f)
        pseudo_gt = np.asarray(pseudo_gt)
    # input gt
    data = read_ply(os.path.join(input_path, cloud_name + ".ply"))
    input_gt = np.asarray(data['class'])

    pseudo_gt, used_superpoint_inds = oracle_labeling(superpoint_inds=superpoint_inds, components=components, input_gt=input_gt, pseudo_gt=pseudo_gt,
                    cloud_name=cloud_name, w=w, sampler_args=sampler_args, prob_class=prob_class, threshold=threshold, budget=budget, min_size=min_size, total_obj=total_obj)

    with open(os.path.join(pseudo_gt_path), "wb") as f:
        pickle.dump(pseudo_gt, f)

    total_obj["unlabeled"][cloud_name] = list(set(total_obj["unlabeled"][cloud_name]) - set(used_superpoint_inds))
    if len(total_obj["unlabeled"][cloud_name]) == 0:
        del total_obj["unlabeled"][cloud_name]

def _help_seed(input_path, data_path, total_obj, current_path, cloud_name, superpoint_inds, w):

    with open(os.path.join(data_path, "superpoint", cloud_name + ".superpoint"), "rb") as f:
        sp = pickle.load(f)
    components = sp["components"]
    # pseudo gt
    pseudo_gt_path = os.path.join(current_path, cloud_name + ".gt")
    with open(pseudo_gt_path, "rb") as f:
        pseudo_gt = pickle.load(f)
        pseudo_gt = np.asarray(pseudo_gt)
    # input gt
    data = read_ply(os.path.join(input_path, cloud_name + ".ply"))
    input_gt = np.asarray(data['class'])

    for superpoint_idx in superpoint_inds:
        point_inds = components[superpoint_idx]
        pseudo_gt[0][point_inds] = 1.0
        pseudo_gt[1][point_inds] = input_gt[point_inds]  # precise label
        w["sp_num"] = w["sp_num"] + 1
        w["p_num"] = w["p_num"] + len(point_inds)

    with open(os.path.join(pseudo_gt_path), "wb") as f:
        pickle.dump(pseudo_gt, f)


    total_obj["unlabeled"][cloud_name] = list(set(total_obj["unlabeled"][cloud_name]) - set(superpoint_inds))
    if len(total_obj["unlabeled"][cloud_name]) == 0:
        del total_obj["unlabeled"][cloud_name]

def compute_entropy(x):

    class_num = np.array(x).shape[-1]
    x = np.reshape(x, [-1, class_num])
    k = np.log2(x)
    where_are_inf = np.isinf(k)
    k[where_are_inf] = 0
    entropy = np.sum(np.multiply(x, k), axis=-1)
    return -1 * entropy

def add_classbal(class_num, region_class, region_uncertainty):
    weights = weights_percentage(list_class=region_class, class_num=class_num)
    class_bal_region_uncertainty = np.multiply(region_uncertainty, np.exp(-np.asarray(weights)))
    return class_bal_region_uncertainty

def add_clsbal(class_num, region_class, region_uncertainty, total_obj):
    list_class = list(region_class) + list(total_obj["selected_class_list"])
    weights = weights_percentage(list_class=list_class, class_num=class_num)
    class_bal_region_uncertainty = np.multiply(region_uncertainty, np.exp(-np.asarray(weights[0:len(region_uncertainty)])))
    return class_bal_region_uncertainty

def get_labeled_selection_cloudname_spidx_pointidx(input_path, data_path, labeled_region_reference_dict, class_num, round_num):
    """
    labeled_region_reference_dict: {cloud_name: [sp_idx]}
    """

    dominant_label_list = []
    labeled_region_reference = []  # ele {cloud_name:, sp_idx:, dominant_point_ids:}

    for cloud_name in labeled_region_reference_dict:
        with open(join(data_path, "superpoint", cloud_name + ".superpoint"), "rb") as f:
            sp = pickle.load(f)
        components = sp["components"]

        sub_ply_file = join(input_path, '{:s}.ply'.format(cloud_name))
        data = read_ply(sub_ply_file)
        cloud_point_label = data['class']  # shape=[point_number]

        sp_idx_list = labeled_region_reference_dict[cloud_name]
        for sp_idx in sp_idx_list:
            point_ids = components[sp_idx]
            dominant_label, idns = _dominant_2(cloud_point_label[point_ids])
            dominant_point_ids = np.array(point_ids)[idns]
            dominant_label_list.append(dominant_label)
            labeled_region_reference.append({"cloud_name": cloud_name, "sp_idx": sp_idx, "dominant_point_ids": dominant_point_ids})

    labeled_region_reference = np.array(labeled_region_reference, dtype='object')

    weights = weights_percentage(list_class=dominant_label_list, class_num=class_num)
    probability = weights / np.sum(weights)
    labeled_all_num = len(probability)
    batch = (round_num - 1) * 1000
    if batch > labeled_all_num:
        batch = labeled_all_num
    selection = np.random.choice(a=labeled_all_num, size=batch, replace=False, p=probability)

    labeled_select_region = {}
    for item in labeled_region_reference[selection]:
        cloud_name = item["cloud_name"]
        sp_idx = item["sp_idx"]
        dominant_point_ids = item["dominant_point_ids"]
        if cloud_name not in labeled_select_region:
            labeled_select_region[cloud_name] = {}
        labeled_select_region[cloud_name][sp_idx] = dominant_point_ids
    return labeled_select_region, batch  # {cloud_name: {sp_idx: dominant_point_ids}}

def compute_features(dataset_name, test_area_idx, sampler_args, round_num, reg_strength, model,
                     labeled_select_regions, unlabeled_candidate_regions):
    labeled_select_features = []
    labeled_select_ref = []
    unlabeled_candidate_features = []
    unlabeled_candidate_ref = []
    if dataset_name == "S3DIS":
        sample_data = S3DIS_Dataset(test_area_idx=test_area_idx, sampler_args=sampler_args,
                                    round_num=round_num, mode="sampling", reg_strength=reg_strength)

    sample_loader = DataLoader(sample_data, batch_size=1, shuffle=True, num_workers=6)

    for i, dat in enumerate(sample_loader):
        last_second_features, cloud_inds, point_idx = model.sess.run([model.last_second_features, model.input_cloud_inds, model.input_input_inds], feed_dict=model.get_feed_dict(dat, False))
        last_second_features = last_second_features[np.argsort(point_idx[0])]

        cloud_name = sample_data.input_cloud_names[cloud_inds[0]]
        if cloud_name in labeled_select_regions:
            for sp_idx in labeled_select_regions[cloud_name]:
                dominant_point_ids = labeled_select_regions[cloud_name][sp_idx]
                labeled_select_features.append(np.mean(last_second_features[dominant_point_ids], axis=0))
                labeled_select_ref.append({"cloud_name": cloud_name, "sp_idx": sp_idx})

        if cloud_name in unlabeled_candidate_regions:
            for sp_idx in unlabeled_candidate_regions[cloud_name]:
                dominant_point_ids = unlabeled_candidate_regions[cloud_name][sp_idx]
                unlabeled_candidate_features.append(np.mean(last_second_features[dominant_point_ids], axis=0))
                unlabeled_candidate_ref.append({"cloud_name": cloud_name, "sp_idx": sp_idx})

    return labeled_select_features, labeled_select_ref, unlabeled_candidate_features, unlabeled_candidate_ref

class SeedSampler:
    """
    use precise labeling
    """
    def __init__(self, input_path, data_path, total_num, sampler_args):
        self.input_path = input_path
        self.data_path = data_path
        self.total_num = total_num
        self.sampler_args = sampler_args

    def _iteration(self, current_path, total_obj, number, w):
        remain_number = 0

        rand_inds = np.random.choice(range(self.total_num), int(number), replace=False)
        length = len(total_obj["unlabeled"])
        cloud_name_list = []
        for cloud_name in total_obj["unlabeled"]:
            cloud_name_list.append(cloud_name)

        each_file_number = np.zeros([length], dtype=np.int32)
        for ind in rand_inds:
            d = ind % length
            each_file_number[d] = each_file_number[d] + 1

        for i in range(length):
            if each_file_number[i] > 0:
                cloud_name = cloud_name_list[i]
                if len(total_obj["unlabeled"][cloud_name]) >= each_file_number[i]:
                    superpoint_inds = np.random.choice(list(total_obj["unlabeled"][cloud_name]), int(each_file_number[i]), replace=False).tolist()
                    _help_seed(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=current_path, cloud_name=cloud_name,
                               superpoint_inds=superpoint_inds, w=w)
                else:
                    superpoint_inds = total_obj["unlabeled"][cloud_name]
                    remain_number = remain_number + each_file_number[i] - len(superpoint_inds)
                    _help_seed(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=current_path, cloud_name=cloud_name,
                               superpoint_inds=superpoint_inds, w=w)

        if remain_number == 0 or len(total_obj["unlabeled"]) == 0:
            # save total_obj
            with open(os.path.join(current_path, "total.pkl"), "wb") as f:
                pickle.dump(total_obj, f)
        else:
            return self._iteration(current_path, total_obj, remain_number, w)

    def sampling(self, model, batch_size, last_round, w):
        if last_round == 0:
            current_path = os.path.join(self.data_path, "superpoint")
        else:
            current_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(last_round))

        round_num = last_round+1
        next_round_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(round_num))
        os.makedirs(next_round_path) if not os.path.exists(next_round_path) else None
        # copy content to next round
        list1 = os.listdir(current_path)
        for file1 in list1:
            p = os.path.join(current_path, file1)
            if os.path.isfile(p) and ".superpoint" not in file1:
                shutil.copyfile(p, os.path.join(next_round_path, file1))

        # read total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "rb") as f:
            total_obj = pickle.load(f)

        self._iteration(current_path=next_round_path, total_obj=total_obj, number=batch_size, w=w)

class AllSampler:
    def __init__(self, input_path, data_path, total_num, sampler_args):
        self.input_path = input_path
        self.data_path = data_path
        self.total_num = total_num
        self.sampler_args = sampler_args

    def sampling(self, model, batch_size, last_round, w, threshold):
        budget = {}
        budget["click"] = batch_size

        if last_round == 1:
            current_path = os.path.join(self.data_path, "superpoint")
        else:
            current_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(last_round))

        round_num = last_round+1
        next_round_path = os.path.join(self.data_path, "sampling", get_sampler_args_str(self.sampler_args), "round_" + str(round_num))
        os.makedirs(next_round_path) if not os.path.exists(next_round_path) else None
        # copy content to next round
        list1 = os.listdir(current_path)
        for file1 in list1:
            p = os.path.join(current_path, file1)
            if os.path.isfile(p) and ".superpoint" not in file1:
                shutil.copyfile(p, os.path.join(next_round_path, file1))

        # read total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "rb") as f:
            total_obj = pickle.load(f)
            if "selected_class_list" not in total_obj:
                total_obj["selected_class_list"] = []

        cloud_name_list = []
        for cloud_name in total_obj["unlabeled"]:
            cloud_name_list.append(cloud_name)

        for cloud_name in cloud_name_list:
            superpoint_inds = total_obj["unlabeled"][cloud_name]
            _help(input_path=self.input_path, data_path=self.data_path, total_obj=total_obj, current_path=next_round_path, cloud_name=cloud_name,
                  superpoint_inds=superpoint_inds, w=w, sampler_args=self.sampler_args, prob_class=None, threshold=threshold, budget=budget, min_size=1)

        # save total_obj
        with open(os.path.join(next_round_path, "total.pkl"), "wb") as f:
            pickle.dump(total_obj, f)

class RandomSampler:
    def __init__(self, input_path, data_path, total_num, sampler_args, min_size):
        self.input_path = input_path
        self.data_path = data_path
        self.total_num = total_num
        self.sampler_args = sampler_args
        self.min_size = min_size

    def _iteration(self, current_path, total_obj, w, threshold, budget):

        rand_inds = np.random.choice(range(self.total_num), budget["click"], replace=False)
        length = len(total_obj["unlabeled"])
        cloud_name_list = []
        for cloud_name in total_obj["unlabeled"]:
            cloud_name_list.append(cloud_name)

        each_file_number = np.zeros([length], dtype=np.int32)
        for ind in rand_inds:
            d = ind % length
            each_file_number[d] = each_file_number[d] + 1

        for i in range(length):
            if each_file_number[i] > 0: