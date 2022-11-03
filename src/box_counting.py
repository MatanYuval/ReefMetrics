from collections import Counter
from functools import reduce
from typing import List
import numpy as np
import scipy.stats

from src.setting import POWER_LIMIT, LENGTH_LIMIT, LENGTH_PRECISION


class BoxCounting:
    N = 8

    def __str__(self):
        return f"depth {self.depth}, p {self.p:.4f}, relative p {self.rel_p:.4f}, children {self.children_num}, points {self.coor_num}"

    def __init__(self, depth, coor, box_counting=None):
        self.depth = depth
        self.p = 1
        self.rel_p = 1
        self.rel_p_exp = 1
        self.chi_2 = 0
        self.coor = coor
        self.coor_num = len(coor)
        self.children: List[BoxCounting] = []
        self.box_counting = box_counting

    def assign_one_level_p_for_children(self):
        for child in self.valid_children:
            child.p = 1 / self.children_num

    @property
    def valid_children(self):
        return filter(lambda x: x is not None, self.children)

    @property
    def children_num(self):
        return len(list(self.valid_children))

    def calc_rel_p(self, parent_p=1):
        for child in self.valid_children:
            child.rel_p *= parent_p * child.p
            child.calc_rel_p(child.rel_p)

    def calc_rel_p_exp(self, parent_p=1):
        for child in self.valid_children:
            child.rel_p_exp *= parent_p * (1 / BoxCounting.N)
            child.calc_rel_p_exp(child.rel_p_exp)

    def calc_chi_2(self):
        self.chi_2 = (self.rel_p - self.rel_p_exp) ** 2 / self.rel_p_exp
        for child in self.valid_children:
            child.calc_chi_2()

    def sum_chi_2(self):
        if self.children_num == 0:
            return 0
        return self.chi_2 + sum([child.sum_chi_2() for child in self.valid_children])

    def reset_relative_p(self):
        for child in self.valid_children:
            child.rel_p = 1
            child.reset_relative_p()

    def _relative_p_of_leaves_inner(self):
        if self.children_num == 0:
            return [self.rel_p]

        ans = []

        for child in self.valid_children:
            ans += child._relative_p_of_leaves_inner()

        return ans

    def statistic_summary_of_node(self, max_power):
        self.calc_rel_p()
        self.calc_rel_p_exp()

        p_leaves = self._relative_p_of_leaves_inner()

        exp_n_deepest = 8 ** max_power
        exp_deepest_prob = np.ones(exp_n_deepest)
        exp_deepest_prob /= exp_n_deepest

        obs_deepest_prob = np.zeros(exp_n_deepest)
        obs_deepest_prob[:len(p_leaves)] = p_leaves

        exp_entropy = scipy.stats.entropy(exp_deepest_prob, base=2)
        obs_entropy = scipy.stats.entropy(obs_deepest_prob, base=2)
        print(f"sum of leaves probabilities: {sum(p_leaves)}")
        print(f"entropy of expected distribution: {exp_entropy} ~ log2({exp_n_deepest})")
        print(f"entropy of observed distribution: {obs_entropy}")

        self.calc_chi_2()
        chi_2_sum = self.sum_chi_2()
        print(f"chi_2 sum: {chi_2_sum}")

    @classmethod
    def inner_tree_builder(cls, coor, start, end, max_power):
        # checks
        assert (coor >= start).prod(axis=1).all()
        assert (coor <= end).prod(axis=1).all()

        curr_length = end - start
        assert np.allclose(curr_length, curr_length[0])
        assert all(curr_length >= 0)

        # calc current power of 2
        curr_power = np.log2(curr_length[0])

        # stop condition 1: if the box contain no point at all or only 1 point
        if len(coor) in [0]:
            return None

        if len(coor) in [0, 1]:
            # return coor box for each of the "following" boxes (until the lowest size of box)
            power_limit = int(np.ceil(curr_power) - POWER_LIMIT)
            return cls(depth=max_power-curr_power, coor=coor,
                       box_counting={np.round(curr_length[0] / 2 ** i, LENGTH_PRECISION): len(coor)
                                     for i in range(power_limit + 1)})

        # stop condition 2: if the box length is the lowest size of box (defined by LENGTH_LIMIT = 2 ** POWER_LIMIT)
        if curr_length[0] <= LENGTH_LIMIT:
            # return the box itself
            return cls(depth=max_power-curr_power, coor=coor,
                       box_counting={np.round(curr_length[0], LENGTH_PRECISION): 1})

        division_list, start_list, end_list = cls.get_cube_division(coor, start, end)

        # recursively find the box counting in each smaller box
        curr_node = cls(depth=max_power - curr_power, coor=coor)

        for i, (p, start, end) in enumerate(zip(division_list, start_list, end_list)):
            assert (p >= start).prod(axis=1).all()
            assert (p <= end).prod(axis=1).all()
            curr_node.children.append(cls.inner_tree_builder(p, start, end, max_power))

        curr_node.assign_one_level_p_for_children()

        # summarize the results from all recursive calls
        # add the current box count
        curr_dict = {np.round(curr_length[0], LENGTH_PRECISION): 1}
        curr_node.assign_box_counting_based_on_children(curr_dict)

        return curr_node

    @property
    def children_box_counting(self):
        return [child.box_counting for child in self.valid_children]

    def assign_box_counting_based_on_children(self, curr_dict):
        sum_dict = dict(reduce(lambda x, y: Counter(x) + Counter(y), [curr_dict] + self.children_box_counting))
        self.box_counting = sum_dict

    @classmethod
    def get_cube_division(cls, coor, start, end):
        # calc the division of the box
        mid = (start + end) / 2

        x_lower = coor[:, 0] < mid[0]
        y_lower = coor[:, 1] < mid[1]
        z_lower = coor[:, 2] < mid[2]

        # divide the coor_num into 8 small boxes
        p_000 = coor[x_lower & y_lower & z_lower]
        p_001 = coor[x_lower & y_lower & ~z_lower]
        p_010 = coor[x_lower & ~y_lower & z_lower]
        p_011 = coor[x_lower & ~y_lower & ~z_lower]
        p_100 = coor[~x_lower & y_lower & z_lower]
        p_101 = coor[~x_lower & y_lower & ~z_lower]
        p_110 = coor[~x_lower & ~y_lower & z_lower]
        p_111 = coor[~x_lower & ~y_lower & ~z_lower]

        # number of coor_num in all box should equal number of coor_num in the iteration
        assert len(p_000) + len(p_001) + len(p_010) + \
               len(p_011) + len(p_100) + len(p_101) + \
               len(p_110) + len(p_111) == len(coor)

        division_list = [p_000, p_001, p_010, p_011,
                         p_100, p_101, p_110, p_111]

        start_list = [np.array([start[0], start[1], start[2]]),
                      np.array([start[0], start[1], mid[2]]),
                      np.array([start[0], mid[1],   start[2]]),
                      np.array([start[0], mid[1],   mid[2]]),
                      np.array([mid[0],   start[1], start[2]]),
                      np.array([mid[0],   start[1], mid[2]]),
                      np.array([mid[0],   mid[1],   start[2]]),
                      np.array([mid[0],   mid[1],   mid[2]])]

        end_list = [np.array([mid[0], mid[1], mid[2]]),
                    np.array([mid[0], mid[1], end[2]]),
                    np.array([mid[0], end[1], mid[2]]),
                    np.array([mid[0], end[1], end[2]]),
                    np.array([end[0], mid[1], mid[2]]),
                    np.array([end[0], mid[1], end[2]]),
                    np.array([end[0], end[1], mid[2]]),
                    np.array([end[0], end[1], end[2]])]

        assert len(division_list) == 8
        assert len(start_list) == 8
        assert len(end_list) == 8

        return division_list, start_list, end_list

    @classmethod
    def get_tree(cls, a: np.ndarray):
        # checks
        assert a.ndim == 2
        assert a.shape[1] == 3
        #assert all(a.min(0) == 0)

        # calc bounding box limits
        bb_limits = np.array(list(zip(a.min(0), a.max(0)))).T
        bb_lengths = bb_limits[1] - bb_limits[0]
        max_length = bb_lengths.max()
        largest_power_of_2 = int(np.ceil(np.log2(max_length)))
        largest_box_length = 2 ** largest_power_of_2

        start = np.array([0, 0, 0])
        end = np.array([max_length, max_length, max_length])

        box_counting_tree = cls.inner_tree_builder(a, start, end, largest_power_of_2)
        return box_counting_tree

    def get_sorted_box_counting_dict(self):
        assert self.box_counting is not None
        return dict(sorted(self.box_counting.items(), key=lambda item: item[1]))
