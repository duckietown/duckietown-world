from abc import ABCMeta, abstractmethod
from typing import List, Dict
import numpy as np


class OptimalTrajectoryTracker(metaclass=ABCMeta):
    """
    Keeps track of input trajectories which are optimal
    or might be come optimal in the future (candidate trajectories).

    """
    @abstractmethod
    def digest_traj(self, egoname: str, scores: Dict[str, float]):
        """
        Digests a new trajectory and evaluates

        :param egoname: identifier of trajectory
        :type egoname: str
        :param scores: scores of input trajectory
        :type: scores: Dict[str, float]
        :return: None
        """

    @abstractmethod
    def get_optimal_trajs(self):
        """
        Retrieves identifier and scores of optimal trajectories
        :return: optimal trajectories
        :rtype: Dict[str, Dict[str, float]]
        """


class LexicographicSemiorderTracker(OptimalTrajectoryTracker):
    """
    Optimality is defined in the sense of lexicographic semiordering where
    the optimal trajectories are part of the survivor set where the scores of the rules are
    lexicographically evaluated and all solutions that are within some bound (also called the slack) of
    the absolute optimum are taken to the next step.
    """
    trajs_tracked: Dict[str, Dict[str, float]]
    rules: List[str]
    # TODO change input type of rules, maybe List[Rule]
    # TODO input of slack variables

    def __init__(self, rules):
        self.trajs_tracked = dict()
        self.rules = rules
        # TODO assert if input to rules is valid

    def digest_traj(self, egoname, scores):
        if not self.trajs_tracked:
            assert isinstance(scores, Dict)
            self.trajs_tracked[egoname] = scores
            return

        self.__update_trajs(egoname, scores)

    def __update_trajs(self, egoname, scores):
        """
        Upon digestion of a new trajectory, the stored trajectories will be updated.
        If the new trajectory is a candidate for the optimal trajectory set it will be added
        and trajectories which are no longer candidates will be discarded.

        :param egoname: identifier of trajectory
        :type egoname: str
        :param scores: scores of input trajectory
        :type: scores: Dict[str, float]
        :return: None
        """
        for rule in self.rules:
            filter_index = []
            input_traj_score = scores[rule]
            for item in self.trajs_tracked:
                item_scores = self.trajs_tracked[item]
                item_score = item_scores[rule]
                if item_score + 0.1 < input_traj_score:
                    if self.__discardable(scores, item_scores, rule): # x strictly succeeds current
                        return
                elif item_score > input_traj_score + 0.1:
                    if self.__discardable(item_scores, scores, rule): # x strictly succeeds current
                        filter_index.append(item)
            self.trajs_tracked = {k: v for k, v in self.trajs_tracked.items() if k not in filter_index}
        self.trajs_tracked[egoname] = scores

    def __discardable(self, x, y, rulex):
        """
        Checks whether a trajectory can be discarded, i.e. it will never be optimal.
        A trajectory X will never be optimal if there exists another trajectory Y for which
        there is a rule such that the score of Y is better than the score of X for that rule and they are
        not within the bound and the scores of Y for all other rules of higher priority are better compared to X.

        :param x: scores of trajectory to be possibly discarded
        :type x: Dict
        :param y: scores of trajectory to possibly discards x
        :param rulex:
        :return:
        """
        for rule in self.rules:
            if x[rule] > y[rule]:
                if rulex == rule:
                    return True
            else:
                return False

    def get_optimal_trajs(self):
        optimal_set = self.trajs_tracked
        for rule in self.rules:
            optimal_score = min(optimal_set.values(), key=lambda x: x[rule])[rule]
            print("Optimal score for ", rule, optimal_score)
            # TODO implement for maximal as well
            optimal_set = {k: v for k, v in optimal_set.items() if v[rule] <= 1.1 * optimal_score}
            for item in optimal_set:
                print(item, ":", optimal_set[item][rule])
        return optimal_set


class LexicographicTracker(OptimalTrajectoryTracker):
    """
    Optimality is defined in the sense of lexicographic ordering where a trajectory X is deemed better than
    a trajectory Y if and only if there exists a rule for which the score of X is better than Y and for all
    rules of higher priority they are equivalent.
    """
    optimal_trajs: Dict[str, Dict[str, float]]
    rules: List[str]
    # TODO change input type of rules

    def __init__(self, rules):
        self.optimal_trajs = dict()
        self.rules = rules
        # TODO assert if input to rules is valid

    def digest_traj(self, egoname, scores):
        if not self.optimal_trajs:
            assert isinstance(scores, Dict)
            self.optimal_trajs[egoname] = scores
            return

        for rule in self.rules:
            input_traj_score = scores[rule]
            for item in self.optimal_trajs:
                item_scores = self.optimal_trajs[item]
                item_score = item_scores[rule]
                # TODO implement for max as well
                if item_score < input_traj_score:
                    return
                elif item_score > input_traj_score:
                    self.optimal_trajs.clear()
                    break
        self.optimal_trajs[egoname] = scores

    def get_optimal_trajs(self):
        return self.optimal_trajs


class ProductOrderTracker(OptimalTrajectoryTracker):
    """
    Optimality is defined in the sense of pareto optimality where a trajectory X is deemed better than
    a trajectory Y if and only if there exists no rule for which the score of Y is better than the score of X.
    """
    optimal_trajs: Dict[str, Dict[str, float]]
    rules: List[str]
    # TODO change input type of rules

    def __init__(self, rules):
        self.optimal_trajs = dict()
        self.rules = rules
        # TODO assert if input to rules is valid

    def digest_traj(self, egoname, scores):
        if not self.optimal_trajs:
            assert isinstance(scores, Dict)
            self.optimal_trajs[egoname] = scores
            return

        filter_index = []
        input_scores_as_array = np.fromiter(scores.values(), dtype=float)
        for item in self.optimal_trajs:
            item_scores_as_array = np.fromiter(self.optimal_trajs[item].values(), dtype=float)
            # TODO implement for max as well
            if np.all(np.less_equal(item_scores_as_array, input_scores_as_array)):
                return
            elif np.all(np.less_equal(input_scores_as_array, item_scores_as_array)):
                filter_index.append(item)

        self.optimal_trajs = {k: v for k, v in self.optimal_trajs.items() if k not in filter_index}
        self.optimal_trajs[egoname] = scores

    def get_optimal_trajs(self):
        return self.optimal_trajs
