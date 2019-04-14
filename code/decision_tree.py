import numpy as np

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

        return

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        self._check_input(features)
        self.tree = Tree()
        # breaking situation
        # no features
        if len(self.attribute_names) == 0:

            if (targets == 1).sum() < (targets == 0).sum():
                self.tree.branches.append(Tree(0, "class"))

            else:
                self.tree.branches.append(Tree(1, "class"))

            return self.tree

        # one target value
        if len(np.unique(targets)) == 1:
            self.tree.branches.append(Tree(targets[0], "class"))
            return self.tree
        # store origin attribute list
        _names_ = []
        for attribute in self.attribute_names:
            _names_.append(attribute)

        # find the maximum information gain index
        def Max_gain():

            info_gain = []
            length = len(self.attribute_names)
            for each in range(length):
                info_gain.append(information_gain(features, each, targets))
            maxi_index = info_gain.index(max(info_gain))
            return maxi_index

        max_index = Max_gain()

        cur_attribute = self.attribute_names[max_index]
        _names_.remove(cur_attribute)

        for feature_value in np.unique(features[:, max_index]):  # get the all possible values of feature

            sub_features = features[features[:, max_index] == feature_value]
            sub_targets = targets[features[:, max_index] == feature_value]
            sub_features = np.delete(sub_features, max_index, axis=1)

            sub_tree = DecisionTree(_names_)  # create a new tree

            sub_tree.fit(sub_features, sub_targets)  # construct the tree structure
            sub_tree.tree.value = feature_value
            sub_tree.tree.attribute_name = cur_attribute  # best attribute
            sub_tree.tree.attribute_index = max_index
            self.tree.branches.append(sub_tree.tree)
        return self.tree



    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        self._check_input(features)

        targets_value_list = []

        if len(self.tree.branches) == 1:
            predictions = [[self.tree.branches[0].value] * len(features)]
            return np.array(predictions)

        attri_list = []
        for attri in self.attribute_names:
            attri_list.append(attri)

        for row_feature in features:  # total number of examples
            targets_value = self.helper(self.tree, row_feature, attri_list)
            targets_value_list.append(targets_value)

        return np.array(targets_value_list)

    def helper(self, node, row_feature, attri_list):
        # if len(node.branches) == 0:
        #   return node.value

        for branch in node.branches:
            if branch.attribute_name == "class":
                return branch.value
            elif (branch.value == row_feature[attri_list.index(branch.attribute_name)]).any():
                # print(branch.value, row_feature[branch.attribute_index])
                targets_value = self.helper(branch, row_feature, attri_list)
                return targets_value

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    def EntropyCount(targets):
        targets_true = (targets == 1).sum()
        targets_false = (targets == 0).sum()

        # entropy
        if targets_true == 0 or targets_false == 0:
            return 0
        else:
            p_c_a = targets_false / len(targets)
            p_c_b = targets_true / len(targets)
            res = -1 * (p_c_a * np.log2(p_c_a) + p_c_b * np.log2(p_c_b))
        return res

    trueValues = np.where(features[:, attribute_index] == 1)
    falseValues = np.where(features[:, attribute_index] == 0)
    entropy_total = EntropyCount(targets)
    entropy_1 = EntropyCount(targets[trueValues[0]])
    entropy_2 = EntropyCount(targets[falseValues[0]])
    len_helper = len(trueValues[0]) + len(falseValues[0])
    count_helper = entropy_1 * (len(trueValues[0]) / len_helper) + entropy_2 * (len(falseValues[0]) / len_helper)
    infor_gain = entropy_total - count_helper

    return infor_gain


if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
