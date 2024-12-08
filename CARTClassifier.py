import numpy as np

def to_one_hot(yX: np.ndarray): # Used while testing. There is a internal to_one_hot function in DecisionTree class
    y = yX[:, 0]
    X = yX[:, 1:] 
    
    n_attributes = X.shape[1]
    n_instances = X.shape[0]

    listx = []
    for attribute in range(n_attributes):
        listx.append(list(set(X[:, attribute])))
    
    one_hot_attributes = np.empty((0, n_instances))
    for i in range(n_attributes):
        for j in listx[i][:-1]:
            att = (X[:, i] <= j)
            one_hot_attributes = np.vstack((one_hot_attributes, att))
    
    one_hot_attributes = np.vstack((y, one_hot_attributes))
    return listx, one_hot_attributes.T


def node_score_gini(y: list):

    classes = list(set(y))
    gini = 1
    for i in classes:
        prob = y.count(i) / len(y)
        gini -= prob ** 2
    return gini

class Node:
 
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class CARTClassifier:
    
    def __init__(self, data, validation_data=None, gain_function=node_score_gini, max_depth=100):

        self.listx = []
        # to_one_hot
        one_hot_data = self.to_one_hot(data)
        # print('one_hot_data', one_hot_data)
        
        y = [row[0] for row in one_hot_data]
        self.classes = list(set(y))
        self.majority_class = max(self.classes, key=y.count)

        self.max_depth = max_depth
        self.root = Node(label = self.majority_class)
        self.gain_function = gain_function

        indices = list(range(1, len(one_hot_data[0])))
        # print('indices', indices)

        self._split_recurs(self.root, one_hot_data, indices)
 
        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)


    def to_one_hot(self, yX: np.ndarray):
        y = yX[:, 0]
        X = yX[:, 1:]
        
        n_attributes = X.shape[1]
        n_instances = X.shape[0]
        one_hot_attributes = np.empty((0, n_instances))
        
        if self.listx == []:
            for attribute in range(n_attributes):
                self.listx.append(list(set(X[:, attribute])))
        
        for i in range(n_attributes):
            for j in self.listx[i][:-1]:
                att = (X[:, i] <= j)
                one_hot_attributes = np.vstack((one_hot_attributes, att))
        
        one_hot_attributes = np.vstack((y, one_hot_attributes))
        return one_hot_attributes.T
    

    def predict(self, features):

        return self._predict_recurs(self.root, features)


    def accuracy(self, data):

        return 1 - self.loss(data)


    def loss(self, data):

        one_hot_data = self.to_one_hot(data)

        cnt = 0.0
        test_Y = [row[0] for row in one_hot_data]
        for i in range(len(one_hot_data)):
            prediction = self.predict(one_hot_data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(one_hot_data)


    def _predict_recurs(self, node, row):
        
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        
        if not node.isleaf:
            if node.left is not None:
                self._prune_recurs(node.left, validation_data)
            
            if node.right is not None:
                self._prune_recurs(node.right, validation_data)
            
            if (node.left.isleaf) and (node.right.isleaf):
                original_loss = self.loss(validation_data)
                original_label = node.label
                left = node.left
                right = node.right

                node.isleaf = True
                node.left = None
                node.right = None
                loss = self.loss(validation_data)
                if original_loss < loss:
                    node.isleaf = False
                    node.label = original_label
                    node.left = left
                    node.right = right
        return    
                    
        
    def _is_terminal(self, node, data, indices):

        y = [row[0] for row in data]
        # print('y', y)        
        
        is_terminal = node.isleaf
        if len(data) == 0 or len(indices) == 0 or len(set(y)) == 1 or node.depth == self.max_depth:
            is_terminal = True            
        # print(len(data) == 0, len(indices) == 0, len(set(y)) == 1, node.depth == self.max_depth)
        

        if len(data) == 0: 
            label = self.majority_class
            # print('if', label)
        else:
            label = max(list(set(y)), key=y.count)
            # print('else', label)
        
        return is_terminal, label
        

    def _split_recurs(self, node, data, indices):

        is_terminal, label = self._is_terminal(node, data, indices)
        node.label = label
        # print('nodelabel', node.label)

        if is_terminal:
            node.isleaf = True
            node.left = None
            node.right = None
            return
        
        if not node.isleaf:
            best_gain = -float('inf')
            best_index = None

            for index in indices:
                gain = self._calc_gain(data, index, self.gain_function)
                if gain > best_gain:
                    best_gain = gain
                    best_index = index

            # print('best_gain', best_gain, 'best_index', best_index)
            
            node.index_split_on = best_index
            indices.remove(best_index)
            node._set_info(best_gain, len(data))

            left_data = [row for row in data if row[best_index] == 0]
            right_data = [row for row in data if row[best_index] == 1]

            node.left = Node(depth=node.depth + 1)
            node.right = Node(depth=node.depth + 1)

            self._split_recurs(node.left, left_data, indices)
            self._split_recurs(node.right, right_data, indices)            
            

    def _calc_gain(self, data, split_index, gain_function=node_score_gini):

        y = [row[0] for row in data]
        xi = [row[split_index] for row in data]
        y_x0 = [row[0] for row in data if row[split_index] == 0]
        y_x1 = [row[0] for row in data if row[split_index] == 1]

        # print('y', y, 'xi', xi, 'y_x0', y_x0, 'y_x1', y_x1)
        if len(y) != 0 and len(xi) != 0:
            Px1 = xi.count(1) / len(xi)
            Px0 = xi.count(0) / len(xi)
            
            gain = gain_function(y) - (Px0 * gain_function(y_x0) + Px1 * gain_function(y_x1))

        else:
            gain = 0

        # print('gain', gain)
        return gain
    

    def print_tree(self):

        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + '1 -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):

        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):

        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
