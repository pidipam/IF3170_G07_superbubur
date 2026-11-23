import pickle
import numpy as np

class Node : 
    def __init__ (
        self,
        feature=None,           # Atribut yang diuji
        threshold=None,         # Threshold pencabangan atribut numerik
        children=None,          # Cabang anak
        class_counts=None,      # Array jumlah class 0 dan 1
        label=None,             # Nilai label leaf
        left=None,              # Node kiri
        right=None,             # Node kanan
        default_child=None      # Cabang dengan sampel terbanyak    
    ) :            
        self.feature = feature
        self.threshold = threshold
        self.children = children if children is  not None else {}
        self.class_counts = class_counts
        self.label = label
        self.left = left
        self.right = right
        self.default_child = default_child


class DecisionTree:
    def __init__ (
        self,
        max_depth=None,             # Max depth tree
        min_samples_split=2,        # Min sample setiap node
        min_samples_leaf=1,         # Min sample setiap leaf
        categorical_idx=None,       # List idx kolom kategorikal
        max_thresholds=32           # Max threshold tipe numerik
    ) : 
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_idx = categorical_idx if categorical_idx is not None else []
        self.root = None
        self.max_thresholds=max_thresholds

    def entropy(self, y) :
        total_tuple = len(y)
        if(total_tuple == 0) : 
            return 0
        
        n_true = np.sum(y == 1)
        n_false = total_tuple - n_true

        p_false =  n_false / total_tuple
        p_true = n_true / total_tuple

        if(p_false == 0) : 
            return - p_true * np.log2(p_true)
        elif(p_true == 0) : 
            return - p_false * np.log2(p_false)

        return -p_false * np.log2(p_false) - p_true * np.log2(p_true)

    def split_information(self, X_col, y, is_categorical=True, threshold=None) : 
        total_samples = len(X_col)
        if total_samples == 0:
            return 0.0

        split_info = 0.0

        if is_categorical:
            categories = np.unique(X_col)
            for c in categories:
                prop = np.sum(X_col == c) / total_samples
                if prop > 0:
                    split_info -= prop * np.log2(prop)
        else:
            prop_left = np.sum(X_col <= threshold) / total_samples
            prop_right = np.sum(X_col > threshold) / total_samples

            if prop_left > 0:
                split_info -= prop_left * np.log2(prop_left)
            if prop_right > 0:
                split_info -= prop_right * np.log2(prop_right)

        return split_info
    
    def information_gain(self, X_col, y, is_categorical=True, threshold=None) : 
        total_samples = len(y)
        if(total_samples == 0) : 
            return 0

        sum_entropy_children = 0

        if(is_categorical) :        # Atribut kategorikal
            categories = np.unique(X_col)
            for c in categories : 
                y_child = y[X_col == c]
                weight = len(y_child) / total_samples
                sum_entropy_children += weight * self.entropy(y_child)

        else :                      # Atribut Numerik
            y_left = y[X_col <= threshold]
            y_right = y[X_col > threshold]

            sum_entropy_children += (len(y_left) / total_samples) * self.entropy(y_left)
            sum_entropy_children += (len(y_right) / total_samples) * self.entropy(y_right)
        
        return self.entropy(y) - sum_entropy_children

    def gain_ratio(self, X_col, y, is_categorical=True, threshold=None) :
        split_info = self.split_information(X_col, y, is_categorical, threshold)
        if(split_info == 0) : 
            return 0

        return self.information_gain(X_col, y, is_categorical, threshold) / split_info

    def best_split(self, X, y) :
        best_feature_idx = None
        best_threshold = None
        best_gain = 0

        for col_idx in range(X.shape[1]) : 
            X_col = X[:, col_idx]
            null_mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in X_col])
            valid_mask = ~null_mask

            if(np.sum(valid_mask) == 0) :
                continue

            X_valid = X_col[valid_mask]
            y_valid = y[valid_mask]

            if(col_idx in self.categorical_idx) : 
                gain_rat = self.gain_ratio(X_valid, y_valid, True, None)
                if(gain_rat > best_gain) : 
                    best_gain = gain_rat
                    best_feature_idx = col_idx
                    best_threshold = None

            else :
                X_num = X_valid.astype(float)
                unique_vals = np.unique(X_num)

                if(len(unique_vals) <= 1) :
                    continue

                mids = (unique_vals[:-1] + unique_vals[1:]) / 2.0
                if(len(mids) > self.max_thresholds) :
                    mids = np.unique(np.quantile(X_num, np.linspace(0.02, 0.98, self.max_thresholds)))

                for mid in mids :
                    n_left = np.sum(X_num <= mid)
                    n_right = len(X_num) - n_left

                    if(n_left < self.min_samples_leaf or n_right < self.min_samples_leaf) :
                        continue                      

                    gain_rat = self.gain_ratio(X_num, y_valid, False, mid)
                    if(gain_rat > best_gain) :
                        best_gain = gain_rat
                        best_feature_idx = col_idx
                        best_threshold = mid

        return best_feature_idx, best_threshold, best_gain

    # Ketika membuat model, pastikan max_depth != None
    def decision_tree_learning(self, X, y, parent_y=None, depth=0) : 
        # Jika data kosong, ambil mayoritas dari parent
        if(len(y) == 0) :
            label, counts = np.unique(parent_y, return_counts=True)
            class_counts = dict(zip(label, counts))
            majority = label[np.argmax(counts)]
            return Node(label=majority, class_counts=class_counts)

        # Hitung distribusi kelas dan kelas mayoritas saat ini
        label, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(label, counts))
        majority = label[np.argmax(counts)]

        # Kasus semua data sudah satu kelas (murni)
        if(len(label) == 1) :
            return Node(label=majority, class_counts=class_counts)

        # Kasus sudah mencapai batas max depth atau min samples split
        if((self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split) :
            return Node(label=majority, class_counts=class_counts)

        # Cari atribut terbaik untuk split
        best_feature_idx, best_threshold, best_gain = self.best_split(X, y)

        # Kasus tidak ada split yang menghasilkan gain positif
        if(best_feature_idx is None or best_gain <= 0) :
            return Node(label=majority, class_counts=class_counts)

        node = Node(
            feature=best_feature_idx,
            threshold=best_threshold,
            label=majority,
            class_counts=class_counts
        )

        X_col = X[:, best_feature_idx]
        null_mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in X_col])
        
        if(best_feature_idx in self.categorical_idx) :     # Atribut kategorikal
            categories = np.unique(X_col[~null_mask])
            best_cat = None
            max_count = -1

            for c in categories :
                count = np.sum((~null_mask) & (X_col == c))
                if(count > max_count) :
                    max_count = count
                    best_cat = c

            for c in categories :
                mask = (~null_mask) & (X_col == c)

                if(c == best_cat) :
                   mask = mask | null_mask
            
                if(np.sum(mask) >= self.min_samples_leaf) :
                    node.children[c] = self.decision_tree_learning(X[mask], y[mask], y, depth + 1)

            if(len(node.children) == 0) :
                return Node(label=majority, class_counts=class_counts)

            node.default_child = node.children.get(best_cat, None)

        else :      # Atribut numerik
            X_num = np.array([np.nan if m else float(v) for v, m in zip(X_col, null_mask)])                                       
            left_mask = (X_num <= best_threshold)
            right_mask = (X_num > best_threshold)

            if(np.sum(left_mask) >= np.sum(right_mask)) :
                left_mask = left_mask | null_mask
                go_left = True
            else :
                right_mask = right_mask | null_mask
                go_left = False

            if(np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf) :
                return Node(label=majority, class_counts=class_counts)

            node.left = self.decision_tree_learning(X[left_mask], y[left_mask], y, depth + 1)       # Nilai <=
            node.right = self.decision_tree_learning(X[right_mask], y[right_mask], y, depth + 1)    # Nilai >
            node.default_child = node.left if go_left else node.right
        
        return node

    def fit(self, X, y): 
        X = np.asarray(X, dtype=object)
        y = np.asarray(y)
        self.classes_ = np.array([0, 1])
        self.root = self.decision_tree_learning(X, y)
        return self


    def predict_row(self, x, node) :
        while(len(node.children) > 0 or node.left is not None or node.right is not None) :
            val = x[node.feature]

            # Kalau nilai fitur NULL, maka ke cabang default
            if(val is None or (isinstance(val, float) and np.isnan(val))) :
                if(node.default_child is not None) :
                    node = node.default_child
                    continue
                else :
                    break

            if(node.feature in self.categorical_idx) :     # Atribut kategorikal
                if(val in node.children) :
                    node = node.children[val]
                else :
                    break
            else :                                         # Atribut numerik
                if(float(val) <= node.threshold and node.left is not None) :
                    node = node.left
                elif(float(val) > node.threshold and node.right is not None) :
                    node = node.right
                else :
                    break

        return node

    def predict(self, X):                # label 0/1
        probs = self.predict_probability(X)
        best_class_idx = np.argmax(probs, axis=1)
        return self.classes_[best_class_idx]

    def predict_probability(self, X):     # probabilitas tiap kelas, shape (n, 2)
        X = np.asarray(X, dtype=object)

        probs = []

        for x in X :
            leaf = self.predict_row(x, self.root)
            total = sum(leaf.class_counts.values())
            row_prob = [leaf.class_counts.get(c, 0) / total for c in self.classes_]
            probs.append(row_prob)

        return np.asarray(probs)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model