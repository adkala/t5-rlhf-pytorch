from data import TreeCreator

import torch.utils.data as data
import random
import json


class TreeSet(data.Dataset):
    def __init__(
        self,
        trees: list,
        open_api_key=None,
        generated_descriptions_path=None,
        use_history=True,
        max_length=100
    ):
        super(TreeSet, self).__init__()
        self.trees = trees
        self.use_history = use_history
        self.max_length = max_length

        self.tree_creator = TreeCreator(open_api_key, generated_descriptions_path)
        self.history = []

    def __getitem__(self, index):
        if self.use_history and index < len(self.history):
            return self.history[index]
        tree = random.choice(self.trees)
        x, y = self.tree_creator.sampleTreeToCategoryTreeAndProducts(
            self._sampleTree(tree))
        self.history.append((x, y))
        return x, y

    def __len__(self):
        return self.max_length

    def _sampleTree(self, base, n=10):
        sample = {}
        for _ in range(n):
            c = base
            s = sample
            while True:
                category = random.choice(list(c.keys()))
                c = c[category]
                if isinstance(c, set):
                    if category not in s:
                        s[category] = set()
                    s = s[category]
                    break
                elif category not in s:
                    s[category] = {}
                s = s[category]
            s.add(random.choice(list(c)))
        return sample
    
    def addTree(self, tree):
        self.trees.append(tree)

# wrapper for TreeSet to enable easier data loading for dumped trees
class DumpedTreeSetWrapper(data.Dataset):
    def __init__(self, tree_set: TreeSet, max_length=100):
        self.tree_set = tree_set
        self.max_length = max_length

    def __getitem__(self, index):
        (dct, dp), dpc  = self.tree_set[index]
        return "%s, %s" % (json.dumps(dct), json.dumps(dp)), json.dumps(dpc)

    def __len__(self): # for use in dataloader
        return self.max_length

    def toggleHistory(self):
        self.tree_set.use_history = not self.tree_set.use_history
        print('use_history switched to %s' % str(self.tree_set.use_history))

        
