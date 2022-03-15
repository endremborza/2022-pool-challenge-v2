import numpy as np
from scipy.spatial import KDTree

def minkowski_distance_p(x, y, p=2):
    """
    Compute the p-th power of the L**p distance between two arrays.
    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.
    Parameters
    ----------
    x : (M, K) array_like
        Input array.
    y : (N, K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    Examples
    --------
    >>> minkowski_distance_p([[0,0],[0,0]], [[1,1],[0,1]])
    array([2, 1])
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf:
        return np.amax(np.abs(y-x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y-x), axis=-1)
    else:
        return np.sum(np.abs(y-x)**p, axis=-1)
    
def query_subset(self, x, subset, p=2):
    subset_vec = np.zeros(self.n)
    subset_vec[subset] = 1
    node = self.tree
    return _query_subset(self, node, x, subset_vec, p)

def _query_subset(self, node, x, subset, p):
    # initialize a boolean array of size n
    child_vec = np.zeros_like(subset)
    
    if isinstance(node, KDTree.innernode):
        # set boolean switches to one if that idx beneath this node
        child_vec[node._node.indices] = 1
        # does this branch contain children in the subset
        if np.dot(child_vec, subset) >= 1:
            # determine which way to traverse
            if x[node._node.split_dim] < node._node.split:
                near, far = node._node.lesser, node._node.greater
            else:
                near, far = node._node.greater, node._node.lesser
            near = _query_subset(self, near, x, subset, p)
            
            # if near branch resulted in a dead end, check the far
            if not near:
                return _query_subset(self, far, x, subset, p)
            # is the further branch closer
            far = _query_subset(self, far, x, subset, p)
            if far:
                if near[1] > far[1]:
                    return far
            return near
    else:
        child_vec[node.indices] = 1
        # does this leaf intersect with subset
        if np.dot(child_vec, subset) >= 1: 
            # get the universe of intersecting points
            universe = np.arange(self.n)[subset.astype(bool)]
            candidates = np.intersect1d(universe, node.indices)
            # compute the candidatae distances
            distances = ((pt, minkowski_distance_p(x, self.data[pt], p))
                         for pt in candidates)
            #return the closest point
            return min(distances, key=lambda tup: tup[1])

def test_subset_query(runs):
    # set up a random kdtree and 
    succ = 0
    for i, x in enumerate(range(runs), 1):
        n = 100000
        coords = np.random.uniform(-20, 20, [n, 2])
        subset = np.random.choice(np.arange(n), 1000, replace=False)
        pt = coords[5]

        tree = KDTree(coords, leafsize=np.log2(n))
        ix, d = query_subset(tree, pt, subset)
        fnn = tree.data[ix]

        brute_force = min(((i, minkowski_distance_p(pt, tree.data[i])) 
                           for i in subset), key=lambda tup: tup[1])
        try:
            assert brute_force == (ix, d)
            succ +=1
        except:
            pass
    print('passed {0:.2f}% of trials'.format(100.0 * succ / runs))

if __name__ == '__main__':
    test_subset_query(30)