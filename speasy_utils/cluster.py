import numpy as np

def _stack_arrays(a,b):
    if (a is None) and (b is None):
        return np.array([])
    if a is None:
        return b
    if b is None:
        return a
    if a.size==0 and b.size==0:
        return np.array([])
    if b.size==0:
        return a
    if a.size==0:
        return b
    if len(a.shape)==1:
        return np.hstack((a,b))
    r= np.vstack((a, b))
    return r
        
def cluster_test(x, y, xyz, method, n_clusters=3, tol=.95, min_size=None, verbose=False):
    if verbose:
        print(f"cluster_test , x.shape: {x.shape}, y.shape: {y.shape}, n_clusters: {n_clusters}")
    if min_size is None:
        min_size = int(.1 * (x.shape[0] / n_clusters))

    
    if x.shape[0] < min_size:
        return None,None,None

    if x.shape[0] <= n_clusters:
        return None,None,None
    
    features, labels, positions = np.array([]),np.array([]), np.array([])
    try:
        spct = method(n_clusters=n_clusters)
    except:
        spct = method()
    z = spct.fit_predict(x)
    classes = set(z)
    
    for i in classes:
        n_class = np.sum(z==i)
        unq, cnts = np.unique(z, return_counts=True)
        if n_class < min_size:
            continue
            
        if cnts.max()/cnts.sum() > tol:
            t_y = np.ones(np.sum(z==i))
            u, c = np.unique(y[z==i], return_counts=True)

            features = _stack_arrays(features, x[z==i])
            positions = _stack_arrays(positions, xyz[z==i])
            
            labels = _stack_arrays(labels, u[np.argmax(c)] * t_y)
        else:
            f,l,pos = cluster_test(x[z==i],y[z==i], xyz[z==i], method, n_clusters=n_clusters-1, tol=tol, min_size=min_size)
            if f is None:
                continue
            if f.shape[0] != l.shape[0]:
                raise Exception("1 Features and labels have to have same size")
            features = _stack_arrays(features, f)
            labels = _stack_arrays(labels, l)
            positions = _stack_arrays(positions, pos)
                
        
        
    if features.shape[0] != labels.shape[0]:
        raise Exception("2 Features and labels have to have same size")
    return features, labels, positions


