import numpy as np

def balance_classes(x,y):
    # minimal class
    u,c = np.unique(y, return_counts=True)
    min_class = u[np.argmin(c)]
    min_size = c.min()
    new_x = x[y==min_class]
    new_y = y[y==min_class]
    for cl in u:
        if cl!=min_class:
            x_temp = x[y==cl]
            r_ind = np.random.choice(range(x_temp.shape[0]), size=min_size)
            new_x = np.vstack((new_x, x_temp[r_ind]))
            new_y = np.hstack((new_y, cl*np.ones(min_size)))
    return new_x, new_y
