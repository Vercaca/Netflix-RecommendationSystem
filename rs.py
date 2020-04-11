from sklearn.decomposition import NMF


def non_negative_matrix_factorization(array, k, max_iter=100, init='random', random_state=0, verbose=True):
    model = NMF(n_components=k, max_iter=max_iter, init=init, random_state=random_state, verbose=verbose)

    W = model.fit_transform(array)
    H = model.components_
    return W, H

