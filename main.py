import logging
import time

import numpy as np

import rs
from preprocess import read_combined_data, filtering, filtering_test
from utils import save_csv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def main():
    logger.info(' Preprocessing ...')

    data_path = 'data/netflix-prize-data/combined_data_1.txt'
    min_num_users = 50
    min_num_movies = 15
    t_start_0 = time.time()
    data_matrix, uid_idx_map, mid_idx_map = filtering(read_combined_data(data_path),
                                                      min_num_users=min_num_users,
                                                      min_num_movies=min_num_movies)
    logger.debug('after preprocessed, time used {: 2f} sec.'.format(time.time() - t_start_0))
    logger.debug('# of data = {}, user x movie = {}'.format(data_matrix.nnz, data_matrix.shape))

    print()
    logger.info(' Training ...')
    t_start = time.time()
    k = 50
    max_iter = 100
    W, H = rs.non_negative_matrix_factorization(data_matrix, k=k, max_iter=max_iter)
    del data_matrix
    logger.debug('after NMF, time used {: 2f} sec.'.format(time.time() - t_start))
    logger.info('Totally used {: 2f} sec.'.format(time.time() - t_start_0))

    print('>> W ( shape={} )'.format(W.shape))
    print('>> H ( shape={} )'.format(H.shape))

    # testing qualifying
    logger.info(' Testing ...')
    test_path = 'data/netflix-prize-data/qualifying.txt'
    test_df = read_combined_data(test_path, mode='test')
    test_matrix, filtered_test_df = filtering_test(test_df, uid_idx_map, mid_idx_map)
    del test_df

    scored_test_matrix = test_matrix.toarray() * np.inner(W, H.T)
    test_scores = [scored_test_matrix[uid][mid] for uid, mid in zip(test_matrix.row, test_matrix.col)]
    filtered_test_df['rating'] = test_scores

    save_csv(filtered_test_df, 'result/result_nu{}_nm{}.csv'.format(min_num_users, min_num_movies))


if __name__ == '__main__':
    main()
