import logging

import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def read_combined_data(data_path, mode='train') -> pd.DataFrame:
    logger.info('> Reading Data from "{}"'.format(data_path))
    lines = []
    with open(data_path, 'r') as fp:
        for line in fp.readlines():
            if ":" in line:
                movie_id = line.split(':')[0]
                continue

            others = line[:-1].split(',')
            lines.append([movie_id] + others)
    if mode in ['train', 'dev']:
        columns = ['movie_id', 'user_id', 'rating', 'date']
    else:
        columns = ['movie_id', 'user_id', 'date']
    return pd.DataFrame(lines, columns=columns)


def build_count_of_groups(df: pd.DataFrame, count_col):
    rnd_col = list(set(df.columns) - {count_col})[0]
    return df.groupby(count_col)[rnd_col].count()


def filter_ids_by_count(series: pd.Series, min_count, max_count):
    return list(
        series.loc[~series.isin(range(min_count, max_count))].index.values)


def filtering_test(data_df: pd.DataFrame, uid_idx_map, mid_idx_map):
    # filter data by movie_ids
    data_df = data_df.loc[data_df['movie_id'].isin(mid_idx_map)]

    # filter data by user_ids
    data_df = data_df.loc[data_df['user_id'].isin(uid_idx_map)]

    row = data_df['user_id'].map(uid_idx_map)
    col = data_df['movie_id'].map(mid_idx_map)
    data = list([1] * len(data_df))
    return sp.coo_matrix((data, (row, col)),
                         shape=(len(uid_idx_map), len(mid_idx_map))), data_df


def filtering(data_df: pd.DataFrame,
              min_num_users=5,
              max_num_users=None,
              min_num_movies=5,
              max_num_movies=None):
    logger.info('# of data before filtering: {}'.format(data_df.shape))

    logger.info('> Counting watched users group by movies')
    movie_count = build_count_of_groups(data_df, 'movie_id')
    logger.debug('# of movies before filtering: {}'.format(len(movie_count)))

    if not max_num_users:
        max_num_users = len(movie_count)
    movie_ids = filter_ids_by_count(movie_count, min_num_users, max_num_users)
    del movie_count
    logger.debug('# of movies after filtering: {}'.format(len(movie_ids)))

    # filter data by movie's 熱度
    data_df = data_df.loc[data_df['movie_id'].isin(movie_ids)]

    logger.info(
        '> Counting number of movies being watched by users, i.e. group by users'
    )
    user_count = build_count_of_groups(data_df, 'user_id')
    logger.debug('# of users before filtering: {}'.format(len(user_count)))
    if not max_num_movies:
        max_num_movies = len(user_count)
    user_ids = filter_ids_by_count(user_count, min_num_movies, max_num_movies)
    del user_count
    logger.debug('# of users after filtering: {}'.format(len(user_ids)))

    # filter data by user's 活躍度
    data_df = data_df.loc[data_df['user_id'].isin(user_ids)]
    logger.debug('# of data after filtering: {}'.format(len(data_df)))

    # build id-idx map
    uid_idx_map = {v: k for k, v in enumerate(user_ids)}
    mid_idx_map = {v: k for k, v in enumerate(movie_ids)}

    row = data_df['user_id'].map(uid_idx_map)
    col = data_df['movie_id'].map(mid_idx_map)
    data = data_df['rating'].astype(int)

    return sp.csr_matrix(
        (data, (row, col)),
        shape=(len(user_ids), len(movie_ids))), uid_idx_map, mid_idx_map
