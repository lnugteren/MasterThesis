# Import libraries
import glob
import warnings
import orjson as json
import os
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import Counter
import math
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import multiprocessing as mp
import networkx as nx
import itertools
import sys
stopwords = stopwords.words('english')


def get_date(unix_time):
    """
    Function that converts unix_time (as in the Amazon data) to datetime format.
    :param unix_time: the unix time
    :return: the datetime format of the unix time
    """
    d = datetime.fromtimestamp(unix_time)
    return d.date()


def convert(score):
    """
    Function that converts a string fraction into a percentage.
    :param score: string fraction
    :return: percentage
    """
    if score == '0/0':
        return 0.0
    num, den = score.split('/')
    return round((float(num) / float(den)) * 100, 6)


def load_data(path):
    """
    Function that loads in the data from a .json file and transfers into a Pandas data frame.
    :param path
    :return: data frame
    """
    data = []
    with open(path, 'r') as f:
        for l in f:
            data.append(json.loads(l.strip()))

    return pd.DataFrame(data)


def too_large_to_choose(meta):
    """
    Function that converts the data into smaller csv chunks to reduce the runtime and space of loading in the data.
    Assumes the csv chunks are already made using model_RQ1
    :param meta: boolean, whether it's the review or meta data
    :return: data frame of the data
    """
    # loading in the chunks
    if meta:
        all_files = sorted(glob.glob(f'Data/csvs_meta_{category}/*.csv'))
    else:
        all_files = sorted(glob.glob(f'Data/csvs_{category}/*.csv'))

    li = []

    for filename in all_files:
        # print(f"now doing {filename}")
        df = pd.read_csv(filename, sep=';\t', header=None)
        li.append(df)

    return pd.concat(li, axis=0, ignore_index=True)


def choose_data(data):
    """
    Choosing the data that you want to load in. Either through load_data or if it's too large through chunks
    :param data: category you want to load
    :return: reviews and meta data
    """
    if category == "Electronics" or category == "Home_and_Kitchen":
        df_meta = too_large_to_choose(True)
        df_meta.columns = ['asin', 'date']
    else:
        metadata = load_data(f'Data/meta_{data}.json')
        df_meta = metadata[['asin', 'date']]

    df_review = too_large_to_choose(False)
    df_review.columns = ['overall', 'vote', 'asin', 'reviewText', 'summary', 'unixReviewTime', 'reviewerID']

    return df_review, df_meta


def nr_of_votes(df, vote_def):
    """
    Function that filters on the minimum number of votes (inclusion criterium)
    :param df
    :param vote_def: minimum number of votes
    :return: filtered df
    """
    df2 = df.fillna(0)

    # convert into usable types
    df2['vote'] = df2['vote'].astype(str).apply(lambda x: x.replace(',', ''))
    df2['vote'] = df2['vote'].astype(float)

    return df2.loc[df2['vote'] >= vote_def]


def remove_duplicates(df):
    """
    Function that removed the duplicated reviews.
    :param df
    :return: df without duplicates.
    """
    return df.drop_duplicates(subset=['vote', 'reviewText', 'summary'], keep='first')


def nr_of_total_votes(df, df_meta):
    """
    Calculates and add the total number of votes per product for calculating the GT.
    :param df
    :param df_meta
    :return: merged df including number of total votes.
    """
    review_amounts = df.groupby(['asin']).size().reset_index(name='#reviews')
    review_amounts_df = pd.merge(df_meta, review_amounts, on='asin', how='left')

    return review_amounts_df


def prepare_df(review_data, meta_data, asin_match, f):
    """
    Preparing the df to only look at the chosen product and include the GT helpfulness
    :param review_data
    :param meta_data
    :param asin_match: product ID we are looking at
    :param f: file for writing output
    :return: updated and filtered df
    """
    # Filter on product ID
    reviews = review_data[review_data.asin == asin_match].drop_duplicates(keep='first')
    product = meta_data[meta_data.asin == asin_match].drop_duplicates(keep='first')

    # Include date, elapsed days, and GT
    reviews['date'] = reviews['unixReviewTime'].astype(int).apply(lambda x: get_date(x))
    maxDays = reviews['date'].max()
    reviews['elapsedDays'] = reviews['date'].apply(lambda x: maxDays - x)
    reviews['elapsedDays'] = reviews['elapsedDays'].apply(lambda x: x.days)

    reviews['helpfulness'] = [y / np.sqrt(x) if x > 0 else y for x, y in zip(reviews['elapsedDays'], reviews['vote'])]

    return reviews, product


def clean_text(text):
    """
    Function that cleans the review text data.
    :param text: review
    :return: cleaned review
    """
    text_tokens = word_tokenize(text.lower())
    tokens_without_sw = [word for word in text_tokens if word not in stopwords and word not in string.punctuation]
    text = ' '.join(tokens_without_sw)
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = ' '.join(text.split())
    text = text.strip()

    return text


def text_to_vector(text):
    """
    Convert text to a vector for calculating the cosine similarity
    :param text
    :return: vector of text
    """
    text = text.lower().strip()
    words = WORD.findall(text)
    return Counter(words)


def get_cosine(vec1, vec2):
    """
    Calculate the cosine similarity between two text vectors
    :param vec1: vector of first text
    :param vec2: vector of second text
    :return: cosine similarity
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def show_graph_with_labels(gr):
    """
    Visualise graph with its labels
    :param gr: graph
    :return: plot showing the graph
    """
    nx.draw(gr, node_size=500, with_labels=True)
    plt.show()


def get_graph_from_matrix(matrix, new_labels={}):
    """
    Function to find the graph corresponding to the matrix
    :param matrix:
    :param new_labels: if added, labels will be updated in the show_graph
    :return: graph
    """
    rows, cols = np.where(matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    if new_labels:
        gr = nx.relabel_nodes(gr, new_labels)
    return gr


def get_adjacency_matrix(df, col_name):
    """
    Function that clusters the data using 3-nn, creates an adjadency matrix as a result
    :param df: review or summary data
    :param col_name: either summary or review
    :return: matrix
    """
    # initialise the matrix
    mat = np.zeros((len(df), len(df)))

    # loop over all summaries per summary to find the cos_sim, no self-loops, not the same reviewer, the 3 highest sim get an edge, if no similarities, no edges
    for i, row in df.reset_index().iterrows():
        vec_from = text_to_vector(row[col_name])
        reviewer = row['reviewerID']
        similarities = {}
        for j, col in df.reset_index().iterrows():
            # no self loops and not the same reviewer
            if i is not j and reviewer != col['reviewerID']:
                # calculate the cos_sim per summary pair
                similarities[j] = get_cosine(vec_from, text_to_vector(col[col_name]))
        # sort cos_sim from highest to lowest
        similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
        # take the 3 highest for k-NN, k=3 nearest neighbors
        for val in list(similarities)[:n_neighbor]:
            if similarities[val] != 0:  # if commented, there will always be 3 neighbors, even when sim = 0
                # add edge
                mat[i][val] = 1

    return mat


def get_adjacency_matrix_parallel(df, i, row, col_name):
    """
    Function that in a parallel manner clusters the data using 3-nn, creates an adjadency matrix as a result
    :param df: review or summary data
    :param col_name: either summary or review
    :return: matrix
    """
    # initialise the matrix
    mat = [0] * len(df)

    # loop over all summaries per summary to find the cos_sim, no self-loops, not the same reviewer, the 3 highest sim get an edge, if no similarities, no edges
    vec_from = text_to_vector(row[col_name])
    reviewer = row['reviewerID']
    similarities = {}
    for j, col in df.reset_index().iterrows():
        # no self loops and not the same reviewer
        if i is not j and reviewer != col['reviewerID']:
            # calculate the cos_sim per summary pair
            similarities[j] = get_cosine(vec_from, text_to_vector(col[col_name]))
    # sort cos_sim from highest to lowest
    similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
    # take the 3 highest for k-NN, k=3 nearest neighbors
    for val in list(similarities)[:n_neighbor]:
        if similarities[val] != 0:  # if commented, there will always be 3 neighbors, even when sim = 0
            # add edge
            mat[val] = 1

    return mat


def get_all_neighbors(gr):
    """
    Function to get all the neighbors of each node using the graph
    :param gr: graph
    :return: dictionary of neighbors
    """
    d = {}
    for node in gr.nodes():
        d[node] = list(nx.neighbors(gr, node))

    return d


def have_bidi_relationship(G, node1, node2):
    """
    Function to find if two nodes have a bidirectional edge
    :param G: graph
    :param node1:
    :param node2:
    :return: boolean, True or False
    """
    return G.has_edge(node1, node2) and G.has_edge(node2, node1)


def all_nodes_bidi_relation(gr):
    """
    Function to find all nodes with bidirectional relations
    :param gr:
    :return: list of lists with the nodes who are bidirectional connected
    """
    biconnections = set()

    for u, v in gr.edges():
        if u > v:
            v, u = u, v
        if have_bidi_relationship(gr, u, v):
            biconnections.add((u, v))

    print(biconnections)


def remove_edges_one_directional(gr):
    """
    Function that removes all one directional edges.
    :param gr: graph
    :return: updated graph
    """
    gr_copy = gr.copy()

    for a, b in itertools.combinations(gr.nodes(), 2):
        if not have_bidi_relationship(gr, a, b):

            if gr.has_edge(a, b):
                gr_copy.remove_edge(a, b)
            elif gr.has_edge(b, a):
                gr_copy.remove_edge(b, a)

    return gr_copy


def find_all_cores(gr):
    """
    Function that find a core by looking at the bidirectionally connected subgraphs
    :param gr: graph
    :return: list of core nodes
    """
    cores = list()
    repeated = list()

    for l, l2 in itertools.combinations(list(nx.simple_cycles(gr)), 2):
        if l != l2 and len(l) >= n_neighbor and (l not in repeated and l2 not in repeated) and set(l) == set(l2):
            repeated.append(l)
            repeated.append(l2)
            cores.append(l2)

    return cores


def reduce_cores(cores):
    """
    Function that removes duplicate cores and subsets of another core from the cores list
    :param cores: list of cores
    :return: updated list of cores
    """
    cores.sort(key=len)

    keep_core = []

    # remove duplicates
    for c in cores:
        c.sort()
        if c not in keep_core:
            keep_core.append(c)

    temp_core = [lst[:] for lst in keep_core]

    # remove subsets
    for m in temp_core:
        for n in temp_core:
            if m != n and set(m).issubset(set(n)):
                keep_core.remove(m)
                break

    return keep_core


def check_majority(cores, neighbor_dict, break_tie):
    """
    Function that checks if there are nodes that need to be added through majority, and adds them
    :param cores: list of cores
    :param neighbor_dict: dictionary of all the neighbors
    :param break_tie: majority break tie
    :return: updates list of cores
    """
    temp_core = [lst[:] for lst in cores]

    # check if there are nodes of which the majority of neighbors are in the same core
    for idx, val in neighbor_dict.items():
        for id_cc, cc in enumerate(cores):
            if idx not in cc:
                intersec = list(set(val) & set(cc))
                if len(intersec) > break_tie:
                    temp_core[id_cc].append(idx)

    # remove duplicates
    dups = []

    for c in temp_core:
        c.sort()
        if c not in dups:
            dups.append(c)

    return dups


def graph_peeling(core2D, core, gr):
    """
    Function that applies the graph peeling, hence, removes nodes that do not have a directed edge from a core
    :param core2D: 2-core
    :param core: core
    :param gr: graph
    :return: new core
    """
    temp_core = core2D[:]

    for node2D in core2D:
        pred = list(gr.predecessors(node2D))
        if node2D not in core and all(pr not in core for pr in pred):
            temp_core.remove(node2D)

    return temp_core


def old_information_and_gain(candidate_review, word_combinations_ranking, f):
    """
    Function that calculates Equations 18 and 20 (overlap of information and gain in information)
    :param candidate_review
    :param word_combinations_ranking: all word combinations of the current ranking, alphabetically ordered
    :param f: file for writing the results
    :return: boolean whether the review qualifies (satisfying Equations 18 and 20) and the updated ranking dictionary
    """
    qualifies = False
    word_combinations_candidate = []

    # in case the candidate review is only a single word(no combination possible)
    if len(candidate_review.split()) == 1:
        word_combinations_candidate.append([candidate_review.split()[0], candidate_review.split()[0]])
    else:
        for w1, w2 in itertools.combinations(sorted(candidate_review.split()), 2):
            word_combinations_candidate.append([w1, w2])

    # if it concerns the first added ranking (will always be added)
    if word_combinations_ranking == {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [],
                                     '9': [], 'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [], 'g': [], 'h': [],
                                     'i': [], 'j': [], 'k': [], 'l': [], 'm': [], 'n': [], 'o': [], 'p': [], 'q': [],
                                     'r': [], 's': [], 't': [], 'u': [], 'v': [], 'w': [], 'x': [], 'y': [], 'z': []}:
        qualifies = True

    # calculate the information overlap
    intersection_length = combination_intersection(word_combinations_candidate, word_combinations_ranking)

    # calculate the gain
    gain = (len(word_combinations_candidate) - intersection_length) / len(word_combinations_candidate)

    print("\t\t\told information: {}, gain: {}".format(intersection_length, gain), file=f)

    # see whether they are satisfying
    if intersection_length >= old_info and gain >= g:
        qualifies = True

    # if candidate will be added to ranking, update ranking word combinations
    if qualifies:
        for val in word_combinations_candidate:
            key = val[0][0]
            if val not in word_combinations_ranking[key]:
                word_combinations_ranking[key].extend([val])

    return qualifies, word_combinations_ranking


def combination_intersection(candidate, ranking):
    """
    Function that looks at the overlap of word combinations between the candidate and the current ranking
    :param candidate: candidate review
    :param ranking: ranking dictionary
    :return: number of found intersections (overlap size)
    """
    intersection_found = 0

    prev_key = '0'
    found_ind = 0
    for val_can in sorted(candidate):
        new_key = val_can[0][0]
        if prev_key < new_key:
            found_ind = 0
            prev_key = new_key
        count = found_ind
        for val2 in sorted(ranking[new_key])[found_ind:]:
            count += 1
            if val_can == val2:
                found_ind = count
                intersection_found += 1
                break

    return intersection_found


def old_information_and_gain_parallel(candidate_review, word_combinations_ranking, f):
    """
    Function that calculates Equations 18 and 20 (overlap of information and gain in information) in a parallel manner
    :param candidate_review
    :param word_combinations_ranking: all word combinations of the current ranking, alphabetically ordered
    :param f: file for writing the results
    :return: boolean whether the review qualifies (satisfying Equations 18 and 20) and the updated ranking dictionary
    """
    qualifies = False
    word_combinations_candidate = []

    # in case the candidate review is only a single word(no combination possible)
    if len(candidate_review.split()) == 1:
        word_combinations_candidate.append([candidate_review.split()[0], candidate_review.split()[0]])
    else:
        for w1, w2 in itertools.combinations(sorted(candidate_review.split()), 2):
            word_combinations_candidate.append([w1, w2])

    # also make dictionary of the word combinations list for candidate review
    word_combinations_candidate2 = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [],
                                    '9': [], 'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [], 'g': [], 'h': [],
                                    'i': [], 'j': [], 'k': [], 'l': [], 'm': [], 'n': [], 'o': [], 'p': [], 'q': [],
                                    'r': [], 's': [], 't': [], 'u': [], 'v': [], 'w': [], 'x': [], 'y': [], 'z': []}
    for val in word_combinations_candidate:
        word_combinations_candidate2[val[0][0]].extend([val])

    # if it concerns the first added ranking (will always be added)
    if word_combinations_ranking == {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [],
                                     '9': [], 'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [], 'g': [], 'h': [],
                                     'i': [], 'j': [], 'k': [], 'l': [], 'm': [], 'n': [], 'o': [], 'p': [], 'q': [],
                                     'r': [], 's': [], 't': [], 'u': [], 'v': [], 'w': [], 'x': [], 'y': [], 'z': []}:
        qualifies = True

    # calculate the information overlap
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        intersection = [pool.apply(parallel_intersection,
                                   args=(word_combinations_ranking[key], candidate))
                        for key, candidate in word_combinations_candidate2.items() if candidate]

    intersection_length = np.sum(intersection)

    # calculate the gain
    gain = (len(word_combinations_candidate) - intersection_length) / len(word_combinations_candidate)

    print("\t\t\told information: {}, gain: {}".format(intersection_length, gain), file=f)

    # see whether they are satisfying
    if intersection_length >= old_info and gain >= g:
        qualifies = True

    # if candidate will be added to ranking, update ranking word combination
    if qualifies:
        for val in word_combinations_candidate:
            key = val[0][0]
            if val not in word_combinations_ranking[key]:
                word_combinations_ranking[key].extend([val])

    return qualifies, word_combinations_ranking


def parallel_intersection(ranking, candidate):
    """
    Function that looks at the overlap of word combinations between the candidate and the current ranking in a parallel manner
    :param candidate: candidate review
    :param ranking: ranking dictionary
    :return: number of found intersections (overlap size)
    """
    found_ind = 0
    intersection_found = 0

    for val_can in sorted(candidate):

        count = found_ind
        for val2 in sorted(ranking)[found_ind:]:
            count += 1
            if val_can == val2:
                found_ind = count
                intersection_found += 1
                break

    return intersection_found


def prepare_all_products():
    """
    Part of the pre-processing
    :return: pre-processed reviews and meta data
    """
    # load in data
    df_review, df_meta = choose_data(category)
    print(f"Data loaded in for {category}")

    # filter on number of reviews
    df_review_pros = nr_of_votes(df_review, 5)
    del df_review
    print("past step 1/3")

    # remove duplicates
    df_review_without_dubs = remove_duplicates(df_review_pros)
    del df_review_pros
    print("past step 2/3")

    # add total number of reviews to meta data
    df_meta_with_amounts = nr_of_total_votes(df_review_without_dubs, df_meta)
    del df_meta
    print("past step 3/3")

    return df_review_without_dubs, df_meta_with_amounts


def find_correct_product(reviews_df, metas_df, nr_rev, f):
    """
    Function to find the specific product (with specific nr_rev) we are looking for
    :param reviews_df: review data
    :param metas_df: meta data
    :param nr_rev: number of reviews we are looking for
    :param f: file for writing output
    :return: single data frame with review and meta data of a single product
    """
    possible_dfs = metas_df[metas_df['#reviews'] == nr_rev]
    # possible_dfs = metas_df[metas_df.asin == asin_man] # if you want to look for a specific product ID
    possible_dfs = possible_dfs.drop_duplicates(subset=['asin'], keep='first')
    if len(possible_dfs) > 0:
        # take the first product with nr_rev reviews
        asin_id = possible_dfs['asin'].iloc[0]
        print("Found asin_id: {}".format(asin_id), file=f)
        print("Shape: ", reviews_df.loc[reviews_df['asin'] == asin_id].shape, file=f)

        # prepare the data frame for merging
        reviews, product = prepare_df(reviews_df, metas_df, asin_id, f)

        print("Should be true:",
              all(reviews['helpfulness'].rank().sort_values().index == reviews['helpfulness'].sort_values().index))
        reviews['ranking'] = reviews['helpfulness'].rank(method='first', ascending=False)

        print("\n", file=f)

        # merge review and product data, drop unneccesary columns
        if nr_rev % 10 > 0:
            one_product_reviews = pd.merge(reviews.sort_values('ranking')[:-(nr_rev % 10)], product, on='asin',
                                           how='left')
            print(f"Reviews' original length = {len(reviews)} ({nr_rev}), but now it's {len(one_product_reviews)}")
        else:
            one_product_reviews = pd.merge(reviews, product, on='asin', how='left')

        one_product_reviews = one_product_reviews.drop(
            ['unixReviewTime', 'date_x', 'helpfulness', '#reviews', 'date_y'], 1)

        return one_product_reviews
    else:
        print("There is no product with {} reviews".format(nr_rev))
        return pd.DataFrame()


def prepare_one_product(df, f):
    """
    Part of pre-processing after finding product
    :param df: df of product (review + meta)
    :param f: file for output writing
    :return: pre-processed df
    """
    # clean the review and summary texts
    df['clean_reviewText'] = [clean_text(str(text)) for text in df['reviewText']]
    df['clean_summary'] = [clean_text(str(text)) for text in df['summary']]

    # check if there are duplicated reviewerIDs
    print("Duplicate reviewers?: ", df[df.duplicated(subset=['reviewerID'])]['reviewerID'], file=f)

    return df


def cluster_model(df, f, nr_rev):
    """
    The actual model with all three steps.
    :param df: data
    :param f: file for output writing
    :param nr_rev: current number of reviews
    """
    # initialisation of parameters and variables
    start_time = time.time()

    clustered_perc = []
    break_tie = n_neighbor / 2

    # thresholds for whether or not to use parallel methods
    parallel_threshold = 200
    sentence_threshold = 80

    relevance_model = list()

    ranking = pd.DataFrame(columns=['ind', 'review', 'relevance'])
    word_combinations_ranking = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [],
                                 '9': [],
                                 'a': [], 'b': [], 'c': [], 'd': [], 'e': [], 'f': [], 'g': [], 'h': [], 'i': [],
                                 'j': [],
                                 'k': [], 'l': [], 'm': [], 'n': [], 'o': [], 'p': [], 'q': [], 'r': [], 's': [],
                                 't': [],
                                 'u': [], 'v': [], 'w': [], 'x': [], 'y': [], 'z': []}

    # start of the model
    count = 0
    for index in range(len(df)):
        count += 1
        resulting_list = list()
        if count >= n_neighbor:

            # -------- STEP 1 -------- #

            print("\nstep 1", file=f)
            print("index added:", index, file=f)

            # get matrix
            if index >= parallel_threshold:
                with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                    mat = [pool.apply(get_adjacency_matrix_parallel,
                                      args=(df[:index + 1], i, row, 'clean_summary'))
                           for i, row in df[:index + 1].iterrows()]
                mat = np.asmatrix(mat)
            else:
                mat = get_adjacency_matrix(df[:index + 1], 'clean_summary')

            # get graph from matrix
            gr = get_graph_from_matrix(mat)

            # get neighbors from graph
            neighbor_dict = get_all_neighbors(gr)

            # get clusters
            gr2 = remove_edges_one_directional(gr)
            cores = find_all_cores(gr2)
            reduced_cores = reduce_cores(cores)
            final_2Dcore_nodes = check_majority(reduced_cores, neighbor_dict, break_tie)

            resulting_list.extend(
                item for sublist in final_2Dcore_nodes for item in sublist if item not in resulting_list)
            resulting_list.sort()

            # report on clustering percentages
            clustered_perc.append(100 - round(len(resulting_list) * 100 / len(df), 4))
            print(
                "percentage unclustered: {}%".format(
                    100 - round(len(resulting_list) * 100 / len(df), 4)), file=f)

            if len(resulting_list) > 0:

                # -------- STEP 2 -------- #

                print("\n\tstep 2", file=f)
                sentence_df = df.filter(resulting_list, axis=0)

                # get matrix
                if len(resulting_list) >= parallel_threshold:
                    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                        mat_sen = [pool.apply(get_adjacency_matrix_parallel,
                                              args=(sentence_df.reset_index(), i, row, 'clean_reviewText'))
                                   for i, row in sentence_df.reset_index().iterrows()]

                    mat_sen = np.asmatrix(mat_sen)
                else:
                    mat_sen = get_adjacency_matrix(sentence_df.reset_index(), 'clean_reviewText')

                # get graph from matrix
                gr_sen = get_graph_from_matrix(mat_sen, dict(zip(list(range(len(resulting_list))), resulting_list)))

                # get neighbors from either graph or matrix
                neighbor_dict_sen = get_all_neighbors(gr_sen)

                # get clusters
                gr2_sen = remove_edges_one_directional(gr_sen)
                cores_sen = find_all_cores(gr2_sen)
                reduced_cores_sen = reduce_cores(cores_sen)
                final_2Dcore_nodes_sen = check_majority(reduced_cores_sen, neighbor_dict_sen, break_tie)

                for id_c2d, core2D in enumerate(final_2Dcore_nodes_sen):

                    # -------- STEP 3 -------- #

                    core_nodes = reduced_cores_sen[id_c2d]
                    if index in core2D:
                        print("\n\t\tstep 3\n\t\tcandidate index: {} with core nodes: {}".format(index, core_nodes),
                              file=f)
                        # graph peeling
                        peeled_2Dcore_node = graph_peeling(core2D, core_nodes, gr_sen)

                        if index in peeled_2Dcore_node:
                            if index in core_nodes:
                                core_nodes.remove(index)

                            # add core (minus candidate) to relevance model
                            salient_reviews = df.filter(core_nodes, axis=0)
                            # needed for time window
                            candidate_days = df.iloc[index]['elapsedDays']
                            relevance_model.extend([i, review, days] for i, review, days in
                                                   zip(salient_reviews.index,
                                                       salient_reviews['clean_reviewText'],
                                                       salient_reviews['elapsedDays'])
                                                   if [i, review, days] not in relevance_model)

                            print(f"\t\tThe relevance model has size: {len(relevance_model)}", file=f)

                            # remove reviews from relevance model outside of the time window
                            cnt_remove = 0
                            index_remove = []
                            for review in relevance_model:
                                if abs(review[2] - candidate_days) >= int(h):
                                    relevance_model.remove(review)
                                    cnt_remove += 1
                                    index_remove.append(review[0])
                            print(f"\t\t{cnt_remove} reviews removed from the relevance model, indices: {sorted(index_remove)}. \n\t\tIt now has size: {len(relevance_model)}", file=f)

                            candidate_review = df.iloc[index]['clean_reviewText']
                            word_vector = Counter(
                                words for review in relevance_model for words in review[1].split())
                            relevance = get_cosine(text_to_vector(candidate_review), word_vector)

                            # check for information gain and overlap
                            if len(candidate_review.split()) >= sentence_threshold:
                                qualifies, word_combinations_ranking = old_information_and_gain_parallel(
                                    candidate_review,
                                    word_combinations_ranking, f)
                            else:
                                qualifies, word_combinations_ranking = old_information_and_gain(candidate_review,
                                                                                                word_combinations_ranking,
                                                                                                f)

                            # add to ranking
                            if qualifies:
                                ranking = ranking.append(
                                    {'ind': index, 'review': candidate_review, 'relevance': relevance},
                                    ignore_index=True).sort_values('relevance',
                                                                   ascending=False).reset_index(
                                    drop=True)
                                print("\n", ranking, file=f)

                            # add to relevance model
                            relevance_model.append([index, candidate_review, candidate_days])

    # printing some output for results
    print("\n--- %s seconds ---" % (time.time() - start_time), file=f)

    print("\n\n Unclustered percentages: {}, mean: {}".format(clustered_perc, round(np.mean(clustered_perc), 4)),
          file=f)

    final_ranking = pd.merge(ranking, df['ranking'].astype(int), left_on='ind', right_index=True, how='left')
    print("\nFinal Ranking:", file=f)
    final_ranking = final_ranking[['ranking', 'ind', 'relevance', 'review']]
    print(final_ranking, file=f)

    final_ranking.to_csv(f'{path}/{nr_rev}/final_ranking.csv', sep=';', index=True, index_label='Model ranking',
                         header=True)


def start_model():
    """
    Helper function to start the model
    """
    # general preprocessing
    reviews, metas = prepare_all_products()

    for nr_rev in nr_revs:
        path2 = f'{path}/{nr_rev}'
        try:
            os.mkdir(path2)
        except OSError:
            print("Creation of the directory %s failed" % path2)
        else:
            print("Successfully created the directory %s" % path2)

        with open(f'{path2}/results.txt', 'a') as f:
            print("------- Category: {} -------".format(category), file=f)
            # find correct product matching the nr_rev
            one_product_reviews = find_correct_product(reviews, metas, nr_rev, f)

            if len(one_product_reviews) > 0:
                print(
                    "Chosen product: {} has {} reviews - {}, it has {} reviews".format(
                        one_product_reviews['asin'].iloc[0],
                        nr_rev,
                        (nr_rev == len(one_product_reviews)),
                        len(one_product_reviews)), file=f)

                cols = ['ranking', 'overall', 'vote', 'elapsedDays', 'reviewText', 'summary']
                one_product_reviews[cols].sort_values('elapsedDays').reset_index(drop=True).to_csv(
                    f'{path2}/groundtruth.csv', sep=';', index=True, header=True)

                # preprocess the specific product
                final_product = prepare_one_product(one_product_reviews, f)

                # apply the specific product
                cluster_model(final_product.sort_values('elapsedDays').reset_index(drop=True), f, nr_rev)
                del one_product_reviews
                del final_product

        print(f"Done with {nr_rev}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    WORD = re.compile(r"\w+")

    # run via terminal
    h = sys.argv[1]
    category = sys.argv[2]

    # global variables
    n_neighbor = 3
    g = 0.3
    old_info = 1

    # if you want to run everything at once
    # categories = ["Office_Products", "Movies_and_TV", "Electronics", "Home_and_Kitchen", "Sports_and_Outdoors"]
    # hs = [1, 2, 7, 30, 90, 183, 365, 730]

    # for category in categories:
    nr_revs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    if category == "Movies_and_TV":
        nr_revs.extend([141, 150, 160])
    elif category == "Electronics":
        nr_revs.extend([140, 151, 160])
    elif category == "Home_and_Kitchen":
        nr_revs.extend([140, 150, 160])
    elif category == "Office_Products":
        nr_revs.extend([140, 151, 160])
    elif category == "Sports_and_Outdoors":
        nr_revs.extend([143, 150, 162])

        # for h in hs:
    path = f'Results/RQ3b/{h}days/{category}'
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)

    # start model
    start_model()
