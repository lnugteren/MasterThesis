import pandas as pd
import time
import numpy as np
import glob
import re
from datetime import datetime


def load_annotated(path, name):
    """
    Function that loads in the annotated scores gathered by Yang et al. (2015)
    :param path: path to the data
    :param name: whether it's HAS (human annotated score) or HS (helpfulness score)
    :return: the loaded data
    """
    df = pd.read_csv(path, sep=" ", header=None)
    df.columns = ['review_ID', name]
    print("df head: \n", df.head(2))

    return df


def load_reviews(path):
    """
    Function that loads in the reviews that Yang et al. (2015) annotated
    :param path: path to the data
    :return: the loaded data
    """
    all_reviews = glob.glob(path + "/*.review")

    df = pd.concat((pd.read_csv(f, sep="\n", header=None).T for f in all_reviews))
    print("Length reviews df: ", len(df))

    df.columns = ['score', 'summary', 'reviewText']
    df['score'] = df['score'].apply(lambda x: x.replace('Score: ', ''))
    df['summary'] = df['summary'].apply(lambda x: x.replace('Summary: ', ''))

    review_IDs = [re.sub(path + '/', '', i) for i in all_reviews]
    review_IDs = [re.sub('.review', '', i) for i in review_IDs]
    df['review_ID'] = review_IDs

    return df


def load_products(path):
    """
    Function that loads in the products that Yang et al. (2015) annotated
    :param path: path to the data
    :return: the loaded data
    """
    all_products = glob.glob(path + "/*.product")

    df = pd.concat((pd.read_csv(f, sep="\n", header=None).T for f in all_products))
    print("Length products df: ", len(df))

    df.columns = ['category', 'product']
    df['category'] = df['category'].apply(lambda x: x.replace('Category: ', ''))
    df['product'] = df['product'].apply(lambda x: x.replace('Name: ', ''))

    review_IDs = [re.sub(path + '/', '', i) for i in all_products]
    review_IDs = [re.sub('.product', '', i) for i in review_IDs]
    df['review_ID'] = review_IDs

    return df


def parse(path):
    """
    Function that parses the Amazon data to match to the annotated data.
    Found on snap.stanford.edu/data/web-Amazon-links.html
    :param path: path to the data
    """
    f = open(path, 'r')
    entry = {}
    for l in f:
        l = l.strip()
        colon_pos = l.find(':')
        if colon_pos == -1:
            yield entry
            entry = {}
            continue
        e_name = l[:colon_pos]
        rest = l[colon_pos + 2:]
        entry[e_name] = rest
    yield entry
    f.close()


def load_full_reviews(path):
    """
    Function that loads in the Amazon data to match to the annotated data.
    Found on snap.stanford.edu/data/web-Amazon-links.html
    :param path: path to the data
    :return: the loaded data
    """
    li = []
    for e in parse(path):
        li.append(e)

    df = pd.DataFrame(li)

    return df


def get_date(unix_time):
    """
    Function that converts unix_time (as in the Amazon data) to datetime format
    :param unix_time: the unix time
    :return: the datetime format of the unix time
    """
    d = datetime.fromtimestamp(unix_time)
    return d.date()


def clean_full_reviews(df):
    """
    Function that preprocesses the Amazon data and adds columns which can be compared to the HAS to establish a ground truth
    :param df: the Amazon data
    :return: preprocessed and extended Amazon data
    """
    # preprocessing
    df = df.dropna()
    df2 = df.drop(columns=['product/price', 'review/profileName'])
    df2 = df2.rename(
        columns={'product/productId': 'productID', 'product/title': 'product', 'review/userId': 'userID',
                 'review/helpfulness': 'HS', 'review/score': 'score', 'review/time': 'time',
                 'review/summary': 'summary', 'review/text': 'reviewText'})

    df2['vote'] = df2['HS'].apply(lambda x: float(str(x).split('/')[0]))
    df2 = df2.loc[df2['time'] != '-1']
    df2['date'] = df2['time'].apply(lambda x: get_date(float(x)))

    # gather the elapsed days per review, per product (amount of days online since the first review)
    max_days = df2.groupby('productID').date.agg(max)
    df2 = pd.merge(df2, max_days, on='productID', how='left')
    df2['elapsedDays'] = df2['date_y'] - df2['date_x']
    df2 = df2.rename(columns={'date_x': 'date', 'date_y': 'maxDay'})
    df2['elapsedDays'] = df2['elapsedDays'].apply(lambda x: x.days)

    # gather the total amount of votes per product
    total_votes = df2.groupby('productID').vote.agg(sum)
    df2 = pd.merge(df2, total_votes, on='productID', how='left')
    df2 = df2.rename(columns={'vote_x': 'vote', 'vote_y': 'totalVotes'})

    # some statistics
    print(
        "#max_days: {} vs total_votes: {} vs #products: {}".format(df2['maxDay'].nunique(), df2['totalVotes'].nunique(),
                                                                   df2['productID'].nunique()))
    print("min day: {}, max day: {}".format(max_days.min(), max_days.max()))
    print("min votes: {}, max votes: {}".format(total_votes.min(), total_votes.max()))

    # add the different helpfulness score possibilities
    df2['helpfulness_A'] = df2['vote']
    df2['helpfulness_B'] = [y / x if x > 0 else y for x, y in zip(df2['elapsedDays'], df2['vote'])]
    df2['helpfulness_C'] = [y / np.power(x, 2) if x > 0 else y for x, y in zip(df2['elapsedDays'], df2['vote'])]
    df2['helpfulness_D'] = [y / np.sqrt(x) if x > 0 else y for x, y in zip(df2['elapsedDays'], df2['vote'])]
    df2['helpfulness_E'] = [y / x if x > 0 else y for x, y in zip(df2['totalVotes'], df2['vote'])]

    return df2


def load_matching_data(path):
    """
    Function that loads in the data with matches between Amazon reviews and the annotated data
    :param path: path to the data
    :return: the loaded in data
    """
    df = pd.read_csv(path, sep=";")
    print("Size of matching data df: ", df.shape)

    return df.set_index('index')


def is_df_sorted(df, colname):
    """
    Function that checks whether a column in a df in sorted
    :param df: the df you want to analyse
    :param colname: the column you want to analyse
    :return: True or False
    """
    return (np.diff(df[colname]) > 0).all()


def check_ground_truth(df, nr_rev):
    """
    Function that checks for each created helpfulness score possibility whether the sorted aligns with the sorting of
    HAS. It prints the number of times each one works and how many checks have been conducted
    :param df: the df you want to analyse, here the matched Amazon with annotated data
    :param nr_rev: the number of reviews you want a comparison between. For the research >1, >2 and >3 was used.
    """
    grouped = df.groupby('productID')

    # initialise the individual counts
    count_A = 0
    count_B = 0
    count_C = 0
    count_D = 0
    count_E = 0

    total_tests = 0

    # Go over all reviews per product
    for product, review in grouped:
        print("Product: ", product)
        if len(review) > nr_rev:
            product_sub = pd.DataFrame(review)

            # check which score works and which doesn't
            A_works = is_df_sorted(product_sub, 'helpfulness_A')
            B_works = is_df_sorted(product_sub, 'helpfulness_B')
            C_works = is_df_sorted(product_sub, 'helpfulness_C')
            D_works = is_df_sorted(product_sub, 'helpfulness_D')
            E_works = is_df_sorted(product_sub, 'helpfulness_E')

            # count the amount of times certain scores work
            if A_works:
                print("Helpfulness_A: ", A_works)
                count_A += 1
            if B_works:
                print("Helpfulness_B: ", B_works)
                count_B += 1
            if C_works:
                print("Helpfulness_C: ", C_works)
                count_C += 1
            if D_works:
                print("Helpfulness_D: ", D_works)
                count_D += 1
            if E_works:
                print("Helpfulness_E: ", E_works)
                count_E += 1

            total_tests += 1

    # print the results for documentation purposes
    print("count_A: {} count_B: {} count_C: {} count_D: {} count_E: {}".format(count_A, count_B, count_C, count_D,
                                                                               count_E))
    print("total tests: ", total_tests)


def check_ground_truth_euclidean(df, nr_rev):
    """
    Function that checks for each created helpfulness score what the euclidean distance is with the sorting of
    HAS. It prints the average euclidean distance
    :param df: the df you want to analyse, here the matched Amazon with annotated data
    :param nr_rev: the number of reviews you want a comparison between. For the research >1, >2 and >3 was used.
    """
    grouped = df.groupby('productID')
    # initialise the individual averages
    avg_A = []
    avg_B = []
    avg_C = []
    avg_D = []
    avg_E = []
    baseline = []

    total_tests = 0

    # Go over all reviews per product
    for product, review in grouped:
        if len(review) == nr_rev:
            product_sub = pd.DataFrame(review)

            # calculate the euclidean distances for each ranking
            avg_A.append(np.linalg.norm(np.array(product_sub['HAS'].rank(method='first')) - np.array(
                product_sub['helpfulness_A'].rank(method='first'))))
            avg_B.append(np.linalg.norm(np.array(product_sub['HAS'].rank(method='first')) - np.array(
                product_sub['helpfulness_B'].rank(method='first'))))
            avg_C.append(np.linalg.norm(np.array(product_sub['HAS'].rank(method='first')) - np.array(
                product_sub['helpfulness_C'].rank(method='first'))))
            avg_D.append(np.linalg.norm(np.array(product_sub['HAS'].rank(method='first')) - np.array(
                product_sub['helpfulness_D'].rank(method='first'))))
            avg_E.append(np.linalg.norm(np.array(product_sub['HAS'].rank(method='first')) - np.array(
                product_sub['helpfulness_E'].rank(method='first'))))
            baseline.append(np.linalg.norm(np.array([1, 2, 3]) - np.array([3, 2, 1])))

            total_tests += 1

    # print the results for documentation purposes
    print("\n{}\n{}\n{}\n{}\n{}\n{}\n".format(avg_A, avg_B, avg_C, avg_D, avg_E, baseline))
    print("avg_A: {} avg_B: {} avg_C: {} avg_D: {} avg_E: {}".format(round(np.mean(avg_A), 3), round(np.mean(avg_B), 3),
                                                                     round(np.mean(avg_C), 3), round(np.mean(avg_D), 3),
                                                                     round(np.mean(avg_E), 3)))
    print("baseline average:", round(np.mean(baseline), 3))


if __name__ == '__main__':

    start_time = time.time()

    # Load in all the ground truth data, change paths for different categories
    df_HAS = load_annotated('ground_truth/human_annotations/electronics.human.score', 'HAS')
    df_HS = load_annotated('ground_truth/human_annotations/electronics.xofy.rate', 'HS')

    df_reviews = load_reviews('ground_truth/reviews/electronics')
    df_products = load_products('ground_truth/reviews/electronics')

    df_data = pd.merge(df_reviews, df_products, on='review_ID', how='left')
    df_data2 = pd.merge(df_data, df_HAS, on='review_ID', how='left')
    df_data3 = pd.merge(df_data2, df_HS, on='review_ID', how='left')

    print("Length of data df after merges: ", len(df_data3))

    df_full_reviews = load_full_reviews('ground_truth/Electronics.txt')
    df_full_reviews = clean_full_reviews(df_full_reviews)

    # Find the reviews of products actually annotated
    temp_products = df_data3['product']
    print("Amount of the same products: ", temp_products.duplicated().sum())
    temp_products = temp_products.drop_duplicates(keep='first')
    print("Amount of unique products: ", df_full_reviews['product'].nunique())
    df_full_reviews = df_full_reviews[df_full_reviews['product'].isin(temp_products)]

    # to csv to hand-match all the annotated instances
    # df_data3.to_csv('ground_truth/df_data3.csv', sep=";")
    # df_full_reviews.to_csv('ground_truth/df_full_reviews.csv', sep=";")

    # prepare the score and vote data for better matching and calculations
    df_data3['score'] = df_data3['score'].apply(lambda x: str(x).split('/')[0])
    df_data3['vote'] = df_data3['HS'].apply(lambda x: float(str(x).split('/')[0]))

    # Load in the matching data (manually created)
    df_match = load_matching_data('ground_truth/match_dfs/electronics_match.csv')

    # Merge the loaded data with the matching csv (hand-matched)
    df_match_review = pd.merge(df_match, df_full_reviews, how='left', left_index=True, right_index=True)
    # only needed for home and outdoors
    # df_match_review = df_match_review.drop(['product_y', 'HS_y', 'summary_y', 'reviewText_y', 'userID_y'], 1)
    # df_match_review = df_match_review.rename(
    #     columns={'product_x': 'product', 'HS_x': 'HS', 'summary_x': 'summary', 'reviewText_x': 'review',
    #              'userID_x': 'userID'})
    df_match_review = pd.merge(df_match_review, df_data3, on=['product', 'review_ID', 'HS', 'score', 'vote'],
                               how='left')
    df_match_review = df_match_review[
        ['review_ID', 'productID', 'HS', 'HAS', 'score', 'vote', 'helpfulness_A', 'helpfulness_B', 'helpfulness_C',
         'helpfulness_D', 'helpfulness_E']]

    df_match_review = df_match_review.sort_values(['productID', 'HAS'])
    print(df_match_review.head(10))

    # Check the validation of the ground truth
    check_ground_truth(df_match_review, 1)
    check_ground_truth_euclidean(df_match_review, 2)

    print("--- %s seconds ---" % (time.time() - start_time))