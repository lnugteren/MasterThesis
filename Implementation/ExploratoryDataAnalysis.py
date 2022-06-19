# Import libraries
import orjson as json
import numpy as np
import pandas as pd
import time
import gzip
import os
from matplotlib import pyplot as plt
import glob


def load_data_gz(path):
    """
    Function that loads in the data from a .gz file and transfers into a Pandas data frame.
    :param path
    :return: data frame
    """
    data = []
    with gzip.open(path, 'r') as f:
        for l in f:
            data.append(json.loads(l.strip()))

    df_data = pd.DataFrame.from_dict(data)

    return df_data


def load_data_2(path):
    """
    Function that loads in the data from a .json file and transfers into a Pandas data frame.
    :param path
    :return: data frame
    """
    data = []
    with open(path, 'r') as f:
        for l in f:
            data.append(json.loads(l.strip()))

    df_data = pd.DataFrame.from_dict(data)

    return df_data


def load_data(path):
    """
    Functio that loads in the data from a .json, but also the specific columsn needed
    :param path:
    :return:
    """
    data = []
    with open(path, 'r') as f:
        for l in f:
            val = json.loads(l.strip())
            if "vote" not in val:
                val["vote"] = np.nan
            if "reviewText" not in val:
                val["reviewText"] = np.nan
            if "summary" not in val:
                val["summary"] = np.nan
            filtered = {
                "overall": val["overall"],
                "vote": val["vote"],
                "asin": val["asin"],
                "reviewText": val["reviewText"],
                "summary": val["summary"],
                "reviewerID": val["reviewerID"]
            }
            data.append(filtered)

    df_data = pd.DataFrame(data)

    return df_data


def choose_data(data, zipped = True):
    """
    Choosing the data youo want to load in, either .json or .gz
    :param data: category you want to load in
    :param zipped: json or gz
    :return: review data
    """
    if zipped:
        reviewdata = load_data_gz(f'Data/{data}.json.gz')
    else:
        reviewdata = load_data(f'Data/{data}.json')

    return reviewdata


def nr_of_votes(df, vote_def):
    """
    Function that filters on the minimum number of votes (inclusion criterium) and reports on some other statistics
    :param df
    :param vote_def: minimum number of votes
    :return: filtered df
    """
    df2 = df.fillna('0')

    print("total reviews: {}".format(len(df2)), file=f)
    df2['vote'] = df2['vote'].astype(str)
    df2['vote'] = df2['vote'].apply(lambda x: x.replace(',',''))

    df2['vote'] = df2['vote'].astype(float)

    print("mean: {}".format(df2['vote'].mean()), file=f)

    df3 = df2.loc[df2['vote'] != 0]
    df4 = df2.loc[df2['vote'] >= vote_def] # votes higher than 0, or 1 for instance
    print("amount with votes: {}".format(len(df3)), file=f)
    print("amount of votes higher than {}: {}".format(vote_def,len(df4)), file=f)
    print("mean of votes >0: {}".format(df3['vote'].mean()), file=f)
    print("mean of votes higher than {}: {}".format(vote_def, df4['vote'].mean()), file=f)

    return df2


def remove_duplicates(df):
    """
    Function that removed the duplicated reviews.
    :param df
    :return: df without duplicates.
    """
    df = df.drop_duplicates(subset=['vote', 'reviewText', 'summary'], keep='first')
    print("total of non-duplicate reviews: ", len(df), file=f)

    return df


def nr_of_total_votes(df):
    """
    Calculates and add the total number of votes per product for calculating the GT.
    :param df
    :return: merged df including number of total votes.
    """
    review_amounts = df.groupby(['asin']).size().reset_index(name='#reviews')
    review_amounts_df = pd.merge(df, review_amounts, on='asin', how='left')

    return review_amounts_df


def find_product(df, nr_rev):
    """
    Find all product IDs with above nr_rev reviews in total
    :param df: all products
    :param nr_rev: minimum number of reviews
    :return: df with only the product IDs and number of reviews, sorted
    """
    new_df = df[df['#reviews'] >= nr_rev]
    return new_df[['asin', '#reviews']].drop_duplicates().sort_values('#reviews')


def data_description_exploration(data, category):
    """
    Function to explore the Amazon review data
    :param data
    :param category
    """
    # printing some statistics
    print("\n\n", file=f)
    print("Number of unique customers: ", data['reviewerID'].nunique(), file=f)
    print("Number of unique products: ", data['asin'].nunique(), file=f)
    print("Number of reviews in total: ", len(data), file=f)
    print("Average #reviews per customer: ", (len(data) / data['reviewerID'].nunique()), file=f)
    print("Average #reviews per product: ", (len(data) / data['asin'].nunique()), file=f)
    print("Number of total review words:", np.sum([len(words.split()) for words in data['reviewText']]), file=f)
    print("Number of total summary words:", np.sum([len(words.split()) for words in data['summary']]), file=f)

    print("\nDescribe overall:\n{}".format(data['overall'].describe()), file=f)
    print("\nDescribe vote:\n{}".format(data['vote'].describe()), file=f)

    # plot distribution of product ratings
    number_of_ratings = data['overall'].value_counts().sort_index(ascending=False)

    numbers = number_of_ratings.index
    quantity = number_of_ratings.values

    plt.figure(figsize=(10, 8))
    plt.pie(quantity, labels=numbers, autopct='%1.1f%%', startangle=90)  # colors=custom_colors)
    central_circle = plt.Circle((0, 0), 0.5, color="white")
    fig = plt.gcf()
    fig.gca().add_artist(central_circle)
    plt.rc('font', size=12)
    plt.title(f"Distribution of Product Ratings for {category}", fontsize=20)
    plt.savefig(f'Results/EDA/{category}/distribution_overall.pdf')

    # plot distribution of votes
    plt.figure(figsize=(12, 8))
    cutoff = np.array([0, 1, 5, 20, 50, 100, 250, 500, 1000, 10000])

    dist_help = data.groupby([pd.cut(data['vote'], bins=cutoff, right=False)]).size()
    print(dist_help, file=f)
    plt.xlabel('Vote distribution')
    plt.ylabel('Total count')
    plt.title(f"Distributions of Votes for {category}", fontsize=20)
    dist_help.plot(kind='bar')
    plt.savefig(f'Results/EDA/{category}/distribution_votes.pdf')

    print("\n\n", file=f)


def too_large_to_choose(path, chunk_size, dat, cols):
    """
    Function that converts the data into smaller csv chunks to reduce the runtime and space of loading in the data.
    :param path
    :param chunk_size: size of each chunk
    :param dat: review or _meta data
    :param cols: which columns should be considered
    :return: data frame of the data
    """
    columns = cols.split(",")

    with open(path, 'r') as f:
        chunk = 0
        line = next(f)
        while line:
            i = 0
            print(f"doing chunk{chunk}")
            with open(f"Data/csvs{dat}/file{chunk}.csv", 'w') as output_f:
                while line:
                    val = json.loads(line.strip())
                    if "vote" not in val:
                        val["vote"] = np.nan
                    if "reviewText" not in val:
                        val["reviewText"] = np.nan
                    if "summary" not in val:
                        val["summary"] = np.nan

                    output_f.write("\t".join(['"'+str(val[key]).replace("\n","\\n")+'"' for key in columns])+"\n")
                    line = next(f)
                    i += 1
                    if i > chunk_size:
                        chunk += 1
                        break

    all_files = sorted(glob.glob(f'Data/csvs{dat}/*.csv'))
    size = round(len(all_files)/4)
    print(size, size*2, size*3)
    li = []

    l = 0
    for filename in all_files:
        if l in range(0, size):
            print(f"now doing {filename}")
            df = pd.read_csv(filename, sep='\t')
            li.append(df)
            l += 1

    final_df = pd.concat(li, axis=0, ignore_index=True)

    return final_df


def explore(category, f):
    """
    Helper function to start the exploration
    :param category:
    :param f: file for writing output
    """
    start_time = time.time()
    print("------- Category: {} -------".format(category), file=f)

    reviewdata = choose_data(category, False)
    df_review = reviewdata[['overall', 'vote', 'asin', 'reviewText', 'summary', 'reviewerID']]

    # if file is too large
    # df_review = too_large_to_choose(f'Data/{category}.json', 1000000, "rev", "overall,vote,asin,reviewText,summary,reviewerID")

    # filter on the minimum number of votes
    df_review_pros = nr_of_votes(df_review, 5)
    del df_review

    # removing duplicates
    df_review_without_dubs = remove_duplicates(df_review_pros)
    del df_review_pros

    # do the exploration (statistics and plots)
    data_description_exploration(df_review_without_dubs, category)

    # find products with specific numbers of reviews
    df_review_without_dubs = df_review_without_dubs.loc[df_review_without_dubs['vote'] > 5]
    df_final = nr_of_total_votes(df_review_without_dubs)
    del df_review_without_dubs

    print("\n", file=f)

    print(find_product(df_final, 10).to_string(), file=f)
    del df_final

    print("--- %s seconds --- \n\n" % (time.time() - start_time), file=f)


if __name__ == '__main__':

    # category = sys.argv[1] # to use terminal, remove the for loop
    categories = ['Appliances', 'Arts_Crafts_and_Sewing', 'Automotive', 'Beauty', 'Books', 'CDs_and_Vinyl',
                  'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Digital_Music', 'Electronics',
                  'Fashion', 'Gift_Cards', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen', 'Industrial_and_Scientific',
                  'Kindle_Store', 'Luxury_Beauty', 'Magazine_Subscriptions', 'Movies_and_TV', 'Musical_Instruments',
                  'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies', 'Prime_Pantry', 'Software',
                  'Sports_and_Outdoors', 'Tools_and_Home_Improvement', 'Toys_and_Games', 'Video_Games']

    for category in categories:
        path = f'Results/EDA/{category}'
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s" % path)

        with open(f'{path}/descriptions.txt', 'a') as f:
            explore(category, f)
