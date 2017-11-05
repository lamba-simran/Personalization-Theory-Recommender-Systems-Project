import gzip
import dateutil
import math
import matplotlib.pyplot as plt
import pandas as pd


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def assignment(item, wanted_categories):
    counter = 0
    for category in wanted_categories:
        if category in item:
            return wanted_categories[counter]
        else:
            counter += 1
    return ''


def assignment2(item, wanted_categories):

    counter = 0
    category = ''
    for category2 in wanted_categories:
        if category2 in item:
            category = category + " " + category2
            counter += 1
        else:
            counter += 1
    return category.strip()


def pick_categories(row, wanted_categories):
    each_game_category = row['categories']
    category = ''
    if len(each_game_category) > 1:
        for i in range(len(each_game_category)):
            each_game = each_game_category[i]
            category = category + " " + each_game[1]
        return assignment2(category, wanted_categories)
    else:
        if len(each_game_category[0]) == 1:
            return ''
        elif len(each_game_category[0]) > 3:
            return assignment(each_game_category[0][2], wanted_categories)
        else:
            return assignment(each_game_category[0][1], wanted_categories)


def mult_cat(row, wanted_categories):
    each_game_category = row['cat']
    counter = 0
    for category2 in wanted_categories:
        if category2 in each_game_category:
            counter += 1
    if counter >= 2:
        return "multi_support"
    else:
        return each_game_category


def define_categories(meta, wanted_categories):
    different_cat = meta.apply(lambda row: pick_categories(row), axis=1)
    meta['cat'] = different_cat
    meta2 = meta[meta['cat'] != '']
    different_cat = meta2.apply(lambda row: mult_cat(row, wanted_categories), axis=1)
    meta['cat2'] = different_cat

    return meta


def get_clean_data(meta):
    cat_table = pd.crosstab(index=meta['cat2'], columns="count")
    print(cat_table.sort_values(by='count', ascending=0))
    meta_clean = meta[meta['cat2'].notnull()]
    meta_clean = meta_clean[['asin', 'cat2']]
    meta_clean.columns = ['asin', 'category']

    return meta_clean


def merge_review_meta(meta_clean, reviews):
    merged_df = pd.merge(meta_clean, reviews, on='asin', how='inner')
    merged_df['date'] = [dateutil.parser.parse(x) for x in merged_df['reviewTime']]
    merged_df = merged_df.drop(merged_df.columns[[1, 4, 5, 6, 8, 9]], 1)
    relevant_data = merged_df[merged_df.date > '2011-01-01']
    relevant_data.to_csv("clean_data.csv")
    distinct_user = relevant_data.drop_duplicates(['asin'])
    rows = distinct_user.shape[0]
    cat_prop = pd.crosstab(index=distinct_user['category'], columns="count")
    cat_prop.hist('count')
    plt.show()
    print(cat_prop['count'].div(rows).multiply(100))

    return relevant_data, distinct_user


def random_item_sampling(sample, relevant_data, wanted_size_by_category):
    for category in wanted_size_by_category:
        size = wanted_size_by_category[category]
        each_batch_size = int(size/4)
        last_batch_size = size - each_batch_size * 3
        cat = relevant_data[relevant_data['category'] == category]
        num_table = pd.crosstab(index=cat['asin'], columns="count")
        num_table.plot.bar()
        plt.show()
        quantile = list(num_table.quantile([.7, .85, .9])['count'])
        print("item ratings break down :", category, quantile)
        sample_id = num_table[num_table['count'] <= quantile[0]].sample(n=each_batch_size).index.tolist()
        sample_id += num_table[(num_table['count'] > quantile[0]) &
                               (num_table['count'] <= quantile[1])].sample(n=each_batch_size).index.tolist()
        sample_id += num_table[(num_table['count'] > quantile[1]) &
                               (num_table['count'] <= quantile[2])].sample(n=each_batch_size).index.tolist()
        sample_id += num_table[num_table['count'] > quantile[2]].sample(n=last_batch_size).index.tolist()
        sample[category] = sample_id

    sampled_items = []
    for category in sample:
        sampled_items += sample[category]

    return sampled_items


def random_user_sampling(relevant_data):
    each_batch_size = 5000
    num_table = pd.crosstab(index=relevant_data['reviewerID'], columns="count")

    quantile = list(num_table.quantile([.9, .95, .985])['count'])
    print("user ratings break down : ", quantile)
    distribution = num_table['count'].value_counts()

    distribution.iloc[[7]] = num_table[num_table['count'] > 7].sum().values[0]
    distribution = distribution.iloc[0:8]
    distribution.plot.bar()

    plt.show()
    sample_id = num_table[num_table['count'] <= quantile[0]].sample(n=each_batch_size).index.tolist()
    sample_id += num_table[(num_table['count'] > quantile[0]) &
                           (num_table['count'] <= quantile[1])].sample(n=each_batch_size).index.tolist()
    sample_id += num_table[(num_table['count'] > quantile[1]) &
                           (num_table['count'] <= quantile[2])].sample(n=each_batch_size).index.tolist()
    sample_id += num_table[num_table['count'] > quantile[2]].sample(n=each_batch_size).index.tolist()

    return sample_id


def main():
    # reviews = getDF('/Users/taeyoungchoi/Documents/Fall 17/Personalization Theory/reviews_Video_Games.json.gz')
    # meta = getDF('/Users/taeyoungchoi/Documents/Fall 17/Personalization Theory/meta_Video_Games.json.gz')
    wanted_categories = ['PC', 'Mac', 'More Systems', 'Nintendo', 'PlayStation', 'multi_support', 'Sony', 'Wii', 'Xbox']

    # meta = define_categories(meta, wanted_categories)
    # meta_clean = get_clean_data(meta)
    # relevant_data, distinct_user = merge_review_meta(meta_clean, reviews)

    relevant_data = pd.read_csv('/Users/taeyoungchoi/Documents/Fall 17/Personalization Theory/clean_data.csv')
    random_user_id = random_user_sampling(relevant_data)
    items_with_sampled_user = relevant_data[relevant_data['reviewerID'].isin(random_user_id)]
    distinct_user = items_with_sampled_user.drop_duplicates(['asin'])
    print("---------------------------------------------\n")
    rows = distinct_user.shape[0]
    cat_prop = pd.crosstab(index=distinct_user['category'], columns="count")

    # cat_prop.plot.bar()
    # plt.show()
    wanted_size_by_category = cat_prop['count'].div(rows).multiply(100).to_dict()
    for item in wanted_categories:
        wanted_size_by_category[item] = math.ceil(wanted_size_by_category[item])

    print(wanted_size_by_category)
    print("dropping insignificant categories: Mac, Sony")
    print("---------------------------------------------\n")
    wanted_size_by_category.pop('Mac', None)
    wanted_size_by_category.pop('Sony', None)
    print(wanted_size_by_category)
    print("the total sample size is : ", sum(wanted_size_by_category.values()))
    print("---------------------------------------------\n")

    sample = {}
    sampled_items = random_item_sampling(sample, items_with_sampled_user, wanted_size_by_category)

    hundred_items = items_with_sampled_user[items_with_sampled_user['asin'].isin(sampled_items)].drop(['Unnamed: 0', 'date'], 1)
    print(len(set(hundred_items.asin)), len(set(hundred_items.reviewerID)))
    hundred_items.to_csv("sampled.csv")
    '''
    '''


if __name__ == "__main__":
    main()