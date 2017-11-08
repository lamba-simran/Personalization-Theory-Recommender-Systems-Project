import pandas as pd
import random

df = pd.read_csv('data/sampled_data.csv')
user_id = set(df['User'].values)
batch1 = random.sample(user_id, 700)
batch2 = random.sample(user_id, 1400)
batch3 = random.sample(user_id, 2100)


size700 = df[df['User'].isin(batch1)]
size1400 = df[df['User'].isin(batch2)]
size2100 = df[df['User'].isin(batch3)]

print(len(set(size700['User'].values)))
print(len(set(size1400['User'].values)))
print(len(set(size2100['User'].values)))

size700.to_csv("data/sample_700.csv")
size1400.to_csv("data/sample_1400.csv")
size2100.to_csv("data/sample_2100.csv")