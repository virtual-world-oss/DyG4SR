
# import pandas as pd
# import os
# ratings = []
# with open("data/beauty/ratings.dat") as f:
#     for l in f:
#         user_id, item_id, rating, timestamp = [_ for _ in l.split('::')]
#         rating = float(rating)
#         timestamp = int(timestamp)
#         ratings.append({
#                 'user_id': user_id,
#                 'item_id': item_id,
#                 'rating': rating,
#                 'timestamp': timestamp,
#                 })
# ratings = pd.DataFrame(ratings)
# print(ratings.shape)

from info_nce import InfoNCE