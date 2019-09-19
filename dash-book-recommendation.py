import dash
import dash_html_components as html
import dash_core_components as dcc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']

ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

books.drop(["imageUrlS", "imageUrlM", "imageUrlL"], axis=1, inplace=True)
books["yearOfPublication"].astype("str", inplace=True)
books.drop(209538, axis=0, inplace=True)
books.drop(221678, axis=0, inplace=True)
books.drop(220731, axis=0, inplace=True)
books["publisher"].fillna("other", inplace=True)

for ages in users["Age"]:
    if ages > 90 or ages < 5:
        users["Age"].replace(ages, np.nan)

users["Age"].fillna(users["Age"].mean(), inplace=True)
users["Age"] = users["Age"].astype("int32")
users.index = users["userID"]
users.drop("userID", axis=1, inplace=True)

combined_rating_books = pd.merge(ratings, books, on="ISBN")
combined_rating_books.drop(["bookAuthor", "yearOfPublication", "publisher"], axis=1, inplace=True)

totalRatings = combined_rating_books.groupby(by=["bookTitle"])["bookRating"].count().reset_index()
combined_totalRating = pd.merge(totalRatings, combined_rating_books, left_on = "bookTitle", right_on="bookTitle")
less_than_10 = combined_totalRating["bookRating_x"] > 20
combined_totalRating["bish"] = less_than_10
combined_totalRating = combined_totalRating[less_than_10]
combined_totalRating_user = pd.merge(combined_totalRating, users, left_on = "userID", right_on="userID", how="left")

combined_onlyUS = combined_totalRating_user[combined_totalRating_user["Location"].str.contains("usa")]
combined_pivottable = combined_onlyUS.pivot(index="ISBN", columns = "userID", values="bookRating_y").fillna(0)
combined_matrix = csr_matrix(combined_pivottable.values)

model = NearestNeighbors(metric="cosine", algorithm="brute")
model.fit(combined_matrix)

query_index = np.random.choice(combined_pivottable.shape[0])
dist, ind = model.kneighbors(combined_pivottable.iloc[query_index, :].values.reshape(1,-1), n_neighbors = 6)

for i in range(len(dist.flatten())):
    if i == 0:
        print("Recommended for {0} \n".format(combined_pivottable.index[query_index]))
    else:
        print("{0}: {1}", format(str(i), combined_onlyUS.index[ind.flatten()[i]]) )

##############
##############
##############
## DASH APP ##
##############
##############


app = dash.Dash()
app.layout = html.Div(
    html.Div([
        html.H4('Book Recommendation'),
        html.Div(id='live-update-text'),
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000, # in milliseconds
            n_intervals=0
        )
    ])

)

@app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Book Recommendations for {}'.format(n), style=style)
    ]

@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):

    # Create the graph with subplots


    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)


    @app.callback(Output('live-update-text', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_metrics(n):
    lon, lat, alt = satellite.get_lonlatalt(datetime.datetime.now())
    style = {'padding': '5px', 'fontSize': '16px'}
    return [
        html.Span('Longitude: {0:.2f}'.format(lon), style=style),
        html.Span('Latitude: {0:.2f}'.format(lat), style=style),
        html.Span('Altitude: {0:0.2f}'.format(alt), style=style)
    ]

import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import re
import seaborn as sns


#Loading data
books = pd.read_csv('books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True)

books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"

books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN

books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)

books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'

users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan

users.Age = users.Age.fillna(users.Age.mean())

users.Age = users.Age.astype(np.int32)

n_users = users.shape[0]
n_books = books.shape[0]

ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]

ratings = ratings[ratings.userID.isin(users.userID)]

sparsity=1.0-len(ratings_new)/float(n_users*n_books)

ratings_explicit = ratings_new[ratings_new.bookRating != 0]
ratings_implicit = ratings_new[ratings_new.bookRating == 0]

ratings_count = pd.DataFrame(ratings_explicit.groupby(['ISBN'])['bookRating'].sum())
top10 = ratings_count.sort_values('bookRating', ascending = False).head(10)
top10.merge(books, left_index = True, right_on = 'ISBN')

users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]

counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]

ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns

n_users = ratings_matrix.shape[0]
n_books = ratings_matrix.shape[1]

ratings_matrix.fillna(0, inplace = True)
ratings_matrix = ratings_matrix.astype(np.int32)

sparsity=1.0-len(ratings_explicit)/float(users_exp_ratings.shape[0]*n_books)

def findksimilarusers(user_id, ratings, metric = metric, k=k):
    similarities=[]
    indices=[]
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)
    loc = ratings.index.get_loc(user_id)
    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()

    return similarities,indices

def predict_userbased(user_id, item_id, ratings, metric = metric, k=k):
    prediction=0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilarusers(user_id, ratings,metric, k) #similar users based on cosine similarity
    mean_rating = ratings.iloc[user_loc,:].mean() #to adjust for zero based indexing
    sum_wt = np.sum(similarities)-1
    product=1
    wtd_sum = 0

    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == user_loc:
            continue;
        else:
            ratings_diff = ratings.iloc[indices.flatten()[i],item_loc]-np.mean(ratings.iloc[indices.flatten()[i],:])
            product = ratings_diff * (similarities[i])
            wtd_sum = wtd_sum + product

    if prediction <= 0:
        prediction = 1
    elif prediction >10:
        prediction = 10

    prediction = int(round(mean_rating + (wtd_sum/sum_wt)))
    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))

    return prediction

predict_userbased(11676,'0001056107',ratings_matrix)

def findksimilaritems(item_id, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]
    ratings=ratings.T
    loc = ratings.index.get_loc(item_id)
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[loc, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()

    return similarities,indices

similarities,indices=findksimilaritems('0001056107',ratings_matrix)

def predict_itembased(user_id, item_id, ratings, metric = metric, k=k):
    prediction= wtd_sum =0
    user_loc = ratings.index.get_loc(user_id)
    item_loc = ratings.columns.get_loc(item_id)
    similarities, indices=findksimilaritems(item_id, ratings) #similar users based on correlation coefficients
    sum_wt = np.sum(similarities)-1
    product=1
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i] == item_loc:
            continue;
        else:
            product = ratings.iloc[user_loc,indices.flatten()[i]] * (similarities[i])
            wtd_sum = wtd_sum + product
    prediction = int(round(wtd_sum/sum_wt))

    if prediction <= 0:
        prediction = 1
    elif prediction >10:
        prediction = 10

    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id,item_id,prediction))      
    return prediction
