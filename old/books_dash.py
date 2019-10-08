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
