def feature_groups():
    numeric = ["Released_Year", "Meta_score", "No_of_Votes", "Runtime_min", "Gross_usd"]
    categorical = ["Certificate", "Genre", "Director", "Star1", "Star2", "Star3", "Star4"]
    text = "Overview"
    target = "IMDB_Rating"
    return numeric, categorical, text, target

def intersect_existing(df, cols):
    return [c for c in cols if c in df.columns]