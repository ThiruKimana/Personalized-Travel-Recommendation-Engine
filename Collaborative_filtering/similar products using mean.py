import pickle
import pandas as pd

# Load prediction rules from data files
file_path = r"C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\data\\mean_predicted_destination_ratings.dat"
means = pickle.load(open(file_path, "rb"))

# Load the destination types
destinations_df = pd.read_csv(r"C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\destinations.csv", index_col="Destination_id")

# Assuming 'means' is a Series or a single-column DataFrame with ratings
# If 'means' is a DataFrame, extract the relevant column for ratings
if isinstance(means, pd.DataFrame):
    user_ratings = means.iloc[:, 0]  # Use the first column if it's a DataFrame
else:
    user_ratings = means  # If it's already a Series or single column

print("The destinations we will recommend:")

# Assign user ratings to the DataFrame
destinations_df['rating'] = user_ratings

# Sort destinations by rating
destinations_df = destinations_df.sort_values(by=['rating'], ascending=False)

# Print the top 5 recommended destinations
print(destinations_df[['Destination name', 'Destination type', 'rating']].head(5))