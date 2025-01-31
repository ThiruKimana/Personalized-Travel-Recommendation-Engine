import numpy as np
import pandas as pd
import pickle
import sys
import os

# Add the directory to the Python path
sys.path.append(os.path.abspath("C:\\Users\\Windows 10 Pro\\Movie-Recommendation-System"))

import matrix_factorization_utilities
from matrix_factorization_utilities import low_rank_matrix_factorization


 # Load user ratings
raw_dataset_df = pd.read_csv("C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\Traveller-Destinations_Ratings_training dataset.csv")

# Ensure we have the correct numerical ratings column
rating_column = 'Rating/Review'  # Updated column name

# Remove filtering to include all travellers
filtered_df = raw_dataset_df

# Create pivot table using the correct rating column
ratings_df = pd.pivot_table(
    filtered_df,
    values=rating_column,  # Use the correct rating column
    index="Traveller_id",
    columns="Destination_id",
    aggfunc="max"
)

# Normalize the ratings (center them around their mean)

normalized_ratings, means = matrix_factorization_utilities.normalize_ratings(ratings_df.to_numpy())
# Convert the ratings matrix to float type
ratings_matrix = ratings_df.to_numpy().astype(float)

# Apply matrix factorization with modified parameters
T, D = low_rank_matrix_factorization(
        ratings_matrix,
        num_features=10,
        regularization_amount=0.1
    )

# Find all predicted ratings by multiplying T by D
mean_predicted_destination_ratings = np.matmul(T, D)
    
# Save the predicted ratings to a CSV file
mean_predicted_destination_ratings_df = pd.DataFrame(
        index=ratings_df.index,
        columns=ratings_df.columns,
        data= mean_predicted_destination_ratings
    )
mean_predicted_destination_ratings = mean_predicted_destination_ratings + means


# Construct full file paths
traveller_features_path = os.path.join( "C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\data", "Traveller_features.dat")
destination_features_path = os.path.join( "C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\data", "Destination_features.dat")
predicted_ratings_path = os.path.join( "C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\data", "mean_predicted_destination_ratings.dat")

# Save files with comprehensive error handling
try:
    # Save Traveller features
    with open(traveller_features_path, "wb") as f:
        pickle.dump(T, f)
    print(f"Traveller features saved to: {traveller_features_path}")

    # Save Destination features
    with open(destination_features_path, "wb") as f:
        pickle.dump(D, f)
    print(f"Destination features saved to: {destination_features_path}")

    # Save predicted ratings
    with open(predicted_ratings_path, "wb") as f:
        pickle.dump(mean_predicted_destination_ratings_df, f)
    print(f"Predicted ratings saved to: {predicted_ratings_path}")

except PermissionError:
    print("Permission denied. Try running the script as administrator.")
except IOError as e:
    print(f"IO Error occurred: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Verify file existence
print("\nVerifying saved files:")
for filepath in [traveller_features_path, destination_features_path, predicted_ratings_path]:
    if os.path.exists(filepath):
        print(f"File exists: {filepath}")
    else:
        print(f"File NOT found: {filepath}")