import numpy as np
import pandas as pd
import os
import sys

# Add the directory to the Python path
sys.path.append(os.path.abspath("C:\\Users\\Windows 10 Pro\\Movie-Recommendation-System"))

import matrix_factorization_utilities
from matrix_factorization_utilities import low_rank_matrix_factorization

# Load user ratings
raw_training_dataset_df = pd.read_csv("C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\Traveller-Destinations_Ratings_training dataset.csv")
raw_testing_dataset_df = pd.read_csv("C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\Traveller-Destinations_Ratings_testing_dataset.csv")

rating_column = 'Rating/Review'

 # Remove filtering to include all travellers
filtered_df = raw_training_dataset_df
filtered_df = raw_testing_dataset_df 
# Convert the running list of traveller ratings into a matrix using the correct rating column
ratings_training_df = pd.pivot_table(
        filtered_df,
        values=rating_column, # Use the correct rating column
        index="Traveller_id",
        columns="Destination_id",
        aggfunc="max"
    )

ratings_testing_df = pd.pivot_table(
       filtered_df,
        values=rating_column,  # Use the correct rating column
        index="Traveller_id",
        columns="Destination_id",
        aggfunc="max"
    )

# Convert the ratings matrix to float type
ratings_matrix = ratings_training_df.to_numpy().astype(float)
    
# Apply matrix factorization to find latent features
T, D = low_rank_matrix_factorization(
        ratings_matrix,
        num_features=11,
        regularization_amount=1.1
    )


# Find all predicted ratings by multiplying T and D
test_predicted_destination_ratings = np.matmul(T, D)

# Measure RMSE
rmse_training = matrix_factorization_utilities.RMSE(ratings_training_df.to_numpy(),test_predicted_destination_ratings)
rmse_testing = matrix_factorization_utilities.RMSE(ratings_testing_df.to_numpy(),test_predicted_destination_ratings)

print("Training RMSE: {}".format(rmse_training))
print("Testing RMSE: {}".format(rmse_testing))
