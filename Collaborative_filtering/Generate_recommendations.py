import numpy as np
import pandas as pd
import os
import sys

# Add the directory to the Python path
sys.path.append(os.path.abspath("C:\\Users\\Windows 10 Pro\\Movie-Recommendation-System"))

import matrix_factorization_utilities
from matrix_factorization_utilities import low_rank_matrix_factorization

try: 
    # Load user ratings
    raw_dataset_df = pd.read_csv("C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\Traveller-Destinations_Ratings_training dataset.csv")

    # Load destination types
    destinations_df = pd.read_csv("C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\destinations.csv", index_col="Destination_id")

    rating_column = 'Rating/Review'

    # Remove filtering to include all travellers
    filtered_df = raw_dataset_df

    # Create pivot table
    ratings_df = pd.pivot_table(
        filtered_df,
        values=rating_column,
        index="Traveller_id",
        columns="Destination_id",
        aggfunc="max"
    )

    # Convert the ratings matrix to float type
    ratings_matrix = ratings_df.to_numpy().astype(float)
    
    # Apply matrix factorization
    T, D = low_rank_matrix_factorization(
        ratings_matrix,
        num_features=10,
        regularization_amount=0.1
    )

    # Find all predicted ratings
    nw_predicted_destination_ratings = np.matmul(T, D)

    print("Enter a Traveller_id to get recommendations (Between 1 and 283):")
    traveller_id_to_search = int(input())

    print("Destinations previously reviewed by traveller_id {}:".format(traveller_id_to_search))

    # Use merge instead of join and specify suffixes if needed
    reviewed_destinations_df = raw_dataset_df[raw_dataset_df['Traveller_id'] == traveller_id_to_search]
    reviewed_destinations_df = reviewed_destinations_df.merge(
        destinations_df,
        left_on='Destination_id',
        right_index=True,
        suffixes=('', '_dest')
    )

    # Get the types of previously reviewed destinations
    reviewed_types = reviewed_destinations_df['Destination type'].unique()

    # Print previously reviewed destinations
    print(reviewed_destinations_df[['Destination name', 'Destination type', 'Rating/Review']])

    input("Press enter to continue.")

    # Generate recommendations
    traveller_ratings = nw_predicted_destination_ratings[traveller_id_to_search - 1]

    # Create a DataFrame with the correct destination IDs and their predicted ratings
    recommendations = pd.DataFrame({
        'Destination_id': ratings_df.columns,
        'rating': traveller_ratings
    })
    recommendations = recommendations.set_index('Destination_id')

    # Merge recommendations with destination information
    recommended_df = recommendations.merge(
        destinations_df,
        left_index=True,
        right_index=True
    )

    # Filter out already reviewed destinations
    already_reviewed = reviewed_destinations_df['Destination_id']
    recommended_df = recommended_df[~recommended_df.index.isin(already_reviewed)]

    # Filter out the types of destinations already reviewed
    recommended_df = recommended_df[~recommended_df['Destination type'].isin(reviewed_types)]

    # Sort by rating
    recommended_df = recommended_df.sort_values(by=['rating'], ascending=False)

    # Print top 5 recommendations
    print(recommended_df[['Destination name', 'Destination type', 'rating']].head(5))

except Exception as e:
    print(f"\nError during execution: {str(e)}")
    print("\nDataFrame info:")
    if 'raw_dataset_df' in locals():
        print("\nRaw dataset info:")
        print(raw_dataset_df.info())
        print("\nRaw dataset columns:", raw_dataset_df.columns.tolist())
    if 'destinations_df' in locals():
        print("\nDestinations info:")
        print(destinations_df.info())
        print("\nDestinations columns:", destinations_df.columns.tolist())
    if 'ratings_df' in locals():
        print("\nRatings pivot table info:")
        print(ratings_df.info())
        print("\nRatings columns:", ratings_df.columns.tolist())