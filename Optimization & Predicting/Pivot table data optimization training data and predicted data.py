import numpy as np
import pandas as pd
import webbrowser
import sys
import os

# Add the directory to the Python path
sys.path.append(os.path.abspath("C:\\Users\\Windows 10 Pro\\Movie-Recommendation-System"))

import matrix_factorization_utilities
from matrix_factorization_utilities import low_rank_matrix_factorization

try:
    # Load user ratings
    raw_dataset_df = pd.read_csv("C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\Traveller-Destinations_Ratings_training dataset.csv")

    # Print the first few rows and columns for inspection
    print("Raw data sample:")
    print(raw_dataset_df.head())
    print("\nColumns in raw data:", raw_dataset_df.columns.tolist())

    # Ensure we have the correct numerical ratings column
    rating_column = 'Rating/Review'  # Updated column name
    
    if rating_column not in raw_dataset_df.columns:
        print(f"Available columns: {raw_dataset_df.columns.tolist()}")
        raise ValueError(f"Rating column '{rating_column}' not found in dataset")

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

    # Convert the ratings matrix to float type
    ratings_matrix = ratings_df.to_numpy().astype(float)
    
    # Print shape and sample of data for debugging
    print("\nMatrix shape:", ratings_matrix.shape)
    print("Sample of ratings matrix:\n", ratings_matrix[:5, :5])
    
    # Apply matrix factorization with modified parameters
    T, D = low_rank_matrix_factorization(
        ratings_matrix,
        num_features=10,
        regularization_amount=0.1
    )

    # Find all predicted ratings by multiplying T by D
    nw_predicted_destination_ratings = np.matmul(T, D)

    # Save the predicted ratings to a CSV file
    nw_predicted_destination_ratings_df = pd.DataFrame(
        index=ratings_df.index,
        columns=ratings_df.columns,
        data=nw_predicted_destination_ratings
    )
    nw_predicted_destination_ratings_df.to_csv("C:\\Users\\Windows 10 Pro\\Documents\\Recommendation Engine\\African Destinations and ratings\\nw_predicted_destination_ratings.csv")

    # Generate HTML output for the ratings DataFrame
    html_ratings = ratings_df.to_html(na_rep="")
    
    # Save the HTML for the original ratings
    with open("D_ratings.html", "w", encoding='utf-8') as f:
        f.write(html_ratings)

    # Generate HTML output for the predicted ratings DataFrame
    html_predicted = nw_predicted_destination_ratings_df.to_html(na_rep="")
    
    # Save the HTML for the predicted ratings
    with open("D_predicted_ratings.html", "w", encoding='utf-8') as f:
        f.write(html_predicted)

    # Open the original ratings HTML in the web browser
    webbrowser.open("D_ratings.html")

    # Open the predicted ratings HTML in the web browser
    webbrowser.open("D_predicted_ratings.html")

except Exception as e:
    print(f"\nError during execution: {str(e)}")
    print("\nDataFrame info:")
    if 'raw_dataset_df' in locals():
        print("\nRaw dataset info:")
        print(raw_dataset_df.info())
    if 'ratings_df' in locals():
        print("\nPivot table info:")
        print(ratings_df.info())