import pandas as pd
import os
import numpy as np

DATA_PATH = os.path.join("Application", "training", "data", "cleaned_data_greenhouse.csv")

def load_dataset():
    """Load the raw dataset without any processing."""
    df = pd.read_csv(DATA_PATH) 
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()
    return df

def load_processed_dataset(encoder=None):
    """
    Load and process the dataset for model training.
    
    Args:
        encoder: Optional one-hot encoder for categorical variables
        
    Returns:
        X: Features DataFrame
        y: Target Series
        df: Complete DataFrame with engineered features
        features: List of feature names
    """
    # Load and clean data
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()

    # Create engineered features
    df["is_daytime"] = df["hourOfDay"].apply(lambda h: 1 if 6 <= h <= 18 else 0)
    
    # Set target variable
    y = df["timeUntilNextWateringInHours"]

    # Process categorical features if encoder is provided
    if encoder is not None:
        encoded = encoder.fit_transform(df[["plantGrowthStage"]])
        stage_cols = encoder.get_feature_names_out(["plantGrowthStage"])
        df_encoded = pd.DataFrame(encoded, columns=stage_cols, index=df.index)
        df = pd.concat([df, df_encoded], axis=1)
    else:
        stage_cols = []

    # Define features used for modeling
    base_features = [
        "Temperature",
        "Soil Humidity",
        "Air Humidity",
        "Light",
        "hourOfDay",
        "is_daytime",
        "timeSinceLastWateringInHours",
    ]
    features = base_features + list(stage_cols)

    # Extract features
    X = df[features]
    return X, y, df, features