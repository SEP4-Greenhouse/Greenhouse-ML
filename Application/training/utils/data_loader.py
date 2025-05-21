# === data_loader.py ===
from Application.training.utils.imports import *

DATA_PATH = os.path.join("Application","training", "data", "greenhouse_data_mal.csv")

def load_dataset():
    """Load greenhouse dataset from CSV."""
    return pd.read_csv(DATA_PATH)


def load_processed_dataset(encoder=None):
    df = pd.read_csv(DATA_PATH)

    # Derived features
    df['TimeUntilNextWatering (hours)'] = df['TimeSinceLastWatering (hours)'].max() - df['TimeSinceLastWatering (hours)']
    df['is_daytime'] = df['HourOfDay'].apply(lambda h: 1 if 6 <= h <= 18 else 0)
    df['moisture_drop_rate'] = df['SoilMoisture (%)'].diff().fillna(0) * -1

    # Encode GrowthStage
    if encoder is not None:
        encoded = encoder.transform(df[['GrowthStage']])
        growth_stage_cols = encoder.get_feature_names_out(['GrowthStage'])
        df = pd.concat([df, pd.DataFrame(encoded, columns=growth_stage_cols)], axis=1)
    else:
        growth_stage_cols = []

    # Define features
    base_features = [
        'SoilMoisture (%)',
        'AirTemperature (°C)',
        'AirHumidity (%)',
        'LightLevel (lux)',
        'HourOfDay',
        'is_daytime',
        'moisture_drop_rate',
    ]
    features = base_features + list(growth_stage_cols)

    X = df[features]
    y = df['TimeUntilNextWatering (hours)']

    return X, y, df, features
