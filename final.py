import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Global Variables
categorical_features = [
    # 'MAC Address',
    'Org Name', 'Start Time Zone', 'End Time Zone', 'Port Type', 'Plug Type',
    'Postal Code', 'Ended By', 'County', 'Station Name',
]

numerical_features = [
    'Total Duration (seconds)', 'Charging Time (seconds)', 'Energy (kWh)',
    'GHG Savings (kg)', 'Gasoline Savings (gallons)', 'Latitude',
    'Longitude', 'Fee', 'temp', 'rhum', 'prcp'
]

standardized_scaler = StandardScaler()

# Clean and Preprocess Data


def clean_data():
    # Load the CSV file
    df = pd.read_csv('dataset/EVChargingWithWeather.csv')

    df = df.head(500)  # For testing, limit rows

    # Convert time-based features to seconds
    df['Total Duration (seconds)'] = pd.to_timedelta(
        df['Total Duration (hh:mm:ss)']).dt.total_seconds()
    df['Charging Time (seconds)'] = pd.to_timedelta(
        df['Charging Time (hh:mm:ss)']).dt.total_seconds()

    # Convert datetime features
    datetime_columns = ['Start Date', 'End Date',
                        'Transaction Date (Pacific Time)']
    for col in datetime_columns:
        # Convert to pandas datetime
        df[col] = pd.to_datetime(df[col])

        # Generate timestamp
        df[f'{col}_timestamp'] = df[col].astype(
            'int64') // 1e9  # Seconds since epoch

        # Extract components
        df[f'{col}_year'] = df[col].dt.year
        df[f'{col}_month'] = df[col].dt.month
        df[f'{col}_day'] = df[col].dt.day
        df[f'{col}_hour'] = df[col].dt.hour
        df[f'{col}_minute'] = df[col].dt.minute
        df[f'{col}_second'] = df[col].dt.second
        df[f'{col}_day_of_week'] = df[col].dt.dayofweek  # Monday=0, Sunday=6
        df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5

    # Update feature lists with generated datetime features
    global numerical_features
    numerical_features.extend([f'{col}_timestamp' for col in datetime_columns])
    numerical_features.extend([f'{col}_{comp}' for col in datetime_columns
                               for comp in ['year', 'month', 'day', 'hour', 'minute', 'second', 'day_of_week', 'is_weekend']])

    # Prepare categorical and numerical DataFrames
    categorical_df = df[categorical_features]
    numerical_df = df[numerical_features]

    # Handle categorical features
    categorical_df = pd.get_dummies(categorical_df, drop_first=True)

    # Handle missing values for numerical features
    numerical_df = numerical_df.fillna(numerical_df.mean())

    # Standardize numerical features
    numerical_df_scaled = pd.DataFrame(standardized_scaler.fit_transform(
        numerical_df), columns=numerical_df.columns)

    # Combine processed categorical and numerical data
    df_processed = pd.concat([categorical_df, numerical_df_scaled], axis=1)
    return df_processed

# Feature Selection Using Random Forest


def random_forest(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # Train a random forest model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    # Get feature importances
    importance = pd.DataFrame(
        {'Feature': X.columns, 'Importance': rf.feature_importances_})
    importance = importance.sort_values(by='Importance', ascending=False)

    print("\nRandom Forest Feature Importances:")
    print(importance)

    return importance

# Dimensionality Reduction Using PCA


def pca(df):
    X = df.values
    pca = PCA()
    pca.fit(X)

    # Plot explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             pca.explained_variance_ratio_, marker='o')
    plt.title("PCA - Explained Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.show()

    print("\nCumulative Explained Variance:")
    print(np.cumsum(pca.explained_variance_ratio_))

    # Reduce dimensions by keeping components that explain 95% variance
    n_components = np.argmax(
        np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
    print(f"Number of Components to keep (95% variance): {n_components}")

    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(X)

    return pd.DataFrame(df_pca)


# Singular Value Decomposition (SVD) Analysis
def svd(df):
    X = df.values
    svd = TruncatedSVD(n_components=min(X.shape), random_state=42)
    svd.fit(X)

    # Plot the singular values
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(svd.singular_values_) + 1),
             svd.singular_values_, marker='o')
    plt.title("SVD - Singular Values")
    plt.xlabel("Singular Value Index")
    plt.ylabel("Singular Value Magnitude")
    plt.show()

    print("\nCondition Number of the Matrix:")
    print(np.linalg.cond(X))

    # Reduce dimensions to rank (number of components)
    df_svd = svd.transform(X)

    return pd.DataFrame(df_svd)


# Variance Inflation Factor (VIF) Calculation
def vif(df):
    X = df.values
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(
        X, i) for i in range(X.shape[1])]

    print("\nVariance Inflation Factor (VIF) Analysis:")
    print(vif_data)

    # Remove features with high VIF (typically > 5)
    high_vif_features = vif_data[vif_data["VIF"] > 5]["Feature"]
    print(f"Features with high VIF (>5): {list(high_vif_features)}")

    df_cleaned = df.drop(columns=high_vif_features)
    return df_cleaned


# Perform Regression
def do_regression(df, target, feature_selection_method=None):
    # Separate target before feature selection
    y = df[target]
    df = df.drop(columns=[target])  # Remove target from features

    # Apply feature selection method
    if feature_selection_method:
        df = feature_selection_method(df)

    # Add the target back to the processed DataFrame
    df[target] = y

    print("df after features", list(df.columns), flush=True)
    X = df.drop(columns=[target])  # All features except the target
    y = df[target]

    # Train-test split (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Add a constant term for intercept
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)

    # Train the regression model
    model = sm.OLS(y_train, X_train_sm).fit()

    # Print the model summary
    print(model.summary())

    # Predict on test data
    y_pred = model.predict(X_test_sm)

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual", marker="o")
    plt.plot(y_pred.values, label="Predicted", marker="x")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.xlabel("Sample")
    plt.ylabel(target)
    plt.show()

    # Calculate additional regression metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model


df = clean_data()

print(df.columns)

# Change to 'random_forest', 'svd', or 'vif' as needed
do_regression(df, 'Energy (kWh)', feature_selection_method=random_forest)
