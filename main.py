import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    'dataset/driver_behavior_route_anomaly_dataset_with_derived_features.csv'
)

df.columns = df.columns.str.strip()
print(df.columns)

print(df['weather_conditions'])

df = pd.get_dummies(
    df,
    columns=[
        'weather_conditions', 'road_type', 'traffic_condition'
    ],
    drop_first=True
)

# print(df.info())
# print(df.describe())
# print(df.head())

features = ['speed', 'acceleration', 'steering_angle', 'heading', 'trip_duration',
            'trip_distance', 'fuel_consumption', 'rpm', 'brake_usage', 'lane_deviation',
            'stop_events', 'acceleration_variation', 'behavioral_consistency_index']

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

print("\nStandardized df:")

pca = PCA(n_components=2)  # Reduce to 2 components for visualization
X_pca = pca.fit_transform(df_scaled)

# Create a DataFrame for the PCA components
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

# Display explained variance ratio
print(f'Explained Variance Ratio: {pca.explained_variance_ratio_}')

# plt.scatter(pca_df['PC1'], pca_df['PC2'])
# plt.title('PCA of Vehicle Trip Data')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

print(f'PCA Components:\n{pca.components_}')
