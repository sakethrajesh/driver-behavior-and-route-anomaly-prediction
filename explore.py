import pandas as pd

drop_features = [
    # fully null columns
    'EVSE ID', 'County', 'System S/N', 'Model Number', 
    
    # unnessary columns
    'User ID', 'MAC Address', 'Org Name', 'Plug In Event Id',
    
    # missing values
    'Driver Postal Code',
]

df = pd.read_csv('dataset/EVChargingWithWeather.csv')
df = df.head(50000)

# Convert time-based features to seconds
df['Total Duration (seconds)'] = pd.to_timedelta(
    df['Total Duration (hh:mm:ss)']).dt.total_seconds()
df['Charging Time (seconds)'] = pd.to_timedelta(
    df['Charging Time (hh:mm:ss)']).dt.total_seconds()

df.drop(['Total Duration (hh:mm:ss)', 'Charging Time (hh:mm:ss)'] + drop_features, axis=1, inplace=True)

# Convert datetime features
datetime_columns = ['Start Date', 'End Date', 'Transaction Date (Pacific Time)']
for col in datetime_columns:
    # Convert to pandas datetime
    df[col] = pd.to_datetime(df[col])

    # Generate timestamp
    df[f'{col}_timestamp'] = df[col].astype(
        'int64') // 1e9  # Seconds since epoch

    # Extract components
    df[f'{col}_year'] = df[col].dt.year.astype(float)
    df[f'{col}_month'] = df[col].dt.month.astype(float)
    df[f'{col}_day'] = df[col].dt.day.astype(float)
    df[f'{col}_hour'] = df[col].dt.hour.astype(float)
    df[f'{col}_minute'] = df[col].dt.minute.astype(float)
    df[f'{col}_second'] = df[col].dt.second.astype(float)
    df[f'{col}_day_of_week'] = df[col].dt.dayofweek.astype(float)  # Monday=0, Sunday=6
    df[f'{col}_is_weekend'] = df[col].dt.dayofweek >= 5
        
df.drop(datetime_columns, axis=1, inplace=True)

# df = pd.get_dummies(df, columns=['Transaction Date (Pacific Time)_is_weekend', 'End Date_is_weekend', 'Start Date_is_weekend', ],drop_first=True)
df = pd.get_dummies(df, columns=['Station Name', 'Start Time Zone', 'End Time Zone', 'Port Type',
       'Plug Type', 'Address 1', 'City', 'State/Province', 'Country',
       'Currency', 'Ended By', 'Postal Code', 'Port Number'],drop_first=True)

df['prcp'] = df['prcp'].fillna(df['prcp'].mean())

# # Display the first few rows of the dataset
# print(df.head())

# # Summary statistics
# print(df.describe())

# Information about the dataset
print(df.info())

print(df.columns)

string_df = df.select_dtypes(include=['float'])
print(string_df.columns)