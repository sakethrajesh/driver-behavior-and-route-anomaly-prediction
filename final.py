import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import statsmodels.api as sm
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


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

drop_features = [
    # fully null columns
    'EVSE ID', 'County', 'System S/N', 'Model Number', 
    
    # unnessary columns
    'User ID', 'MAC Address', 'Org Name', 'Plug In Event Id',
    
    # missing values
    'Driver Postal Code',
]

standardized_scaler = StandardScaler()

# Clean and Preprocess Data

def clean_data():
    df = pd.read_csv('dataset/EVChargingWithWeather.csv')
    df = df.head(500)

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
    
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    df[numerical_cols] = standardized_scaler.fit_transform(df[numerical_cols])
    
    return df

def get_train_test_df(df, target, feature_selection_method=None):
    y = df[target]
    y = np.argmax(y, axis=1)
    X = df.drop(target, axis=1)
    
    pattern = '^Ended By_'
    columns_to_drop = X.filter(regex=pattern).columns
    X.drop(columns=columns_to_drop, axis=1, inplace=True)

    if feature_selection_method:
        X = feature_selection_method(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test

# Feature Selection Using Random Forest
def random_forest(df, target='Energy (kWh)'):
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
    # Convert DataFrame to numeric values, coerce errors to NaN
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values
    df_numeric = df_numeric.dropna()

    X = df_numeric.values
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
    # print(np.linalg.cond(X))

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

    y = df[target]
    df = df.drop(columns=[target])

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

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train the regression model
    model.fit(X_train, y_train)

    # Print model coefficients (Optional)
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")

    # Predict on test data
    y_pred = model.predict(X_test)

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual", marker="o")
    plt.plot(y_pred, label="Predicted", marker="x")
    plt.legend()
    plt.title("Actual vs Predicted")
    plt.xlabel("Sample")
    plt.ylabel(target)
    plt.show()

    # Calculate additional regression metrics
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return model

def multinomial_logistic_regression(df, target, feature_selection_method=None):
    
    X_train, X_test, y_train, y_test = get_train_test_df(df, target)
    
    # Initialize multinomial logistic regression model
    logModel = LogisticRegression()
    
    # Parameter grid for GridSearchCV
    param_grid = {
        'penalty': ['l2'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['newton-cg', 'lbfgs', 'sag'],
        'max_iter': [5000, 10000]
    }
    
    # Perform Grid Search for hyperparameter tuning
    clf = GridSearchCV(logModel, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    
    # Fit the model on training data
    clf.fit(X_train, y_train)
    
    # Best model after grid search
    best_model = clf.best_estimator_
    print(f'Best Parameters: {clf.get_params}')
    
    # Evaluate the model on test data
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy on test data: {test_accuracy:.3f}')
    
    return best_model

def random_forest_classification(df, target, feature_selection_method=None, method=None):
    
    X_train, X_test, y_train, y_test = get_train_test_df(df, target)

    # Bagging-specific parameter grid
    param_grid = {
        # Base model parameters prefixed with `estimator__`
        'estimator__n_estimators': [int(x) for x in np.linspace(start=10, stop=80, num=10)],
        'estimator__max_features': ['auto', 'sqrt'],
        'estimator__max_depth': [2, 4],
        'estimator__min_samples_split': [2, 5],
        'estimator__min_samples_leaf': [1, 2],
        'estimator__bootstrap': [True, False]
    }
    
    rf_Model = RandomForestClassifier()
    
    if method == 'bagging':
        rf_Model = BaggingClassifier(estimator=rf_Model)
        param_grid.update({
            'n_estimators': [10, 50],  # Number of bagging iterations
            'max_samples': [0.5, 1.0],  # Fraction of samples for each bag
            'max_features': [0.5, 1.0],  # Fraction of features for each bag
            'bootstrap': [True, False],  # Bootstrapping or not
        })
    
    elif method == 'boosting':
        rf_Model = AdaBoostClassifier(estimator=rf_Model)
    
    elif method == 'stacking':
        rf_Model = StackingClassifier(estimator=rf_Model, final_estimator=rf_Model)
    
    rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = {}, cv = 3, verbose=2, n_jobs = -1)
    
    rf_Grid.fit(X_train, y_train)
    
    print(f'Best Parameters: {rf_Grid.best_params_}')
    
    print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
    print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')
    

def naive_bayes_classification(df, target, feature_selection_method=None):
    
    X_train, X_test, y_train, y_test = get_train_test_df(df, target)

    model = GaussianNB()
    
    param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

    rf_Grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)

    rf_Grid.fit(X_train, y_train)
    
    print(f'Best Parameters: {rf_Grid.best_params_}')
    
    print(f'Train Accuracy - : {rf_Grid.score(X_train, y_train):.3f}')
    print(f'Test Accuracy - : {rf_Grid.score(X_test, y_test):.3f}')
    

def knn_classification(df, target, feature_selection_method=None):
    X_train, X_test, y_train, y_test = get_train_test_df(df, target)

    knn = KNeighborsClassifier()
    
    # Define the parameter grid
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors
        'weights': ['uniform', 'distance'],  # Weighting scheme
        'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    y_pred = grid_search.predict(X_test)

    # Best parameters and model
    best_knn = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # print(f'Train Accuracy - : {rf_Grid.score(X_train, y_train):.3f}')
    # print(f'Test Accuracy - : {rf_Grid.score(X_test, y_test):.3f}')
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
   
    
def svn_classification(df, target, feature_selection_method=None):
    X_train, X_test, y_train, y_test = get_train_test_df(df, target)

    # Initialize the model
    model = SVC()
    
    # Define the parameter grid
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
        'degree': [2, 3, 4],  # Degree for poly kernels
        'gamma': ['scale', 'auto']  # Kernel coefficient
    }

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters and model
    best_model = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Predict on the test set
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
        
    
    

    

df = clean_data()

do_regression(df, 'Energy (kWh)', feature_selection_method=None)

# random_forest_classification(df, ['Ended By_Customer', 'Ended By_Plug Out at Vehicle',], method='stacking')

# multinomial_logistic_regression(df, ['Ended By_Customer', 'Ended By_Plug Out at Vehicle',])

# naive_bayes_classification(df, ['Ended By_Customer', 'Ended By_Plug Out at Vehicle',])

# knn_classification(df, ['Ended By_Customer', 'Ended By_Plug Out at Vehicle',])

# svn_classification(df, ['Ended By_Customer', 'Ended By_Plug Out at Vehicle',])