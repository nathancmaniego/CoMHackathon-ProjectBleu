# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from scipy.spatial import cKDTree
import json

# Load CSV files
rainfall_data = pd.read_csv('rainfalldata.csv')
storm_node_data = pd.read_csv('stormsewer_node.csv')
storm_pipe_data = pd.read_csv('stormsewer_pipes.csv')


# Update column names or adjust the merge function accordingly
rainfall_data.rename(columns={'date': 'Date',
                              'precipitation': 'Rainfall_Intensity',
                              'max_temperature': 'Max_Temperature',
                              'avg_hourly_temperature': 'Avg_Hourly_Temperature'}, inplace=True)  # Correct column names based on print output

# Extract month from the date column
rainfall_data['Date'] = pd.to_datetime(rainfall_data['Date'])
rainfall_data['Month'] = rainfall_data['Date'].dt.month

# Select useful features from each dataset
rainfall_features = rainfall_data[['Date', 'Month', 'Rainfall_Intensity', 'Max_Temperature', 'Avg_Hourly_Temperature']].copy()
storm_node_features = storm_node_data[['ASSETID', 'ASSETGROUP', 'ASSETTYPE', 'X', 'Y']].copy()
storm_pipe_features = storm_pipe_data[['ASSETID', 'DIAMETER', 'PIPECLASS', 'Shape__Length']].copy()

# Drop rows with NaN values in coordinates to avoid errors in KDTree
storm_node_features.dropna(subset=['X', 'Y'], inplace=True)

# Check if storm node data has any coordinates before merging
if 'X' in storm_node_features.columns and 'Y' in storm_node_features.columns:
    node_coords = storm_node_features[['X', 'Y']].values

    # Ensure coordinates are finite
    if np.isfinite(node_coords).all() and len(node_coords) > 0:
        # Use KDTree for efficient spatial matching
        rainfall_coords = rainfall_features[['Rainfall_Intensity', 'Max_Temperature']].apply(lambda row: (row['Rainfall_Intensity'], row['Max_Temperature']), axis=1).tolist()
        # Filter out any NaN or infinite values from rainfall_coords
        rainfall_coords = np.array(rainfall_coords)
        finite_mask = np.isfinite(rainfall_coords).all(axis=1)
        rainfall_coords = rainfall_coords[finite_mask]
        filtered_rainfall_features = rainfall_features[finite_mask]

        tree = cKDTree(node_coords)
        distances, indices = tree.query(rainfall_coords, k=1)

        # Add nearest storm node information to rainfall data
        nearest_nodes = storm_node_features.iloc[indices].reset_index(drop=True)
        rainfall_features = pd.concat([filtered_rainfall_features.reset_index(drop=True), nearest_nodes], axis=1)

# Check for common ASSETID values before merging
if 'ASSETID' in rainfall_features.columns and 'ASSETID' in storm_pipe_features.columns:
    common_assetids = set(rainfall_features['ASSETID']).intersection(set(storm_pipe_features['ASSETID']))
else:
    common_assetids = set()

if len(common_assetids) > 0:
    # Merge the result with storm pipe data based on ASSETID if it exists in both datasets
    merged_data = pd.merge(rainfall_features, storm_pipe_features, how='inner', on='ASSETID')
else:
    merged_data = rainfall_features.copy()

# If DIAMETER and Shape__Length columns are missing, attempt proximity-based matching or set defaults
if 'DIAMETER' not in merged_data.columns or 'Shape__Length' not in merged_data.columns:
    if 'DIAMETER' not in merged_data.columns:
        merged_data['DIAMETER'] = storm_pipe_features['DIAMETER'].mean() if not storm_pipe_features['DIAMETER'].isna().all() else 1.0  # Assign average or default value
    if 'Shape__Length' not in merged_data.columns:
        merged_data['Shape__Length'] = storm_pipe_features['Shape__Length'].mean() if not storm_pipe_features['Shape__Length'].isna().all() else 100.0  # Assign average or default value

# Impute missing values in merged data
merged_data = merged_data.ffill().bfill()


# Add synthetic 'Node_Capacity' to simulate actual conditions based on drainage design requirements
# Update Node_Capacity calculation to reflect storm drainage capacity based on guidelines (e.g., limiting max intensity)
merged_data['Node_Capacity'] = (merged_data['Rainfall_Intensity'] * 1.5 + merged_data['Max_Temperature']).clip(0, 75)  # Updated limits based on storm drainage design

# Define features X and labels y
if 'DIAMETER' in merged_data.columns and 'Shape__Length' in merged_data.columns:
    X = merged_data[['Rainfall_Intensity', 'Max_Temperature', 'Avg_Hourly_Temperature', 'Node_Capacity', 'DIAMETER', 'Shape__Length']]
    y = merged_data['Release_Decision'] = ((merged_data['Rainfall_Intensity'] > 10) & (merged_data['Node_Capacity'] < 40)).astype(int)  # Adjusted threshold based on stormwater guidelines

    # Impute missing values in X
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Feature selection to select the most impactful features
    selector = SelectKBest(score_func=f_classif, k='all')
    X_selected = selector.fit_transform(X_imputed, y)
    selected_features = X.columns[selector.get_support()]

    # Split data into training and testing sets if there are enough samples
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Train a Random Forest model with cross-validation to better assess performance
        model = RandomForestClassifier(max_depth=5, n_estimators=50, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=10)

        # Train the model on the full training set
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Evaluate model accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])  # Ensure both labels are included

        report = classification_report(y_test, y_pred, labels=[0, 1], zero_division=1)  # Handle zero division warnings

        # Calculate stormwater captured with and without blue roofs
        stormwater_with_blue_roof = merged_data.loc[merged_data['Release_Decision'] == 0, 'Rainfall_Intensity'].sum() * 0.8  # 80% retention efficiency
        stormwater_without_blue_roof = merged_data['Rainfall_Intensity'].sum()
        stormwater_saved = stormwater_without_blue_roof - stormwater_with_blue_roof
        percentage_saved = (stormwater_saved / stormwater_without_blue_roof) * 100
        print(f'Stormwater Saved by Blue Roofs: {stormwater_saved:.2f} units ({percentage_saved:.2f}%)')

        # Visualize the data
        plt.scatter(merged_data['Month'] + merged_data['Date'].dt.day / 31, merged_data['Rainfall_Intensity'], c=merged_data['Release_Decision'], cmap='viridis')
        plt.xlabel('Month (with daily spread)')
        plt.ylabel('Rainfall Intensity')
        plt.title('Monthly Rainfall Intensity with Release Decision')
        plt.colorbar(label='Release Decision (0 or 1)')
        plt.show()

        # Export merged data and predictions to JSON
        merged_data['Prediction'] = model.predict(X_selected)
        merged_data_to_export = merged_data.to_dict(orient='records')
        with open('merged_data.json', 'w') as json_file:
            json.dump(merged_data_to_export, json_file, indent=4, default=str)


# Calculate stormwater captured with and without blue roofs
stormwater_with_blue_roof = merged_data.loc[merged_data['Release_Decision'] == 0, 'Rainfall_Intensity'].sum() * 0.8  # 80% retention efficiency
stormwater_without_blue_roof = merged_data['Rainfall_Intensity'].sum()
stormwater_saved = stormwater_without_blue_roof - stormwater_with_blue_roof
percentage_saved = (stormwater_saved / stormwater_without_blue_roof) * 100

# Add print statements for released water and rain retained
released_water = stormwater_without_blue_roof - stormwater_saved
retained_rain = stormwater_saved
print(f'Released Water: {released_water:.2f} units')
print(f'Rain Retained by Blue Roofs: {retained_rain:.2f} units')

print(f'Stormwater Saved by Blue Roofs: {stormwater_saved:.2f} units ({percentage_saved:.2f}%)')






















