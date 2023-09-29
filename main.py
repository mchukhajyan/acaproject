import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import StandardScaler,  MinMaxScaler, RobustScaler

columns_keep = ['id', 'host_response_time', 'host_response_rate', 'host_acceptance_rate',
                'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
                'latitude', 'longitude', 'property_type', 'room_type', 'accommodates',
                'bathrooms_text', 'bedrooms', 'beds', 'price', 'minimum_nights_avg_ntm',
                'maximum_nights_avg_ntm', 'review_scores_rating', 'review_scores_accuracy',
                'review_scores_checkin', 'review_scores_cleanliness',
                'review_scores_communication', 'review_scores_location',
                'review_scores_value', 'instant_bookable']


def data_get(filename):
    listings = pd.read_csv(filename)
    return listings


def data_prepare(data_original, columns_keep):
    data = data_original[columns_keep]

    # Rename the 'bathrooms_text' column to 'bathrooms'
    data = data.rename(columns={'bathrooms_text': 'bathrooms'})

    for columns_float in ['host_response_rate', 'host_acceptance_rate', 'price', 'bathrooms']:
        if data[columns_float].dtype == float or data[columns_float].dtype == int:
            continue
        if columns_float == 'bathrooms':
            data['bathrooms'] = data['bathrooms'].str.extract('(\d+)')
        data[columns_float] = pd.to_numeric(data[columns_float].str.replace('[\%\$,]', '', regex=True), errors='coerce')

    # Print the updated column names to verify the change
    print(data.columns)

    # Replace missing values in 'bedrooms' column with 0
    for missing_column in ['bedrooms', 'beds']:
        data[missing_column] = data[missing_column].fillna(0)

    data = data.dropna(
        subset=['price', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'latitude', 'longitude',
                'bathrooms',  'bedrooms', 'beds', 'review_scores_location',  'room_type', 'review_scores_rating', 'review_scores_value'])

    # Delete outliers, later
    return data



def data_engineer_features(data_original):
    data = data_original.copy()

    print_correlation_heatmap(listings)

    # Change features, add new features, etc...
    features_keep = ['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_location', 'neighbourhood_group_cleansed', 'room_type', 'review_scores_rating', 'review_scores_value']
    data = data[features_keep]
    data = data.dropna()
    return data


def print_correlation_heatmap(data):
    numerical_features = listings.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numerical_features.corr()

    # heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap for Numerical Features')
    plt.show()


def train_and_test_regression(data, ModelAlgorithmClass):

#Split the data into training and testing sets
    X = data.drop('price', axis=1)  # Features
    y = data['price']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Linear Regression model
    model = ModelAlgorithmClass()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mse)  # Calculate RMSE

    # MSE, RMSE
    print(f'{model} Metrics:')
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Mean Absolute Error (MAE): {mae}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')

    # Plot actual vs. predicted prices
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title(f'Actual vs. Predicted Prices with {model}')
    plt.show()


    # Function to train and validate regression model
    def train_and_validate_regression(data, ModelAlgorithmClass):
        X = data.drop('price', axis=1)  # Features
        y = data['price']  # Target variable

    #Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        model = ModelAlgorithmClass()
        model.fit(X_train, y_train)

        y_pred_valid = model.predict(X_valid)
        y_pred_test = model.predict(X_test)

        mse_valid = mean_squared_error(y_valid, y_pred_valid)
        mae_valid = mean_absolute_error(y_valid, y_pred_valid)
        rmse_valid = math.sqrt(mse_valid)

        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = math.sqrt(mse_test)

        print(f'Validation Metrics for {model}')
        print(f'Mean Squared Error (MSE): {mse_valid}')
        print(f'Mean Absolute Error (MAE): {mae_valid}')
        print(f'Root Mean Squared Error (RMSE): {rmse_valid}')

        print("\nTest Metrics:")
        print(f'Mean Squared Error (MSE): {mse_test}')
        print(f'Mean Absolute Error (MAE): {mae_test}')
        print(f'Root Mean Squared Error (RMSE): {rmse_test}')

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_valid, y=y_pred_valid)
        plt.xlabel('Actual Price (Validation Set)')
        plt.ylabel('Predicted Price (Validation Set)')
        plt.title('Actual vs. Predicted Prices (Validation Set)')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred_test)
        plt.xlabel('Actual Price (Test Set)')
        plt.ylabel('Predicted Price (Test Set)')
        plt.title(f'Actual vs. Predicted Prices (Test Set) for {model}')
        plt.show()




def data_normalize(data_original, ScalerClass):
    data = data_original.copy()
    # Normalize data here

    data = data_original.copy()
    # Select the columns you want to normalize (excluding the target variable 'price')
    columns_to_normalize = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'review_scores_location', 'review_scores_rating', 'review_scores_value']

    scaler = ScalerClass()
 # Fit and transform the selected columns
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])
    return data



listings = data_get('listings.csv')
print(listings.head())

listings = data_prepare(listings, columns_keep)
print(listings.head())


# most expensive
top_10_prices = listings.sort_values(by='price', ascending=False).head(10)
print("\nTop 10 Prices (Highest to Lowest):")
print(top_10_prices[['id', 'price']])


# cheapest
bottom_10_prices = listings.sort_values(by='price').head(10)
print("\nBottom 10 Prices (Lowest to Highest):")
print(bottom_10_prices[['id', 'price']])



# boxplot for all prices
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.boxplot(x=listings['price'], orient='h', color='skyblue', width=0.4, fliersize=6)
plt.xlabel('Price', fontsize=14)
plt.title('Horizontal Boxplot of Prices', fontsize=16)
plt.xticks(fontsize=12)
plt.show()

#drop the prices greater than 20000
listings = listings[listings['price'] <= 20000]
sns.set_style("whitegrid")
plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
sns.boxplot(x='price', data=listings, orient='h', color='skyblue', width=0.4, fliersize=6)
plt.xlabel('Price', fontsize=14)
plt.title('Horizontal Boxplot of Prices Less Than 20000', fontsize=16)
plt.xticks(fontsize=12)
plt.show()


#boxplots by room type
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sns.boxplot(x='room_type', y='price', data=listings, palette='hls')
plt.xlabel('Room Type', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Boxplots of Prices by Room Type', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


#boxplots by neighbourhood group
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
sns.boxplot(x='neighbourhood_group_cleansed', y='price', data=listings, palette='husl')
plt.xlabel('Neighbourhood Group', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Boxplots of Prices by Neighbourhood Groups', fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.show()


# Boxplots of room types colored by neighbourhood groups
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))

# Create a categorical plot (catplot) to show boxplots of prices by 'room_type' and 'neighbourhood_group'
sns.catplot(x='room_type', y='price', hue='neighbourhood_group_cleansed', data=listings, kind='box', palette='husl', height=6, aspect=2, legend_out= False)

# Add labels and title
plt.xlabel('Room Type', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Boxplots of Prices by Room Type and Neighbourhood Group', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title='Neighbourhood Group', title_fontsize=12, fontsize=10)
plt.show()

#calculating the outliers
# IQR (Interquantile Range)
Q1 = listings['price'].quantile(0.25)
Q3 = listings['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Count the number of outliers
outliers = listings[(listings['price'] < lower_bound) | (listings['price'] > upper_bound)]
outliers_count = len(outliers)

print(f"Number of outliers: {outliers_count}")

# as the number of outliers is 732, which is very small compared to the overall data, let's drop the outliers
listings = listings[(listings['price'] >= lower_bound) & (listings['price'] <= upper_bound)]

#after the dropping of outliers
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.catplot(
    x='room_type',
    y='price',
    hue='neighbourhood_group_cleansed',
    data=listings,
    kind='box',
    palette='husl',
    height=6,
    aspect=2,
    legend_out=False
)

# Add labels and title
plt.xlabel('Room Type', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Boxplots of Prices by Room Type and Neighbourhood Group', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Specify the legend's position
plt.legend(title='Neighbourhood Group', title_fontsize=12, fontsize=10, loc='upper right')
plt.show()


# Define a function to remove outliers based on IQR for a specific room type
def remove_outliers_by_room_type(group):
    Q1_rt = group['price'].quantile(0.25)
    Q3_rt = group['price'].quantile(0.75)
    IQR_rt = Q3_rt - Q1_rt
    lower_bound_rt = Q1_rt - 1.5 * IQR_rt
    upper_bound_rt = Q3_rt + 1.5 * IQR_rt
    return group[(group['price'] >= lower_bound_rt) & (group['price'] <= upper_bound_rt)]


# Get the unique room types in your DataFrame
unique_room_types = listings['room_type'].unique()

# Create an empty DataFrame to store the filtered results
filtered_listings = pd.DataFrame()

# Iterate through each room type and apply the outlier removal function
for room_type in unique_room_types:
    room_type_group = listings[listings['room_type'] == room_type]
    filtered_group = remove_outliers_by_room_type(room_type_group)
    filtered_listings = pd.concat([filtered_listings, filtered_group])

# Reset the index to obtain a single DataFrame
filtered_listings = filtered_listings.reset_index(drop=True)


#after dropping outliers from room types
sns.set(style="whitegrid")
plt.figure(figsize=(12, 8))
sns.catplot(
    x='room_type',
    y='price',
    hue='neighbourhood_group_cleansed',
    data=filtered_listings,
    kind='box',
    palette=sns.color_palette("Set2"),
    height=6,
    aspect=2,
    legend_out=False
)

# Add labels and title
plt.xlabel('Room Type', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Boxplots of Prices by Room Type and Neighbourhood Group: Outliers Removed', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Specify the legend's position
plt.legend(title='Neighbourhood Group', title_fontsize=12, fontsize=10, loc='upper right')
plt.show()


filtered_listings= data_engineer_features(filtered_listings)
print(filtered_listings.head())

#change categorical to numerical
categorical_columns = ['neighbourhood_group_cleansed', 'room_type']
# Perform one-hot encoding for the specified columns
listings_encoded = pd.get_dummies(filtered_listings, columns=categorical_columns, prefix=categorical_columns)
print(listings_encoded.head(5))

listings_encoded = data_normalize(listings_encoded, StandardScaler)

#also MinMaxScaler,Robust Scaler
#listings_encoded = data_normalize(listings_encoded, MinMaxScaler)
#listings_encoded = data_normalize(listings_encoded, RobustScaler)
print(listings_encoded.head())

train_and_test_regression(listings_encoded, LinearRegression)
#train_and_test_regression(listings_encoded, Ridge)