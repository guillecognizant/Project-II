import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

import constants


# Plotting 
def histograms_eda(data):
    ''' Function that plots the dataset data's columns into histostoplots containing its count '''
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")

    # Create histograms for the numerical columns
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12))
    fig.suptitle('Histograms of Real Estate Data', fontsize=16)

    for i, col in enumerate(constants.cols):
        sns.histplot(data[col], kde=True, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(col)
        axes[i//2, i%2].set_xlabel('')
        axes[i//2, i%2].set_ylabel('')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def scatterplots_eda(data):
    ''' Function that plots Scatter plots to observe the relationship with house price '''
    # Scatter plots to observe the relationship with house price
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle('Scatter Plots with House Price of Unit Area', fontsize=16)

    # Scatter plot for each variable against the house price
    sns.regplot(data=data, x=constants.house_age_col, y=constants.price_col, ax=axes[0, 0])
    sns.regplot(data=data, x=constants.mtr_dist_col, y=constants.price_col, ax=axes[0, 1])
    sns.regplot(data=data, x=constants.convin_str_col, y=constants.price_col, ax=axes[1, 0])
    sns.regplot(data=data, x=constants.lat_col, y=constants.price_col, ax=axes[1, 1])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def scattermap_eda(data):
    ''' Function to plot points in the map '''
    fig = px.scatter_mapbox(data, 
                            lat=constants.lat_col, 
                            lon=constants.lon_col, 
                            #hover_name="Address", 
                            #hover_data=["Address", "Listed"],
                            color=constants.price_col,
                            color_continuous_scale='viridis',
                            size=constants.price_col,
                            zoom=8, 
                            height=800,
                            width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def scatter3d_eda (data):
    ''' Function to plot scatter plats in the 3D space of 3 variables '''
    fig = px.scatter_3d(data,x=constants.lon_col, y=constants.lat_col, 
                            z=constants.mtr_dist_col, 
                            color=constants.price_col, 
                            hover_data = [constants.price_col],
                            color_continuous_scale = "rainbow")

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    fig.show()

def correlation_matrix_eda(data):
    ''' Function to construct correlation matrix between variables '''
    # Correlation matrix
    correlation_matrix = data.corr()

    # Plotting the correlation matrix
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix')
    plt.show()

# ------------------------------------------


# Models prep 
def data_sep_train_test(data):
    ''' Function to separate dataset into train and test '''
    # Selecting features and target variable
    features = [constants.mtr_dist_col, constants.convin_str_col, constants.lat_col, constants.lon_col]
    target = constants.price_col

    X = data[features]
    y = data[target]

    # Splitting the dataset into training and testing sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

def data_normalization(X_train, X_test):
    ''' Normalization for train and test data'''
    #Normalize the data
    scaler = StandardScaler()

    return scaler.fit_transform(X_train), scaler.transform(X_test)

def plot_output_pred(y_test, y_pred_lr):
    ''' Plot output vs predicted '''
    # Visualization: Actual vs. Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_lr, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs. Predicted House Prices')
    plt.show()


# Models 
## Linear regression
def linear_regression_train(X_train_scaled, y_train):
    ''' Train linear regression '''
    # Model initialization
    model = LinearRegression()

    # Training the model
    model.fit(X_train_scaled, y_train)
    return model

## GridSearch and train model
def grid_search_train(model, param_grid, kf, X_train_scaled, y_train):
    ''' Train model with gridsearch '''
    best_model = GridSearchCV(model,
                        param_grid,
                        cv=kf,
                        scoring='neg_mean_absolute_error',
                        return_train_score=True,
                        refit=True)

    best_model.fit(X_train_scaled, y_train)

    return best_model

## Select best model with cross validation
def cross_val_models(models, X_train, y_train):
    '''Calculate the metrics for each model'''
    # dictionary to hold the evaluation metrics for each model
    results = {}

    # train and evaluate each model
    for name, model in models.items():
        print(f"MODEL: {name}")
        pipe = Pipeline(
        [
            ("scalar", StandardScaler()),
            ("estimator",model),
        ],
        verbose=False,
        )
        
        CV = 5

        scores_MAE = cross_val_score(pipe, X_train, y_train, scoring=make_scorer(mean_absolute_error), cv=CV)
        scores_R2 = cross_val_score(pipe, X_train, y_train, scoring=make_scorer(r2_score), cv=CV)
        scores_MSE = cross_val_score(pipe, X_train, y_train, scoring=make_scorer(mean_squared_error), cv=CV)
        
        
        # storing the metrics
        results[name] = {"MAE": scores_MAE.mean(),
                        "MAE std": scores_MAE.std(),
                        "R2": scores_R2.mean(),
                        "R2 std":scores_R2.std(),
                        "MSE": scores_MSE.mean(),
                        "MSE std":scores_MSE.std()
                        }

    return results

