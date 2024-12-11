# Import the Libraries needed
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import skew, kurtosis
import numpy as np


# Function to plot the fertility rate trend for a specific country
def LinePlot_Cnt(df_fert, country, start_year, end_year):
    """
    Creates a line plot for fertility rates over a range of years for a
    specific country.

    Parameters:
    - df_fert: pandas DataFrame, dataset containing fertility data
    - country: str, name of the country
    - start_year: int, start year for the trend
    - end_year: int, end year for the trend

    Returns:
    - Generates a Line Plot
    """
    # Generate a list of years within the specified range
    years = list(range(start_year, end_year + 1))
    # Filter the dataset for the specified country
    country_data = df_fert[df_fert['Country Name'] == country]
    # Create a DataFrame to for the specified years
    trend_data = pd.DataFrame({
        'Year': years,  
        'Fertility Rate': country_data[[str(year) for year in years]]
        .values.flatten()
    })
    # Remove any rows with missing data
    trend_data.dropna(inplace=True)
    # Plot the line for fertility rates over time
    plt.figure(figsize=(10, 6))
    plt.plot(trend_data['Year'], trend_data['Fertility Rate']
             , linestyle='-', color='b')
    # Add title and labels to the plot
    plt.title(f'Fertility Rate Trend: {country}', fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Fertility rate, total (births per woman)', fontsize=14)
    # Adjust the layout
    plt.tight_layout()
    # Display the plot
    plt.show()


# Function to create a bar plot
def BarPlot_Top10(df_fert, year):
    """
    Creates a bar plot for fertility rates of the top 10 countries in a given
    year.

    Parameters:
    - df_fert: pandas DataFrame, dataset containing fertility data
    - year: int, the year for which the bar plot will be created

    Returns:
    - Generates a Bar Plot
    """
    # Convert the year to string
    year_column = str(year)
    # Filter the dataset to get the top 10 countries
    top_10_FertRate = df_fert[['Country Name'
                               , year_column]].sort_values(by=year_column
                                            , ascending=False).head(10)
    # Create a bar plot for the top 10 countries
    plt.figure(figsize=(12, 6))
    plt.bar(top_10_FertRate['Country Name'], top_10_FertRate[year_column]
            , color='skyblue')
    # Add title and labels to the plot
    plt.title(f'Top 10 Fertility Rates in {year}', fontsize=16)  
    plt.xlabel('Country', fontsize=14) 
    plt.ylabel('Fertility rate, total (births per woman)', fontsize=14)
    # Rotate the x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=14)
    # Adjust the layout 
    plt.tight_layout()
    # Display the plot
    plt.show()
    

# Function to create a box plot for the distribution of fertility
def boxplot_yr_OL(df_fert, year):
    """
    Creates a box plot for the distribution 
    of fertility rates in a given year.

    Parameters:
    - df_fert: pandas DataFrame, dataset containing fertility data
    - year: int, the year for which the box plot will be created

    Returns:
    - Generates a Box Plot
    """
    # Convert the year to a string
    year_column = str(year)
    # Extract fertility rates for the year
    fertility_rates = df_fert[year_column].dropna()
    # Create a box plot for the fertility rate
    plt.figure(figsize=(8, 6))
    plt.boxplot(fertility_rates, vert=False, patch_artist=True
                , boxprops=dict(facecolor='blue'))
    # Add title and labels to the plot
    plt.title(f'Fertility Rate Distribution in {year}', fontsize=16)
    plt.xlabel('Fertility rate, total (births per woman)', fontsize=14)
    # Adjust the layout
    plt.tight_layout()
    # Display the plot
    plt.show()


# Function to perform KMeans clustering and print silhouette scores
def Clust_and_Shill_Scores(df_fert, year):
    """
    Performs clustering and prints the silhouette scores
    for each cluster count.

    Parameters:
    - df_fert: pandas DataFrame, dataset containing fertility data
    - year: int, the year for which clustering will be performed

    Returns:
    - best_k: int, optimal number of clusters based on silhouette score
    - labels: numpy array, cluster labels for each country
    - silhouette_scores: list of silhouette scores for each k
    - centers: numpy array, cluster centers in the original scale
    """
    # Convert the year to a string
    year_column = str(year)
    # Select the data for the specified year
    fertility_rates = df_fert[['Country Name', year_column]].dropna()
    fertility_values = fertility_rates[[year_column]]
    # Standardize the data for clustering
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(fertility_values)
    # Initialize lists to store silhouette scores
    silhouette_scores = []
    all_labels = []
    all_centers = []
    # Print header for silhouette scores
    print("Silhouette Scores for each cluster count:")
    # Iterate over the range of cluster numbers (k) to evaluate clustering
    for k in range(2, 10 + 1):
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(scaled_data)
        # Compute the silhouette score for the current clustering
        sil_score = silhouette_score(scaled_data, labels)
        # Store the results
        silhouette_scores.append(sil_score)
        all_labels.append(labels)
        # Transform cluster centers back to the original scale
        all_centers.append(scaler.inverse_transform(kmeans.cluster_centers_))
        # Print the silhouette score for the current k
        print(f"{k} clusters: Silhouette Score = {sil_score:.4f}")
    # Identify the optimal number of clusters
    best_k = 2 + np.argmax(silhouette_scores)
    best_labels = all_labels[best_k - 2]
    best_centers = all_centers[best_k - 2]
    return best_k, best_labels, silhouette_scores, best_centers


# Function to compute basic statistics and a correlation matrix
def compute_statistics(df_fert, year):
    """
    Computes mean, median, standard deviation, skewness, and kurtosis for a
    given year. Also generates a correlation matrix and basic summary
    statistics.

    Parameters:
    - data: pandas DataFrame, dataset containing fertility data
    - year: int, the year for which statistics will be computed

    Returns:
    - stats_dict: dict, containing all computed statistics
    - corr_matrix: pandas DataFrame, correlation matrix of the dataset
    """
    year_column = str(year)
    fertility_values = df_fert[year_column].dropna()

    # Compute statistical values of the data
    mean_val = fertility_values.mean()
    median_val = fertility_values.median()
    std_val = fertility_values.std()
    skew_val = skew(fertility_values)
    kurtosis_val = kurtosis(fertility_values)
    #Display the Summary stats
    print("Summary Statistics (describe):")
    print(fertility_values.describe())
    # Compute correlation matrix
    numeric_data = df_fert.select_dtypes(include=['float64', 'int64'])
    corr_matrix = numeric_data.corr()
    # Creating the dictionary with the values calculated
    stats_dict = {
        "Mean": mean_val,
        "Median": median_val,
        "Standard Deviation": std_val,
        "Skewness": skew_val,
        "Kurtosis": kurtosis_val
    }
    return stats_dict, corr_matrix


def plot_Clusters(df_fert, year, labels, centers,
                                x_feature='Fertility Rate', y_feature=None):
    """
    Plots the clustered results from K-Means clustering in 2D.

    Parameters:
    - data: pandas DataFrame, original dataset containing fertility data
    - year: int, the year for which clustering is performed
    - labels: numpy array, cluster labels for each country
    - centers: numpy array, cluster centers in the original scale
    - x_feature: str, the feature for the x-axis (default: 'Fertility Rate')
    - y_feature: str or None, 
      optional feature for the y-axis. If None, applies
      jittering for the y-axis.

    Returns:
    - None: Displays the clustered plot
    """
    year_column = str(year)
    x_data = df_fert[[year_column]].dropna().values.flatten()

    if y_feature:
        y_data = df_fert[y_feature].dropna().values.flatten()
        valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
        x_data = x_data[valid_indices]
        y_data = y_data[valid_indices]
    else:
        np.random.seed(42)
        y_data = np.random.normal(0, 0.1, len(x_data))

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x_data, y_data, c=labels, cmap='Set1', s=50, alpha=0.7,
        edgecolor='k')

    for idx, center in enumerate(centers):
        plt.scatter(center[0], 0 if y_feature is None else center[1],
                    color='black', marker='x', s=200,
                    label=f'Cluster {idx + 1}')

    plt.title(f"Clustered Plot for Fertility Rates in {year}", fontsize=16)
    plt.xlabel('Fertility rate, total (births per woman)', fontsize=14)
    plt.ylabel("Scaled Data of Fertility Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
   
# Polynomial fitting function
def fitting_Fun(x, a, b, c, d):
    """
    Calculates the value of a cubic polynomial at given x.

    Parameters:
    - x (number or array): The value(s) at which the polynomial is evaluated.
    - a, b, c, d (number): Coefficients of the cubic polynomial.

    Returns:
    - number or array: The value of the cubic polynomial at x.
    """
    return a * x**3 + b * x**2 + c * x + d


# Function to calculate error range for polynomial fitting
# Referred from lecture materials
def Confintcal_Fun(x, f, params, cov_matrix):
    """
    Calculates the error range for a polynomial function and its parameters.

    Parameters:
    - x (number or array): The input value(s) at which the error is evaluated.
    - f (function): The polynomial function for which the error is calculated.
    - params (list or array): Coefficients of the polynomial.
    - cov_matrix (array): The covariance matrix of the polynomial parameters.

    Returns:
    - numpy.ndarray: The calculated error values corresponding to each x.
    """
    # Initialize an array of zeros
    var = np.zeros_like(x)
    # Loop through each parameter in the polynomial
    for i in range(len(params)):
        # Calculate the derivative of the polynomial
        deriv1 = derpolyfun(x, f, params, i)
        for j in range(len(params)):
            deriv2 = derpolyfun(x, f, params, j)
            var += deriv1 * deriv2 * cov_matrix[i, j]
    # Return the square root of the variance       
    return np.sqrt(var)


# Function to calculate derivative of polynomial
# Referred from lecture materials
def derpolyfun(x, f, params, index):
    """
    Calculates the derivative of a polynomial function with respect to one
    of its coefficients.

    Parameters:
    - x (number or array): The value(s) at which the derivative is calculated.
    - f (function): The polynomial function for which the derivative is
      calculated.
    - params (list or array): Coefficients of the polynomial.
    - index (int): The index of the coefficient.

    Returns:
    - numpy.ndarray or number: The derivative of the polynomial function.
    """
    val = 1e-6
    # Create an array of zero
    delta = np.zeros_like(params)
    delta[index] = val * abs(params[index])
    # Calculate the 'up' and 'low' of the parameter set
    up = params + delta
    low = params - delta
    diff = 0.5 * (f(x, *up) - f(x, *low))
    return diff / (val * abs(params[index]))


# Read the dataset
fertility_data = pd.read_csv('fertdata_csv.csv')
# Print the info of the dataset
fertility_data.info()
# Plot fertility trend for the United Kingdom
LinePlot_Cnt(fertility_data, 'United Kingdom', 1960, 2022)
# Bar plot for top 10 fertility rates in 2022
BarPlot_Top10(fertility_data, 2022)
# Box plot for fertility rate distribution in 2022
boxplot_yr_OL(fertility_data, 2022)
# calculating Clustering and silhouette scores
best_k, labels, silhouette_scores, centers = Clust_and_Shill_Scores(
    fertility_data, 2022)
print(f"\nOptimal Number of Clusters: {best_k}")
print(f"Cluster Centers: {centers}")
# Plot the elbow plot
plt.figure(figsize=(10, 6))
plt.plot(range(2, 10 + 1), silhouette_scores, marker='o')
# Add titles and axis labels
plt.title('Elbow Plot for Optimal Clusters (Year:2022)', fontsize=16)
plt.xlabel('Number of Clusters', fontsize=14)
plt.ylabel('Sum of Squared Errors (SSE)', fontsize=14)
# Adjust the layout and display the plot
plt.tight_layout()
plt.show()

plot_Clusters(df_fert=fertility_data, year=2022, labels=labels,
                           centers=centers, x_feature='Fertility Rate')

# Compute and display statistics
stats, correlation_matrix = compute_statistics(fertility_data, 2022)
# Display the statistics
print("\nMajor Moments:")
for stat, value in stats.items():
    print(f"{stat}: {value:.4f}")
# Display the Correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Load and transpose UK fertility data
Fertdata_UK = pd.read_csv('fertdatauk.csv')
Fertdata_UK_T = Fertdata_UK.T
# Read the CSV file containing fertility data for the UK  
Fertdata_UK_T.columns = ['FertilityRate']
Fertdata_UK_T = Fertdata_UK_T.drop('Year')
# Reset the index to turn the transposed rows
Fertdata_UK_T.reset_index(inplace=True)  
# Rename the index column to 'Year'
Fertdata_UK_T.rename(columns={'index': 'Year'}, inplace=True) 
Fertdata_UK_T['Year'] = Fertdata_UK_T['Year'].astype(int)
 # Convert 'FertilityRate' to floats
Fertdata_UK_T['FertilityRate'] = Fertdata_UK_T['FertilityRate'].astype(float) 

# Prepare data for fitting
x_val = Fertdata_UK_T['Year'].values.astype(float)  
y_val = Fertdata_UK_T['FertilityRate'].values.astype(float)

# Fitting the polynomial model to the data
popt, pcov = curve_fit(fitting_Fun, x_val, y_val)

# Calculate the confidence intervals for the past data
y_err = Confintcal_Fun(x_val, fitting_Fun, popt, pcov)
# Generate future predictions
fut_x = np.arange(max(x_val) + 1, 2041)
# Predict future fertility rates using the fitted model
fut_y = fitting_Fun(fut_x, *popt)  

# Calculate the confidence intervals for future predictions
y_fut_err = Confintcal_Fun(fut_x, fitting_Fun, popt, pcov)
# Plotting the polynomial fit and predictions
plt.figure(figsize=(10, 6))
# Plot past values with error bars
plt.errorbar(x_val, y_val, yerr=y_err, fmt='go'
             , label='Original Data (with error bars)', 
             ecolor='red', capsize=3)
# Plot the polynomial fit for past data
plt.plot(x_val, fitting_Fun(x_val, *popt), 'b-', label='Polynomial fit')
# Fill the error range for past data
plt.fill_between(x_val, fitting_Fun(x_val, *popt) - y_err,
                 fitting_Fun(x_val, *popt) + y_err, color='red', alpha=0.2,
                 label='Error range of Original values')
# Plot future predictions
plt.plot(fut_x, fut_y, 'b--', label='Future values')
# Fill the error range for future predictions
plt.fill_between(fut_x, fut_y - y_fut_err, fut_y + y_fut_err,
                 color='lightblue', alpha=0.5,
                 label='Error range of Future values')
# Add title and labels
plt.title('Polynomial Fit & Predicting Future for UK Fertility Rates'
          , fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Fertility rate, total (births per woman)', fontsize=14)
plt.legend()
# Display the plot
plt.show()
