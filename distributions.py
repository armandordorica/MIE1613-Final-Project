import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.stats import probplot
from scipy.stats import gamma
from scipy.stats import expon
from scipy.stats import lognorm


from scipy.stats import  norm






def generate_qq_plot_normal(data, title='QQ Plot', label='input_data'):
    # Calculate the z-scores of the sorted input data
    sorted_data = np.sort(data)
    z_scores = (sorted_data - np.mean(data)) / np.std(data)
    
    # Generate theoretical quantiles
    quantiles = np.array([(i - 0.5) / len(data) for i in range(1, len(data) + 1)])
    theoretical_quantiles = stats.norm.ppf(quantiles)
    
    # Plot the QQ plot
    plt.figure(figsize=(8, 8))
    plt.plot(theoretical_quantiles, z_scores, 'o', label=label)
    plt.plot(theoretical_quantiles, theoretical_quantiles, 'r', label='Standard Normal')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig("plots/" +title +".png")
    

def get_cdf_df(input_df, categorical_var): 
    temp_df = input_df.groupby(categorical_var).size().reset_index(name='count')
    temp_df.sort_values(by=categorical_var, ascending=True, inplace=True)
    temp_df['cum_sum'] = temp_df['count'].cumsum()
    temp_df['total'] = temp_df['count'].sum()
    temp_df['pct_over_total'] = temp_df['count']/temp_df['total']
    temp_df['cum_pct'] = temp_df['cum_sum']/temp_df['total']
    return temp_df


def get_capped_df_by_categorical_cdf(input_df, category_var,cap_pct=0.90): 
    """
    Summary: Given a dataframe with a categorical variable of interest, return a dataframe with categories that explain `cap_pct` percentage of the total rows 
    
    Input: 
        input_df - dataframe containing the categorical variable, i.e. pin_l2_interest
        category_var - a string describing the name of the categorical variable 
        cap_pct - pct where you want to cap the categories, i.e. a cap_pct of 0.90 
                    will give you a dataframe with only the categories that make up 
                    the first 90% of the entities 
        
    Output: 
        A dataframe with only the categories that make up 
                    the first 90% of the entities 
        
    """
    
    temp_df= input_df.groupby(category_var).size().reset_index(name='count')
    temp_df.sort_values(by='count', ascending=False, inplace=True)
    temp_df['total_count'] = temp_df['count'].sum()
    temp_df['cum_sum'] = temp_df['count'].cumsum()
    temp_df['cum_sum_pct']= temp_df['cum_sum']/temp_df['total_count']
    temp_df = temp_df[temp_df['cum_sum_pct']<cap_pct]
    
    return input_df[input_df[category_var].isin(list(temp_df[category_var].unique()))]
    
    
def get_confidence_interval(input_df, variable_name, title= '', conf_interval=95): 
    cdf_df = get_cdf_df(input_df, variable_name)
    mean = np.mean(input_df[variable_name])

    conf_interval = 95
    tails_width = 100-conf_interval

    tail_width = tails_width/2
    
    low_bound = cdf_df[cdf_df['cum_pct']<=tail_width/100][variable_name].max()
    
    high_bound = cdf_df[cdf_df['cum_pct']<=(100-tail_width)/100][variable_name].max()
    
    plt.figure(figsize=(20,10))
    sns.distplot(input_df[variable_name])
    plt.axvline(x = np.round(low_bound,4), color = 'b', ls='--', label = f'Low bound {np.round(low_bound,4)}')   
    plt.axvline(x = np.round(mean,4), color = 'r', ls='--', label = f'Mean {np.round(mean,4)}') 

    plt.axvline(x = np.round(high_bound,4), color = 'b', ls='--', label = f'High bound {np.round(high_bound,4)}') 
    plt.xlabel(f"Values of {variable_name}")
    
    if title == '':
        title = f"Distribution of {variable_name} with confidence interval bounds of {conf_interval}%"
    else: 
        pass
    plt.title(title, fontsize=20)
    plt.legend()
    plt.show()
    
    
    return low_bound, mean, high_bound
    

def plot_qq_plot(observed_data, fitted_data, plot_title='QQ plot'): 
    # Sort both observed and fitted data
    observed_data_sorted = np.sort(observed_data)
    fitted_data_sorted = np.sort(fitted_data)

    # Compute quantiles
    observed_quantiles = np.arange(1, len(observed_data_sorted) + 1) / (len(observed_data_sorted) + 1)
    fitted_quantiles = np.arange(1, len(fitted_data_sorted) + 1) / (len(fitted_data_sorted) + 1)

    # Interpolate fitted data quantiles at observed data quantiles
    fitted_data_quantiles = np.interp(observed_quantiles, fitted_quantiles, fitted_data_sorted)

    # Plot the QQ plot
    plt.scatter(observed_data_sorted, fitted_data_quantiles)
    plt.plot([min(observed_data_sorted), max(observed_data_sorted)], [min(observed_data_sorted), max(observed_data_sorted)], 'r--', label='Ideal QQ Plot')
    plt.xlabel('Observed Data Quantiles')
    plt.ylabel('Fitted Data Quantiles')
    plt.legend()
    plt.title(plot_title)
    plt.show()
    

def fit_log_normal(data, plot_title=''): 
    # Fit a lognormal distribution to your data
    shape, loc, scale = lognorm.fit(data)

    # Create the lognormal distribution with the fitted parameters
    fitted_lognorm = lognorm(s=shape, loc=loc, scale=scale)

    # Define the range for the x-axis
    x_range = np.linspace(min(data), max(data), 1000)

    # Evaluate the fitted lognormal PDF for each value in the x_range
    pdf_fitted = fitted_lognorm.pdf(x_range)

    # Plot the histogram of the data
    plt.hist(data, bins='auto', density=True, alpha=0.5, label='Data')

    # Plot the fitted lognormal PDF
    plt.plot(x_range, pdf_fitted, label='Fitted Lognormal')
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.title(plot_title)
    plt.legend()
    plt.show()
    
    return pdf_fitted



def get_fitted_pdf_normal(data): 
    mu, sigma = np.mean(data), np.std(data)
    fitted_normal = norm(loc=mu, scale=sigma)
    x_range = np.linspace(min(data), max(data), 1000)
    pdf_fitted = fitted_normal.pdf(x_range)
    return pdf_fitted


def get_fitted_pdf_log_normal(data, plot_title=''): 
    # Fit a lognormal distribution to your data
    shape, loc, scale = lognorm.fit(data)

    # Create the lognormal distribution with the fitted parameters
    fitted_lognorm = lognorm(s=shape, loc=loc, scale=scale)

    # Define the range for the x-axis
    x_range = np.linspace(min(data), max(data), 1000)

    # Evaluate the fitted lognormal PDF for each value in the x_range
    pdf_fitted = fitted_lognorm.pdf(x_range)
    
    return pdf_fitted


def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between the true values and predicted values.

    Parameters:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The RMSE value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse


def plot_pdf_estimation(data, plot_title=''): 

    # Estimating the PDF using Gaussian Kernel Density Estimation (KDE)

    # Estimating the PDF using Gaussian Kernel Density Estimation (KDE)
    kde = gaussian_kde(data)

    # Define the range for the x-axis
    x_range = np.linspace(min(data), max(data), 1000)

    # Evaluate the estimated PDF for each value in the x_range
    pdf_estimation = kde.evaluate(x_range)

    # Normalize the PDF to match the scale of the histogram
    n, _ = np.histogram(data, bins='auto', density=True)
    pdf_estimation_normalized = pdf_estimation * (n.max() / pdf_estimation.max())

    # Plot the histogram
    plt.hist(data, bins='auto', density=True, alpha=0.5, label='Histogram')

    # Plot the estimated PDF, normalized to match the scale of the histogram
    plt.plot(x_range, pdf_estimation_normalized, label='PDF')
    plt.xlabel('Data')
    plt.ylabel('Density')
    plt.legend()
    plt.title(plot_title)
    plt.show()

    
def estimate_pdf(data): 
    kde = gaussian_kde(data)
    # Define the range for the x-axis
    x_range = np.linspace(min(data), max(data), 1000)
    # Evaluate the estimated PDF for each value in the x_range
    pdf_estimation = kde.evaluate(x_range)
    
    return pdf_estimation



def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between the true values and predicted values.

    Parameters:
    y_true (array-like): The true values.
    y_pred (array-like): The predicted values.

    Returns:
    float: The RMSE value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse