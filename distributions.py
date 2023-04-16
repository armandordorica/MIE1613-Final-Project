import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    
    
def get_confidence_interval(input_df, variable_name, conf_interval=95): 
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
    plt.title(f"Distribution of {variable_name} with confidence interval bounds of {conf_interval}%", fontsize=20)
    plt.legend()
    plt.show()
    
    
    return low_bound, mean, high_bound
    
