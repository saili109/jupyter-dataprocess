from matplotlib import pyplot 
import seaborn
import numpy as np


CAT_FRAC_THRESHOLD = 0.5
CAT_NUM_THRESHOLD = {'object': 1000, 'int64': 5, 'float64': 10}


def describe_column(column):
    """Calculate descriptives for a column"""
    col_dtype = str(column.dtype)
    null_fraction = column.isnull().sum() / float(len(column))
    null_percentage = str(round(null_fraction*100, 1)) + '%'
    unique = len(column.value_counts())
    
    notnull = column.notnull().sum()
    is_categorical = 'Null'
    if col_dtype in ['object']:
        is_categorical = 'Categorical'
    elif col_dtype in ['int64', 'float64'] and notnull > 0:
        judge_categorical = (unique / notnull <= CAT_FRAC_THRESHOLD and
                          unique <= CAT_NUM_THRESHOLD[col_dtype])
        is_categorical = 'Discrete' if judge_categorical else 'Continuous'

    if col_dtype in ['int64', 'float64']:
        minimum = str(round(column.min(),2))
        maximum = str(round(column.max(),2))
        median = str(round(column.median(),2))
        mean = str(round(column.mean(),2))
        std = str(round(column.std(),2))
    else:
        minimum = maximum = mean = median = std = 'Null'
        
    col_descriptives_dict = {
        "dtype": col_dtype,
        "null_percentage": null_percentage,
        'distinct': str(unique),
        "categorical": is_categorical,
        
        'minimum':minimum,
        'maximum':maximum,
        "median": median,
        'mean': mean,
        "std": std
    }
    return col_descriptives_dict


def plot_distribution(column, title, is_categorical = None):
    """Plot distribution of a column"""
    description = describe_column(column)
    fig, ax = pyplot.subplots()
    
    if is_categorical is not None:
        categorical_value = is_categorical
    else:
        categorical_value = description['categorical']

    if categorical_value == "Continuous":
        col_plot = column[~np.isnan(column)]
        seaborn.distplot(col_plot)
        ax.set(xlabel=column.name, ylabel='Distribution(kde)', title=title)
    elif categorical_value == "Categorical" or categorical_value == "Discrete":
        try:
            column=round(column, 2)
        except:
            pass
        seaborn.countplot(x=column, palette="GnBu_d")
        ax.set(xlabel=column.name, ylabel='Count', title=title)
    ax.yaxis.label.set_fontsize(14)
    ax.xaxis.label.set_fontsize(14)
    ax.title.set_fontsize(18)
    
    ax.figure.tight_layout()
    return fig
