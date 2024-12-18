'''charts.py
Plotting functions for categorical data
Aikins Acheampong
CS 251: Data Analysis and Visualization
Spring 2024
'''
import numpy as np
import matplotlib.pyplot as plt



def sidebarplot(values, labels, title, show_counts=True, figsize=(6, 7), sort_by='na', top_k=None):
    '''Horizontal bar plot with bar lengths `values` (the "x values") with associated labels `labels` on the y axis.

    Parameters:
    -----------
    values: ndarray. shape=(num_levels). Each numeric value to plot as a separate horizontal bar coming out of the y axis.
    labels: list of str. len=num_labels. Labels associated with each bar / numeric value in `values`.
    title: str. Title of the plot.
    show_counts: bool. Whether to show each numeric value as text next to each bar.
    fig_sz: tuple of 2 ints. The width and height of the figure.

    NOTE:
    - Assign the output of plt.barh to a variable named ax. i.e.
        ax = plt.barh(...)
    If show_counts is set to True, then add the code:
        if show_counts:
            plt.bar_label(ax, values)    
    to make the values appear on the plot as text next to each bar.
    - If your values show up next to the bars with many significant digits, add the following line of code to have Numpy
    round each displayed value to the nearest 0.01:
        values = np.round(values, 2)
    '''
    sorted_values, sorted_labels = sort(values, labels, sort_by)
    sorted_values = np.round(sorted_values, 2)
    
    if top_k is not None and top_k < len(sorted_values):
        top_indices = np.argsort(sorted_values)[-top_k:]
        sorted_values = sorted_values[top_indices]
        sorted_labels = sorted_labels[top_indices]
    
    plt.figure(figsize=figsize)
    ax = plt.barh(np.arange(len(sorted_labels)), sorted_values)
    plt.yticks(labels=sorted_labels, ticks=np.arange(len(sorted_labels)))
    plt.title(title)

    if show_counts:
        plt.bar_label(ax, labels=[f'{val}' for val in sorted_values])

def sort(values, labels, sort_by='na'):
    '''Sort the arrays `values` and `labels` in the same way so that corresponding items in either array stay matched up
    after the sort.

    Parameters:
    -----------
    values: ndarray. shape=(num_levels,). One array that should be sorted
    labels: ndarray. shape=(num_levels,). Other array that should be sorted
    sort_by: str. Method by which the arrays should be sorted. There are 3 possible values:
        1. 'na': Keep the arrays as-is, no sorting takes place.
        2. 'value': Reorder both arrays such that the array `values` is sorted in ascending order.
        3. 'label': Reorder both arrays such that the array `labels` is sorted in ascending order.

    Returns:
    -----------
    ndarray. shape=(num_levels,). Sorted `values` array. Corresponding values in `labels` remain matched up.
    ndarray. shape=(num_levels,). Sorted `labels` array. Corresponding values in `values` remain matched up.


    NOTE:
    - np.argsort might be helpful here.
    '''
    if sort_by == 'value':
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]
    elif sort_by == 'label':
        sorted_indices = np.argsort(labels)
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]
    else:
        sorted_values = values
        sorted_labels = labels
    
    return sorted_values, sorted_labels


def grouped_sidebarplot(values, header1_labels, header2_levels, title, figsize=(6, 7)):
    '''Horizontal side-by-side bar plot with bar lengths `values` (the "x values") with associated labels.
    `header1_labels` are the levels of `header`, which appear on the y axis. Each level applies to ONE group of bars next
    to each other. `header2_labels` are the levels that appear in the legend and correspond to different color bars.

    POSSIBLE EXTENTION. NOT REQUIRED FOR BASE PROJECT

    (Useful for plotting numeric values associated with combinations of two categorical variables header1 and header2)

    Parameters:
    -----------
    values: ndarray. shape=(num_levels). Each numeric value to plot as a separate horizontal bar coming out of the y axis.
    labels: list of str. len=num_labels. Labels associated with each bar / numeric value in `values`.
    title: str. Title of the plot.
    show_counts: bool. Whether to show each numeric value as text next to each bar.
    fig_sz: tuple of 2 ints. The width and height of the figure.

    Example:
    -----------
    header1_labels = ['2020', '2021']
    header2_labels = ['Red', 'Green', 'Blue']
    The side-by-side bar plot looks like:

    '2021' 'Red'  ----------
     y=2   'Green'------
           'Blue' ----------------

    '2020' 'Red'  -------------------
     y=1   'Green'-------------------
           'Blue' ---------

    In the above example, the colors also describe the actual colors of the bars.

    NOTE:
    - You can use plt.barh, but there are offset to compute between the bars...
    '''
    num_groups = len(header1_labels)
    num_subgroups = len(header2_levels)
    
    
    bar_width = 0.8 / num_subgroups
    indices = np.arange(num_groups)
    
    plt.figure(figsize=figsize)
    
    for i, level in enumerate(header2_levels):
        bar_positions = indices - 0.4 + (i + 0.5) * bar_width
        plt.barh(bar_positions, values[:, i], height=bar_width, label=level)
    
    plt.yticks(ticks=indices, labels=header1_labels)
    plt.title(title)
    plt.legend(title='Categories')
    plt.xlabel('Values')
    plt.ylabel('Groups')

    plt.show()
