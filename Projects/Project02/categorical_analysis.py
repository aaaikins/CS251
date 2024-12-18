'''categorical_analysis.py
Run analyses with categorical data
Aikins Acheampong
CS 251: Data Analysis and Visualization
Fall 2024
'''
import numpy as np

import analysis

from categorical_data import CatData

import matplotlib.pyplot as plt

import charts

class CatAnalysis(analysis.Analysis):
    def __init__(self, data):
        '''CatAnalysis constructor

        (This method is provided to you and should not require modification)

        Parameters:
        -----------
        data: `CatData`.
            `CatData` object that stores the dataset.
        '''
        super().__init__(data)

    def cat_count(self, header):
        '''Counts the number of samples that have each level of the categorical variable named `header`

        Example:
            Column of self.data for `cat_var1`: [0, 1, 2, 0, 0, 1, 0, 0]
            This method should return `counts` = [5, 2, 1].

        Parameters:
        -----------
        header: str. Header of the categorical variable whose levels should be returned.

        Returns:
        -----------
        ndarray. shape=(num_levels,). The number of samples that have each level of the categorical variable named `header`
        list of strs. len=num_levels. The level strings of the categorical variable  `header` associated with the counts.

        NOTE:
        - Your implementation should rely on logical indexing. Using np.unique is not allowed here.
        - A single loop over levels is totally fine here.
        - `self.data` stores categorical levels as INTS so it is helpful to work with INT-coded levels when doing the counting.
        The method should, however, return the STRING-coded levels (e.g. for plotting).
        '''
        counts = []

        string_levels = self.data.get_cat_levels_str(header)

        column_index = self.data.header2col[header]

        int_levels = self.data.get_cat_levels_int(header)

        for int_level in int_levels:
            level_mask = self.data.data[:, column_index] == int_level

            count = self.data.data[level_mask, column_index].size

            counts.append(count)
            
        return np.array(counts), string_levels

    def cat_mean(self, numeric_header, categorical_header):
        '''Computes the mean of values of the numeric variable `numeric_header` for each of the different categorical
        levels of the variable `categorical_header`.

        POSSIBLE EXTENSION. NOT REQUIRED FOR BASE PROJECT

        Example:
            Column of self.data for `numeric_var1` = [4, 5, 6, 1, 2, 3]
            Column of self.data for `cat_var1` = [0, 0, 0, 1, 1, 1]

            If `numeric_header` = "numeric_var1" and `categorical_header` = "cat_var1", this method should return
            `means` = [5, 2].
            (1st entry is mean of all numeric var values with corresponding int level of 0,
             2nd entry is mean of all numeric var values with corresponding int level of 1)

        Parameters:
        -----------
        numeric_header: str. Header of the numeric variable whose values should be averaged.
        categorical_header: str. Header of the categorical variable whose levels determine which values of the
            numeric variable that should be averaged.

        Returns:
        -----------
        ndarray. shape=(num_levels,). Means of values of the numeric variable `numeric_header` for each of the different
            categorical levels of the variable `categorical_header`.
        list of strs. len=num_levels. The level strings of the categorical variable  `categorical_header` associated with
            the counts.

        NOTE:
        - Your implementation should rely on logical indexing. Using np.unique is not allowed here.
        - A single loop over levels is totally fine here.
        - As above, it is easier to work with INT-coded levels, but the STRING-coded levels should be returned.
        - Since your numeric data has nans in it, you should use np.nanmean, which ignores any nan values. Otherwise, the
        according to np.mean, the mean of any collection of numbers that include at least one nan will always be nan.
        You can easily swap np.mean with np.nanmean: https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html
        '''
        means = []

        string_levels = self.data.get_cat_levels_str(categorical_header)
        int_levels = self.data.get_cat_levels_int(categorical_header)
        
        numeric_col = self.data.header2col[numeric_header]
        categorical_col = self.data.header2col[categorical_header]

        for int_level in int_levels:
            level_mask = self.data.data[:, categorical_col] == int_level
            numeric_values = self.data.data[level_mask, numeric_col]
 
            mean_value = np.nanmean(numeric_values)
            means.append(mean_value)
        
        return np.array(means), string_levels


    def cat_count2(self, header1, header2):
        '''Counts the number of samples that have all combinations of levels coming from two categorical headers
        (`header1` and `header2`).

        POSSIBLE EXTENSION. NOT REQUIRED FOR BASE PROJECT

        Parameters:
        -----------
        header1: str. Header of the first categorical variable
        header2: str. Header of the second categorical variable

        Returns:
        -----------
        ndarray. shape=(header1_num_levels, header2_num_levels). The number of samples that have each combination of
            levels of the categorical variables `header1` and `header2`.
        list of strs. len=header1_num_levels. The level strings of the categorical variable  `header1`
        list of strs. len=header2_num_levels. The level strings of the categorical variable  `header2`

        Example:

        header1_level_strs: ['a', 'b']
        header2_level_strs: ['y', 'z']

        counts =
                [num samples with header1 value 'a' AND header2 value 'y', num samples with header1 value 'a' AND header2 value 'z']
                [num samples with header1 value 'b' AND header2 value 'y', num samples with header1 value 'b' AND header2 value 'z']

        NOTE:
        - To combine two logical arrays element-wise, you can use the & operator or np.logical_and
        '''
        string_levels_1 = self.data.get_cat_levels_str(header1)
        int_levels_1 = self.data.get_cat_levels_int(header1)
        string_levels_2 = self.data.get_cat_levels_str(header2)
        int_levels_2 = self.data.get_cat_levels_int(header2)
        
        col_1 = self.data.header2col[header1]
        col_2 = self.data.header2col[header2]


        counts = np.zeros((len(int_levels_1), len(int_levels_2)), dtype=int)

        for i, int_level_1 in enumerate(int_levels_1):
            for j, int_level_2 in enumerate(int_levels_2):
                level_mask = (self.data.data[:, col_1] == int_level_1) & (self.data.data[:, col_2] == int_level_2)
                counts[i, j] = np.sum(level_mask)
        
        return counts, string_levels_1, string_levels_2


    def cat_var(self, numeric_header, categorical_header):
        """
        Computes the standard deviation of values of the numeric variable `numeric_header` for each level of `categorical_header`.

        Parameters:
        -----------
        numeric_header: str. Header of the numeric variable to compute the standard deviation for.
        categorical_header: str. Header of the categorical variable to group by.

        Returns:
        -----------
        ndarray. Standard deviation values of the numeric variable for each categorical level.
        list of strs. Corresponding string levels of the categorical variable.
        """
        variances = []
        string_levels = self.data.get_cat_levels_str(categorical_header)
        int_levels = self.data.get_cat_levels_int(categorical_header)
        numeric_col = self.data.header2col[numeric_header]
        categorical_col = self.data.header2col[categorical_header]

        for int_level in int_levels:
            level_mask = self.data.data[:, categorical_col] == int_level
            numeric_values = self.data.data[level_mask, numeric_col]
            
            std_value = np.nanstd(numeric_values)
            variances.append(std_value)
        
        return np.array(variances), string_levels
    
    
    def sideBarplot(self, animals, years, count):
        x = np.arange(len(years))
        width = 0.3

        fig, ax = plt.subplots(layout='constrained')
        
        for i in range(len(animals)):
            offset = i * width
            bar = ax.barh(x + offset, count[i], width, label=animals[i])
            ax.bar_label(bar, padding=2, rotation=0, fontsize=10)

        ax.set_xlabel('Counts of animal type')
        ax.set_ylabel('Year')
        ax.set_yticks(x + (width * (len(animals) - 1) / 2), years, fontsize=11)

        plt.legend()
        plt.show()