'''classifer.py
Generic classifier data type
AIKINS ACHEAMPONG
CS 251: Data Analysis and Visualization
Fall 2024
'''
import numpy as np

class Classifier:
    '''Parent class for classifiers'''
    def __init__(self, num_classes):
        '''
        
        TODO:
        - Add instance variable for `num_classes`
        '''
        self.numClasses = num_classes

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''
        correct = np.sum(y == y_pred)
        accuracy = correct/len(y)
        return accuracy
    
    def confusion_matrix(self, y, y_pred):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        matrix = np.zeros((self.numClasses, self.numClasses), dtype=int)
        for true_label, pred_label in zip(y, y_pred):
            matrix[true_label, pred_label] += 1
        return matrix

    def train(self, data, y):
        '''Every child should implement this method. Keep this blank.'''
        pass

    def predict(self, data):
        '''Every child should implement this method. Keep this blank.'''
        pass