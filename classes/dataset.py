
import numpy as np
from threading import Thread
from threading import Semaphore
from .datapoint import Datapoint



class Dataset:
    """A datapoints' storage divided into three chunks.

    The dataset consists of train, validation and test subsets.
    Total quantity of an elements in each datapoint's feature vector is calculated by
    formula: x_rightmost - x_leftmost - 1

    :Example:

        >>> dataset = Dataset()
        >>> dataset.generate()

    :param x_leftmost: bottom boundary of each datapoint's feature vector
    :param x_rightmost: top boundary of each datapoint's feature vector

    """
    def __init__(self, x_leftmost: int = 0, x_rightmost: int = 102, size: int = 10000):

        # Check args
        if x_leftmost <= 0:
            raise ValueError(type(self).__name__ + ' args: x_leftmost <= 0')
        elif (x_rightmost - x_leftmost) <= 100:
            raise ValueError(type(self).__name__ + ' args: x_rightmost - x_leftmost <= 100')
        elif (isinstance(size, int) != True) or size < 0:
            raise ValueError(type(self).__name__ + ' args: incorrect size')

        # Apply args
        # Quantity of datapoints in the dataset
        self.__size = size
        # Boundaries of every datapoint's Xs
        self.__bins_leftmost = x_leftmost
        self.__bins_rightmost = x_rightmost

        # Subsets
        self.clear_subsets()


    def __generate_subset(self, subset: np.array) -> None:
        '''Generate datapoints for a specific *subset*.

        Primary usage as a thread target func (for performance improvement).

        :param: subset: a numpy array of Datapoints 

        '''
        self.__gen_in_progress.acquire()

        # Generate subset
        for datapoint_index in range(len(subset)):
            subset[datapoint_index] = Datapoint(self.__bins_leftmost, self.__bins_rightmost)
            subset[datapoint_index].generate()

        self.__gen_in_progress.release()


    def generate(self) -> int:
        '''Generate a whole dataset with randomly generated datapoints, assigned to 3 subsets
        (train subset, validation subset, test subset).

        :returns: status (0 if successful)

        '''
        print("dataset generating...")

        # Clean subsets
        self.clear_subsets()

        # Enable independent multithreaded generation of the subsets
        subsets = [self.__train_subset, self.__valid_subset, self.__test_subset]
        generators = map(lambda subset :
                         Thread(target = self.__generate_subset, args =(subset, ), daemon=True),
                         subsets)

        # Start subsets' generation
        for generator in generators: generator.start()

        # Wait for completion
        for i in range(len(subsets)):
            self.__gen_in_progress.acquire()

        print('dataset has been generated\n')

        return 0


    def __get_vectorized_X_Y(self, subset: np.array):
        '''Represent a given *subset* of datapoints as an input matrix (X) and an output vector (Y).

        :param: subset: a numpy array of Datapoints

        :returns: the tuple (features, labels), where features is a nxd numpy matrix,
        and labels is a length-n vector of True/False labels.
        :rtype: Tuple

        '''

        # Quantity of datapoints (n)
        num_of_points = len(subset)
        # Quantity of features (d)
        if num_of_points == 0:
            return np.nan, np.nan
        else:
            num_of_features = subset[0].feature_vector_length

        # 2D (Xs) "input data"
        # each row - single datapoint (1, 2, ..., n)
        # each column - feature (1, 2, ..., d)
        feature_matrix = np.empty(shape=(num_of_points, num_of_features), dtype=np.float64)

        # 1D (Y) "target data" (1, 2, ..., n)
        actual_labels_vector = np.empty(num_of_points, dtype=bool)

        # Assemble subset in a vector form
        for i in range(len(subset)):
            feature_matrix[i, :] = subset[i].feature_vector
            actual_labels_vector[i] = subset[i].actual_label

        return feature_matrix, actual_labels_vector


    @property
    def vectorized_X_Y_train(self):
        '''Represent train subset as an input matrix (X) and an output vector (Y).

        :returns: train subset (features, labels), where features is a nxd numpy matrix,
        and labels is a length-n vector of True/False labels.
        :rtype: Tuple

        '''
        return self.__get_vectorized_X_Y(self.__train_subset)


    @property
    def vectorized_X_Y_valid(self):
        '''Represent validation subset as an input matrix (X) and an output vector (Y).

        :returns: validation subset (features, labels), where features is a nxd numpy matrix,
        and labels is a length-n vector of True/False labels.
        :rtype: Tuple

        '''
        return self.__get_vectorized_X_Y(self.__valid_subset)


    @property
    def vectorized_X_Y_test(self):
        '''Represent test subset as an input matrix (X) and an output vector (Y).

        :returns: test subset (features, labels), where features is a nxd numpy matrix,
        and labels is a length-n vector of True/False labels.
        :rtype: Tuple

        '''
        return self.__get_vectorized_X_Y(self.__test_subset)


    def clear_subsets(self) -> int:
        '''Prepare clean storage for all dataset's subsets.

        Standard 60-20-20 dataset splitting is used.

        :returns: status (0 if successful)

        '''
        # Subset for basic model fitting
        self.__train_subset = np.empty(int(self.__size*0.6), dtype=Datapoint)
        # Subset for model's hyperparameters' tuning
        self.__valid_subset = np.empty(int(self.__size*0.2), dtype=Datapoint)
        # Subset for final model's evaluation
        self.__test_subset = np.empty(self.__size - int(self.__size*(0.6 + 0.2)), dtype=Datapoint)

        # Sync overall dataset generator with three subsets' subgenerators
        self.__gen_in_progress = Semaphore(3)

        return 0

