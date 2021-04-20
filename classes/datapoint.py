
from sys import exc_info
from os import path
import numpy as np
import random
from matplotlib import pyplot as plt
plt.style.use('seaborn')



class Datapoint:
    """A representation of a single datapoint.

    Class that stores feature vector, actual and predicted labels.
    Total quantity of an elements in the datapoint's feature vector is calculated by
    formula: x_rightmost - x_leftmost - 1

    :Example:

        >>> datapoint = Datapoint()
        >>> datapoint.generate()
        >>> datapoint.show()

    :param x_leftmost: bottom boundary of the datapoint's feature vector
    :param x_rightmost: top boundary of the datapoint's feature vector

    """
    def __init__(self, x_leftmost: int = 1, x_rightmost: int = 102):

        # Check args
        if x_leftmost <= 0:
            raise ValueError(type(self).__name__ + ' args: x_leftmost <= 0')
        elif (x_rightmost - x_leftmost) <= 100:
            raise ValueError(type(self).__name__ + ' args: x_rightmost - x_leftmost <= 100')

        # Input variables (X = x_1, x_2,..., x_n)
        # X's boundaries
        self.__bins_leftmost = x_leftmost
        self.__bins_rightmost = x_rightmost
        # Quantity of features
        self.__bins_quantity = int(self.__bins_rightmost - self.__bins_leftmost)
        self.__bins = np.linspace(self.__bins_leftmost, self.__bins_rightmost, self.__bins_quantity)
        self.__bin_edges = np.histogram_bin_edges(a=self.__bins, bins=self.__bins, range=None, weights=None)
        # Values of features (X)
        self.__feature_vector = np.empty(len(self.__bin_edges) - 1, dtype=np.float64)

        # Output variable (y)
        # Real y
        self.__actual_label = np.nan
        # Estimated y
        self.__predicted_label = np.nan


    @property
    def feature_vector_length(self) -> int:
        '''Get datapoint's feature vector's length.

        :returns: quantity of elements in the feature vector

        '''
        return len(self.__feature_vector)


    @property
    def feature_vector(self) -> np.array:
        '''Get datapoint's feature vector.

        :returns: datapoint's feature vector

        '''
        return self.__feature_vector


    @feature_vector.setter
    def _feature_vector(self, fv: np.array = np.nan) -> None:
        '''Set datapoint's feature vector.

        :param fv: feature vector's value

        :raises ValueError: *fv*'s shape or data type is incorrect w.r.t. datapoint's initial settings

        '''
        # Check compatibility
        if self.__feature_vector.shape != fv.shape:
            raise ValueError(type(self).__name__ + ': incorrect shape of fv')
        if self.__feature_vector.dtype != fv.dtype:
            raise ValueError(type(self).__name__ + ': incorrect data type of fv')
        else:
            self.__feature_vector = fv      


    @property
    def actual_label(self) -> bool:
        '''Get datapoint's real label.

        :returns: datapoint's actual label

        :raises ValueError: datapoint's actual label is not set

        '''
        if (isinstance(self.__actual_label, bool)) != True:
            raise ValueError(type(self).__name__ + ': actual_label type is undefined')
        else:
            return self.__actual_label


    @actual_label.setter
    def _actual_label(self, al: bool = False) -> None:
        '''Set datapoint's real (true) label.

        :param al: actual label's value (False if this is class 0 curve, True if this is class 1 curve)

        '''
        self.__actual_label = bool(al)


    @property
    def predicted_label(self) -> bool:
        '''Get datapoint's determined (estimated) label.

        :returns: datapoint's predicted label

        '''
        if (isinstance(self.__predicted_label, bool)) != True:
            raise ValueError(type(self).__name__ + ': predicted_label type is undefined')
        else:
            return self.__predicted_label


    @predicted_label.setter
    def predicted_label(self, pl: bool = False) -> None:
        '''Set datapoint's determined (estimated) label.

        :param pl: predicted label's value (False if this is class 0 curve, True if this is class 1 curve)

        '''
        self.__predicted_label = bool(pl)


    def features_normalization(self) -> None:
        '''Fit the feature vector's values into [0, 1] interval.

        '''
        np.seterr(divide='raise')

        try:
            self.__feature_vector[self.__feature_vector < 0] = 0
            self.__feature_vector = self.__feature_vector / np.max(self.__feature_vector)

        except FloatingPointError: # handle division by zero
            self.__feature_vector = self.__feature_vector * 0


    def show(self) -> None:
        '''Visualize datapoint's feature vector, actual and predicted labels.

        '''
        # Prepare X axis
        bin_centers = 0.5*(self.__bin_edges[1:] + self.__bin_edges[:-1])

        # Prepare processing status
        if (isinstance(self.__actual_label, bool) != True) or (isinstance(self.__predicted_label, bool) != True):
            plot_title='UNDEFINED'
            curve_color="royalblue"
        elif self.__actual_label == self.__predicted_label: # matched
            plot_title='class ' + str(int(self.__actual_label))+ ' curve'
            curve_color="lime"
        else:
            plot_title='NOT MATCHED'
            curve_color="crimson"

        # Plot setup
        plt.figure(figsize=(7, 7))
        plt.plot(bin_centers, self.__feature_vector,
                 label="actual label is <"+str(self.__actual_label)+">, predicted label is <"+str(self.__predicted_label)+">",
                 color=str(curve_color))
        plt.legend(loc='upper center')
        plt.xlabel('Slot ID')
        plt.ylabel('Value')
        plt.title(str(plot_title))

        plt.show()


    def generate(self) -> int:
        '''Generate datapoint's feature vector (X) and it's actual label (y)
        based on randomly selected shape (curve) from a pool.

        :returns: status (0 if successful)

        '''
        print("datapoint generating...", end="", flush=True)

        # Pair of spikes presence (class 0 curve by default)
        self.__actual_label = False

        # Generate "actual" label and feature vector of a datapoint based on samples from randomly chosen PDF
        samples_quantity = random.randint(25000, 100000)
        chosen_curve = random.randint(0, 7)
        if chosen_curve in range (0, 3): # Gaussian Mixture | Gaussian | Inverse Gaussian
            # Generate randomized X^hat
            gauss_mean_solo = random.randint(self.__bins_leftmost, self.__bins_rightmost)
            # Generate randomized s.d.
            gauss_sigma = np.sqrt(self.__bins_quantity)*np.random.uniform(0, 1)
            # Generate samples from a randomized distribution with noisy mu, sigma and samples' quantity
            if chosen_curve == 0: # Gaussian Mixture with 2 peaks
                # THe class of curves that we want to distinguish from others (class 1)
                self.__actual_label = True
                # Generate pair of means within the required range
                gauss_mean_duo_first = random.randint(self.__bins_leftmost,
                                                      self.__bins_leftmost + int(0.8*self.__bins_quantity))
                gauss_mean_duo_second = gauss_mean_duo_first + random.randint(int(0.2*self.__bins_quantity),
                                                                              int(self.__bins_rightmost - gauss_mean_duo_first))
                # Generate Mixture
                samples_duo_first = np.random.normal(loc=gauss_mean_duo_first,
                                                     scale=gauss_sigma,
                                                     size=int(samples_quantity/2))
                samples_duo_second = np.random.normal(loc=gauss_mean_duo_second,
                                                      scale=gauss_sigma*np.random.uniform(0, 1),
                                                      size=int(samples_quantity/2))
                samples = np.concatenate((samples_duo_first, samples_duo_second), axis=0)

            elif chosen_curve == 1: # Gaussian
                samples = np.random.normal(loc=gauss_mean_solo, scale=gauss_sigma, size=samples_quantity)

            else: # Inverse Gaussian
                samples = np.random.wald(mean=gauss_mean_solo, scale=gauss_sigma, size=samples_quantity)

        elif chosen_curve == 3: # Pareto
            # Shape of the distribution
            pareto_shape = np.random.uniform(0, 1)
            samples = np.random.pareto(a=pareto_shape, size=samples_quantity)

        elif chosen_curve == 4: # Exponential
            exp_beta = np.random.uniform(0, 1000)
            samples = np.random.exponential(scale=exp_beta, size=samples_quantity)

        elif chosen_curve == 5: # Chi-square
            # Number of a degrees of freedom
            chi_df = np.random.uniform(0, 10)
            samples = np.random.chisquare(df=chi_df, size=samples_quantity)

        elif chosen_curve >= 6:
            if chosen_curve == 6: # Sine + Gaussian
                samples_misfitted = np.arcsin(np.random.uniform(-1, 1)) + np.random.normal(scale=np.random.uniform(0, 1),
                                                                                           size=len(self.__bins))
            else: # Noisy sine
                samples_misfitted = np.arcsin(np.random.uniform(np.random.uniform(-1,0),
                                                                np.random.uniform(0, 1),
                                                                size=samples_quantity))
            samples = ((samples_misfitted / np.max(samples_misfitted)) * (self.__bins_quantity/2)) + int(self.__bins_quantity/2)

        # Compute the occurrences of input data that fall within each bin
        self.__feature_vector, bin_edges = np.histogram(samples, bins=self.__bins, density=True)
        self.features_normalization()

        # Additional randomizations
        # Invert horizontally
        if random.randint(0, 1) == 1:
            self.__feature_vector = np.flip(self.__feature_vector, 0)
        # Compress vertically
        if random.randint(0, 1) == 1:
            self.__feature_vector = self.__feature_vector * np.random.uniform(0.5, 1)

        print("...OK")
        return 0


    def save(self, filepath_without_extension: str) -> None:
        '''Save the datapoint to a storage.

        :param filepath_without_extension: the path where the datapoint's file will be saved

        '''
        print("Datapoint saving...", end="", flush=True)
        np.savez_compressed(filepath_without_extension, left_x=np.array([self.__bins_leftmost]),
                                    right_x=np.array([self.__bins_rightmost]),
                                    fv = self.__feature_vector,
                                    al=np.array([self.__actual_label]), 
                                    pl=np.array([self.__predicted_label]))
        print("...OK")


    @classmethod
    def load(cls, filepath_without_extension: str, need_normalize: bool = False):
        '''Load existing datapoint from a disk.

        :param filepath_without_extension: the path where the datapoint's file will be taken
        :param need_normalize: the need to normalize feature vector's values

        :returns: datapoint loaded from a storage
        :rtype: Datapoint

        :raises IOError: if provided by *filepath_without_extension*.npz file is unreadable or does not exist
        :raises ValueError: the *filepath_without_extension*.npz contains an object array

        '''
        print("Datapoint loading...", end="", flush=True)
        if path.exists(filepath_without_extension + ".npz") == True:
            loaded_data = np.load(file=filepath_without_extension + ".npz", allow_pickle=False)
        else:
            print("...NO FILE")
            tb = exc_info()[2]
            raise IOError(cls.__name__ + ': cannot load file' + filepath_without_extension + '.npz').with_traceback(tb)

        dp = Datapoint(loaded_data['left_x'][0], loaded_data['right_x'][0])
        dp._feature_vector = loaded_data['fv']

        dp._actual_label = loaded_data['al'][0]
        dp.predicted_label = loaded_data['pl'][0]

        if need_normalize == True:
            dp.features_normalization()

        print("...OK")
        return dp

