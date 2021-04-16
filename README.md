# Hydra
* _Purpose:_ recognition of a pair of bursts.
* _Usage:_ **hydra** <ins>X_LEFTMOST</ins> <ins>X_RIGHTMOST</ins>
* _PARAMETERS:_

    **X_LEFTMOST** bottom boundary of the examined graph's slots.<br/>
	**X_RIGHTMOST** top boundary of the examined graph's slots.<br/>


---


* _Example:_

hydra 1 102<br/>


---

* _Description:_ from different, somehow distributed normalized graphs (datapoints) **Hydra** can determine curves with a pair of spikes (two peaks/bursts) using a deep learning architecture of a trained multi-layer perceptron. The number of features (slots, bins, sectors, disjoint categories) is greater than or equal to 100 for each datapoint. Those slots form the X axis of the graph. The value of each normalized feature lies in the range from 0 to 1. These values form the graph's Y axis. The graph's X and Y in combination represent some shape (curve) (in other words, a feature vector with (**X_RIGHTMOST** - **X_LEFTMOST** - 1) elements). For this task, **Hydra** randomly generates a dataset with datapoints (feature vectors and their corresponding True(1)/False(0) labels) based on sampling from a different PDFs, trains a model on this dataset and labels a new given datapoint based on the model's learning expirience. Saves/Loads a new datapoint and/or a trained model for future use, displays the processed datapoint.

