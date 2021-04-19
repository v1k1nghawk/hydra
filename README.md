# Hydra
* _Purpose:_ recognition of a curve with a pair of bursts.
* _Usage:_ **inferencer.py** <ins>X_LEFTMOST</ins> <ins>X_RIGHTMOST</ins>
* _PARAMETERS:_

    **X_LEFTMOST** bottom boundary of the examined graph's slots.<br/>
    **X_RIGHTMOST** top boundary of the examined graph's slots.<br/>


---


* _Description:_

From different, somehow distributed normalized graphs (datapoints) **Hydra** can determine curves with a pair of spikes (two peaks/bursts) using a deep learning architecture of a multi-layer perceptron. The number of features (slots, bins, sectors, disjoint categories) must be greater than or equal to 100 for each datapoint. Those slots form the X axis of the graph. The value of each normalized feature lies in the range from 0 to 1. These values form the graph's Y axis. The graph's X and Y in combination represent some shape (curve) (in other words, a feature vector with (**X_RIGHTMOST** - **X_LEFTMOST** - 1) elements).</ins>

For its task **Hydra** randomly generates a dataset with datapoints (feature vectors and their corresponding True(1)/False(0) labels) based on sampling from a number of probability density functions, trains a model on this dataset, and labels a new given datapoint (makes prediction) based on the model's training. Saves/Loads the new datapoint (in the NPZ format) and/or the trained model (in the TensorFlow SavedModel format) for future use, displays the processed datapoint.</ins>


* _Software used during development:_ Python 3.8.7, Keras 2.4.3, Numpy 1.19.5, Matplotlib 3.3.4


---


* _Call example:_

inferencer.py 1 102<br/>


* _Prediction examples:_


| Two Bursts Curves      | Other Curves      |
|------------|-------------|
| <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_TRUE01.png" width="400"> | <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_FALSE01.png" width="400"> |
| <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_TRUE02.png" width="400"> | <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_FALSE02.png" width="400"> |
| <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_TRUE03_cover.png" width="400"> | <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_FALSE03.png" width="400"> |
| <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_TRUE04.png" width="400"> | <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_FALSE04.png" width="400"> |
| <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_TRUE05.png" width="400"> | <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_FALSE05.png" width="400"> |
| <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_TRUE06.png" width="400"> | <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_FALSE06.png" width="400"> |
| <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_TRUE07.png" width="400"> | <img src="https://github.com/v1k1nghawk/hydra/blob/media/plot_FALSE07.png" width="400"> |
