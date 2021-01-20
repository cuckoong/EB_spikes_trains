# EB_spikes_trains
Python Code for Paper: An Efficient and Flexible Spike Train Model via Empirical Bayes


## [simulation-experiment.ipynb](https://github.com/cuckoong/EB_spikes_trains/blob/master/simulation-experiment.ipynb)
Estimate spike counts of the simulation data using the 'SODS' model

## [experimental data.ipynb](https://github.com/cuckoong/EB_spikes_trains/blob/master/experimental%20data.ipynb)
Estimate neural network between the real retinal cells using the 'SODS' model.
the real retinal cell from the [ret-1 dataset](http://crcns.org/data-sets/retina/ret-1).
We have preprocessed the dataset based on the [instruction](http://crcns.org/data-sets/retina/ret-1/about-ret-1) using matlab, and save the experimental data as [.mat version](https://github.com/cuckoong/EB_spikes_trains/tree/master/exerimental_data).

## [experimental_data_draw_network.ipynb](https://github.com/cuckoong/EB_spikes_trains/blob/master/experimental_data_draw_network.ipynb)
Visualize the neural network based on the experimental data



# package required: 
python>=3.6 \
pystan=2.19.1 \
matplolib=3.2.1 \
scipy=1.4.1 \
numpy=1.18.3 \
pandas=1.0.3 \
scikit-learn=0.23.1 \
jupyter \
sklearnSciencePlots=1.0.3      
