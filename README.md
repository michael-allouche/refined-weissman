# refined-weissman
Implementation of the paper ["A refined Weissman estimator of extreme quantile"](https://hal.inria.fr/hal-03266676v2/document),
by [Jonathan El Methni](https://scholar.google.fr/citations?user=JjjH8N8AAAAJ&hl=fr), [Stéphane Girard](http://mistis.inrialpes.fr/people/girard/)

The repo contains the codes for comparing the 8 extreme quantile estimators on both simulated and real-data.

## Abstract
Weissman extrapolation methodology for estimating extreme quantiles from heavy-tailed distribution is based on two estimators: an order statistic to estimate an intermediate quantile and an estimator of the tail-index. 
The common practice is to select the same intermediate sequence for both estimators.
In this work, we show how an adapted choice of two different  intermediate sequences leads to a reduction of the asymptotic bias associated with the resulting refined Weissman estimator. 
The asymptotic normality of the latter estimator is established and a data-driven method is introduced for the practical selection of the intermediate sequences.
Our approach is compared to other bias reduced estimators of extreme quantiles both on simulated and real data.


## Dependencies
Install the requirements for each software version used
- Python 3.8

`pip install -r requirements.txt`
- R 3.

`install.packages("evt0")`

## Usage

### Simulated data
Seven heavy-tailed distributions are implemented in `./extreme/distribution.py`:

**Burr, NHW, Fréchet, Fisher, GPD, Inverse Gamma, Student**.

In `run_evt_estimators.py`, one can update the `dict_runner` with the desired parametrization. 
Next, run `run_evt_estimators.py` to compute all the quantile estimators at both quantile levels $`\alpha=1/(n)`$ and $`\alpha=1/(2n)`$ . 
For example, estimations applied to 1000 replications of 500 samples issued from a Burr distribution:

`python run_evt_estimators.py -d burr -r 1000 -n 500`

Once the run is finished, all the RMSE for each estimator are saved in the folder `./ckpt`.

In the notebook, you can display a table result. For example

```
from extreme.estimators import evt_estimators 
evt_estimators(n_replications=1000, params={"evi":0.125, "rho": -1.},
                distribution="burr", 
               n_data=500, n_quantile="2n")
```
```
Estimators     W	RW	CW	CH	CHp	PRBp	CHps	PRBps

RMSE	      0.0471	0.0095	0.0063	0.0155	0.0149	0.015	0.0135	0.0164
```
You can also plot the bias, the variance and the RMSE

```
from extreme import visualization as statsviz
statsviz.evt_quantile_plot(n_replications=1000, 
   		           params={"evi":0.125, "rho": -0.125}, 
                           distribution="burr", 
                           n_data=500, 
                           n_quantile="2n")
```
![simulations](imgs/simulations_test.jpg)


### Real data

## Citing
@unpublished{girard2021revisiting,\
	TITLE = {{A refined Weissman estimator for extreme quantiles}},\
	AUTHOR = {El Methni, J. and Girard, S.},\
	URL = {{\tt https://hal.inria.fr/hal-03266676}}, \
	YEAR = {2021}
}