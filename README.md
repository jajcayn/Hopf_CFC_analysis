# Cross-frequency coupling in the brain

* A simple study using a network of Hopf oscillators*

## Implementation details

We use [`neurolib`](https://github.com/neurolib-dev/neurolib) for all our simulations. Our network is a network of Hopf oscillators. Single "unit" is a subnetwork with one slow and one fast Hopf, connected via multiplicative coupling. In a case of more than one "unit", **only** the slow ones are interconnected, using a diffusive coupling.

## Basic parameters

- `w`: oscillator frequency
- `a`: Hopf bifurcation parameters

### Approximate frequencies

| w     	| frequency [Hz] 	|
|-------	|----------------	|
| 0.003 	| 0.5            	|
| 0.005 	| 0.8            	|
| 0.01  	| 2              	|
| 0.05  	| 8              	|
| 0.06  	| 10             	|
| 0.08  	| 12             	|
| 0.1   	| 17             	|
| 0.2   	| 32             	|
| 0.3   	| 50             	|
