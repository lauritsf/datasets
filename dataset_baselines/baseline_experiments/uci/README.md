# UCI dataset uncertainty baselines

We compare the uncertainty estimates of the following methods on the UCI datasets:


## Results

### log_likelihood
| model                                         | Boston Housing              | Concrete Strength         | Energy Efficiency         | Kin8nm                 | Power Plant             | Wine Quality (Red)        | Yacht Hydrodynamics     |
|:----------------------------------------------|:----------------------------|:--------------------------|:--------------------------|:-----------------------|:------------------------|:--------------------------|:------------------------|
| Deterministic homoscedastic                   | -27.8418 +- 1.2253 (n=2)    | -3.4096 +- 0.3168 (n=3)   | -0.9724 +- 0.4223 (n=3)   | 1.2704 +- 0.0037 (n=3) | -2.8294 +- 0.0204 (n=3) | -1.7547 +- 0.0245 (n=3)   | -2.7705 +- 1.9649 (n=3) |
| Deterministic heteroscedastic                 | -4253.6294 +- nan (n=1)     | -30.9778 +- 13.5501 (n=3) | -19.0449 +- 13.2542 (n=3) | 1.3118 +- 0.0132 (n=3) | -2.7906 +- 0.0188 (n=3) | -14.1405 +- 12.3212 (n=3) | 0.2510 +- 0.2914 (n=3)  |
| Deterministic homoscedastic Ensemble (M=10)   | -5.6989 +- 0.1810 (n=2)     | -2.8951 +- 0.2599 (n=3)   | -0.7435 +- 0.3270 (n=3)   | 1.3082 +- 0.0026 (n=3) | -2.8280 +- 0.0211 (n=3) | -1.0342 +- 0.0511 (n=3)   | -1.0362 +- 1.0595 (n=3) |
| Deterministic heteroscedastic Ensemble (M=10) | -246.3989 +- 240.0081 (n=2) | -3.0632 +- 0.0912 (n=3)   | -3.8643 +- 3.0707 (n=3)   | 1.3678 +- 0.0108 (n=3) | -2.7862 +- 0.0157 (n=3) | -10.0671 +- 9.0437 (n=3)  | 0.6927 +- 0.1480 (n=3)  |
### mean_squared_error
| model                                         | Boston Housing          | Concrete Strength       | Energy Efficiency      | Kin8nm                 | Power Plant             | Wine Quality (Red)     | Yacht Hydrodynamics    |
|:----------------------------------------------|:------------------------|:------------------------|:-----------------------|:-----------------------|:------------------------|:-----------------------|:-----------------------|
| Deterministic homoscedastic                   | 18.0193 +- 0.1906 (n=2) | 24.8733 +- 2.5754 (n=3) | 0.3176 +- 0.0634 (n=3) | 0.0086 +- 0.0001 (n=3) | 33.1712 +- 0.6109 (n=3) | 0.7458 +- 0.0030 (n=3) | 0.3209 +- 0.1266 (n=3) |
| Deterministic heteroscedastic                 | 26.5256 +- nan (n=1)    | 51.8865 +- 4.1702 (n=3) | 0.4476 +- 0.0735 (n=3) | 0.0092 +- 0.0001 (n=3) | 32.6929 +- 0.7917 (n=3) | 0.7536 +- 0.0305 (n=3) | 1.3530 +- 0.9730 (n=3) |
| Deterministic homoscedastic Ensemble (M=10)   | 34.8558 +- 4.9161 (n=2) | 23.8991 +- 2.5352 (n=3) | 0.3179 +- 0.0589 (n=3) | 0.0087 +- 0.0000 (n=3) | 33.2704 +- 0.6108 (n=3) | 0.7269 +- 0.0344 (n=3) | 0.3572 +- 0.1709 (n=3) |
| Deterministic heteroscedastic Ensemble (M=10) | 25.3951 +- 2.8032 (n=2) | 47.9613 +- 4.5885 (n=3) | 1.2801 +- 0.7224 (n=3) | 0.0093 +- 0.0001 (n=3) | 32.7510 +- 0.6888 (n=3) | 0.7547 +- 0.0295 (n=3) | 1.1818 +- 0.7632 (n=3) |
### root_mean_squared_error
| model                                         | Boston Housing         | Concrete Strength      | Energy Efficiency      | Kin8nm                 | Power Plant            | Wine Quality (Red)     | Yacht Hydrodynamics    |
|:----------------------------------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|:-----------------------|
| Deterministic homoscedastic                   | 3.1159 +- 0.0961 (n=2) | 4.2572 +- 0.1220 (n=3) | 0.4884 +- 0.0242 (n=3) | 0.0876 +- 0.0003 (n=3) | 5.4190 +- 0.0170 (n=3) | 0.7502 +- 0.0030 (n=3) | 0.4165 +- 0.0461 (n=3) |
| Deterministic heteroscedastic                 | 3.5179 +- nan (n=1)    | 5.3513 +- 0.2325 (n=3) | 0.4908 +- 0.0267 (n=3) | 0.0862 +- 0.0007 (n=3) | 5.3184 +- 0.0289 (n=3) | 0.7814 +- 0.0089 (n=3) | 0.4577 +- 0.1550 (n=3) |
| Deterministic homoscedastic Ensemble (M=10)   | 4.0314 +- 0.2299 (n=2) | 4.3167 +- 0.1014 (n=3) | 0.4937 +- 0.0180 (n=3) | 0.0883 +- 0.0000 (n=3) | 5.4316 +- 0.0154 (n=3) | 0.7681 +- 0.0148 (n=3) | 0.4342 +- 0.0604 (n=3) |
| Deterministic heteroscedastic Ensemble (M=10) | 3.5097 +- 0.2267 (n=2) | 5.5128 +- 0.1571 (n=3) | 0.6711 +- 0.1421 (n=3) | 0.0867 +- 0.0006 (n=3) | 5.3271 +- 0.0213 (n=3) | 0.7837 +- 0.0090 (n=3) | 0.4494 +- 0.1210 (n=3) |

## Dataset details

| Name                  | # Examples | # Features | # Targets | Label                      | Notes               |
| --------------------- | ---------- | ---------- | --------- | -------------------------- | ------------------- |
| `boston_housing`      | 506        | 13         | 1         | `MEDV`                     |                     |
| `concrete_strength`   | 1030       | 8          | 1         | `concrete_compresive_strength` |                     |
| `energy_efficiency`   | 768        | 8          | 2         | `Y1`                       | `Y2` is excluded    |
| `kin8nm`              | 8192       | 8          | 1         | `y`                        |                     |
| `naval_propulsion`    | 11934      | 16         | 2         | `GTTC`                     | `GTCD` is excluded  |
| `power_plant`         | 9568       | 4          | 1         | `PE`                       |                     |
| `protein_structure`   | 45730      | 9          | 1         | `RMSD`                     |                     |
| `wine_quality_red`    | 1599       | 11         | 1         | `quality`                  |                     |
| `yacht_hydrodynamics` | 308        | 6          | 1         | `resid_resist`             |                     |
