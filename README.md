# Hansen's Superior Predictive Ability (SPA) Test
This Python project implements Hansen's Superior Predictive Ability (SPA) Test, a statistical test for evaluating the performance of multiple strategies against a null hypothesis. The SPA test adjusts for the correlation structure between strategies and accounts for multiple comparisons, identifying statistically significant outperforming strategies.

## Features
- Calculates Sharpe ratios for multiple strategies based on their log returns.
- Transforms Sharpe ratios using Hansen's adjustment to account for correlation effects.
- Computes significance thresholds and identifies strategies that outperform a given null Sharpe ratio.
- Implements multiple-comparison correction to control for false positives.

## Installation
### Requirements
This project requires Python 3.8 or later and the following libraries:
- numpy
- pandas
- scipy

To install the required libraries, run:
```
pip install numpy pandas scipy
```
### Files
- hansen_spa.py: Main Python script implementing the SPA test.
- log_returns_matrix.csv: Sample dataset containing log returns for various strategies.

## Usage
1. Prepare a CSV file containing log returns for multiple strategies. Each column should represent a strategy, and each row should represent a time period. Ensure the index is a timestamp or sequential period identifier.
2. Running the Test
Use the following steps to run the SPA test:
    ```
    python hansen_spa.py
    ```
    The script reads the log returns data from log_returns_matrix.csv and performs the SPA test with default parameters.
3. The output will list statistically significant strategies or indicate if none were found.

## References
- Pav, S. E. (2019). "Conditional inference on the asset with maximum Sharpe ratio".
- Hansen, P. R. (2005). "A Test for Superior Predictive Ability". Journal of Business & Economic Statistics.