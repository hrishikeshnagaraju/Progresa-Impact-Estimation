# Progresa-Impact-Estimation
Evaluating Effectiveness of a Government Subsidy

## Introduction to the project

The goal of this project is to implement some of the econometric techniques to measure the impact of Progresa on secondary school enrollment rates. <br>
For this project, I used data from the [Progresa program](http://en.wikipedia.org/wiki/Oportunidades), a government social assistance program in Mexico.<br>
 This program, as well as the details of its impact, are described in the paper "[School subsidies for the poor: evaluating the Mexican Progresa poverty program](http://www.sciencedirect.com/science/article/pii/S0304387803001858)", by Paul Shultz. 


## Strategy for impact evaluation

The key idea here is that when estimating the impact of a treatment, our goal is to determine the Average Treatment Effect (ATE). However, our estimate can be biased if there is selection bias, which occurs when the treatment and control groups differ in ways other than the treatment itself.

To summarize:

Impact Estimate (D): The difference in observed outcomes between the treatment and control groups.
Average Treatment Effect (ATE): The true effect of the treatment on the treated group.
Selection Bias (B): The difference in potential outcomes between the treatment and control groups that is not due to the treatment.
In the absence of selection bias, the impact estimate 
𝐷
D would equal the ATE. However, if selection bias is present, 
𝐷
D will be the sum of the ATE and the selection bias, leading to a biased estimate of the treatment's effect.<br>
Keeping this equation in mind, I will go through; <br>

1. **Descriptive analysis** : Explore the dataset's characteristics and **see if there's a selection bias between the two groups**.
2. **Simple difference** : Estimate the causal impact $D$ by simple difference, by tabular analysis and regression.
3. **Difference-in-difference** : Re-estimate the causal effect $D$ by Difference-in-Difference, the method which is more suitable for this dataset.





### The timeline of the Progresa program:

 * Baseline survey conducted in 1997
 * Intervention begins in 1998, "Wave 1" of surveys conducted in 1998
 * "Wave 2" of surveys conducted in 1999
 * Evaluation ends in 2000, at which point the control villages were treated. 
 
The data are actual data collected to evaluate the impact of the Progresa program. In this file, each row corresponds to an observation taken for a given child for a given year. There are two years of data (1997 and 1998), and just under 40,000 children who are surveyed in each year. For each child-year observation, the following variables are collected:

| Variable name | Description|
|------|------|
|year	  |year in which data is collected
|sex	  |male = 1|
|indig	  |indigenous = 1|
|dist_sec |nearest distance to a secondary school|
|sc	      |enrolled in school in year of survey|
|grc      |grade enrolled|
|fam_n    |family size|
|min_dist |	min distance to an urban center|
|dist_cap |	min distance to the capital|
|poor     |labeled poor: pobre = 1, no pobre = 0|
|progresa |treatment : basal =1, '0' = 0|
|hohedu	  |years of schooling of head of household|
|hohwag	  |monthly wages of head of household|
|welfare_index|	welfare index used to classify poor|
|hohsex	|gender of head of household (male=1)|
|hohage	|age of head of household|
|age	|years old|
|folnum	|individual id|
|village|	village id|
|sc97	|schooling in 1997|
|grc97  |grade enrolled in 1997

---

# 1. Descriptive analysis

### Data Exploration & null check

I checked some statistics (mean, median and standard deviation) for all of the demographic variables in the dataset. 
I also checked null values of each column.



```python
# import libraries
import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#import dataset
df = pd.read_csv('progresa_sample.csv')

#Take a look 
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>sex</th>
      <th>indig</th>
      <th>dist_sec</th>
      <th>sc</th>
      <th>grc</th>
      <th>fam_n</th>
      <th>min_dist</th>
      <th>dist_cap</th>
      <th>poor</th>
      <th>...</th>
      <th>hohedu</th>
      <th>hohwag</th>
      <th>welfare_index</th>
      <th>hohsex</th>
      <th>hohage</th>
      <th>age</th>
      <th>village</th>
      <th>folnum</th>
      <th>grc97</th>
      <th>sc97</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>97</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.473</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>7</td>
      <td>21.168384</td>
      <td>21.168384</td>
      <td>pobre</td>
      <td>...</td>
      <td>6</td>
      <td>0.0</td>
      <td>583.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>13</td>
      <td>163</td>
      <td>1</td>
      <td>7</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>98</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.473</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>7</td>
      <td>21.168384</td>
      <td>21.168384</td>
      <td>pobre</td>
      <td>...</td>
      <td>6</td>
      <td>0.0</td>
      <td>583.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>14</td>
      <td>163</td>
      <td>1</td>
      <td>7</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>97</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.473</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>7</td>
      <td>21.168384</td>
      <td>21.168384</td>
      <td>pobre</td>
      <td>...</td>
      <td>6</td>
      <td>0.0</td>
      <td>583.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>12</td>
      <td>163</td>
      <td>2</td>
      <td>6</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>98</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.473</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>7</td>
      <td>21.168384</td>
      <td>21.168384</td>
      <td>pobre</td>
      <td>...</td>
      <td>6</td>
      <td>0.0</td>
      <td>583.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>13</td>
      <td>163</td>
      <td>2</td>
      <td>6</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>97</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.473</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>7</td>
      <td>21.168384</td>
      <td>21.168384</td>
      <td>pobre</td>
      <td>...</td>
      <td>6</td>
      <td>0.0</td>
      <td>583.0</td>
      <td>1.0</td>
      <td>35.0</td>
      <td>8</td>
      <td>163</td>
      <td>3</td>
      <td>2</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
#Check null values.
for col in df.columns:
    print(f'dtype of {col} is {type(df[col][0])}. Missing {df[col].isnull().sum()} values.')
```

    dtype of year is <class 'numpy.int64'>. Missing 0 values.
    dtype of sex is <class 'numpy.float64'>. Missing 24 values.
    dtype of indig is <class 'numpy.float64'>. Missing 300 values.
    dtype of dist_sec is <class 'numpy.float64'>. Missing 0 values.
    dtype of sc is <class 'numpy.float64'>. Missing 8453 values.
    dtype of grc is <class 'numpy.float64'>. Missing 6549 values.
    dtype of fam_n is <class 'numpy.int64'>. Missing 0 values.
    dtype of min_dist is <class 'numpy.float64'>. Missing 0 values.
    dtype of dist_cap is <class 'numpy.float64'>. Missing 0 values.
    dtype of poor is <class 'str'>. Missing 0 values.
    dtype of progresa is <class 'str'>. Missing 0 values.
    dtype of hohedu is <class 'numpy.int64'>. Missing 0 values.
    dtype of hohwag is <class 'numpy.float64'>. Missing 0 values.
    dtype of welfare_index is <class 'numpy.float64'>. Missing 210 values.
    dtype of hohsex is <class 'numpy.float64'>. Missing 20 values.
    dtype of hohage is <class 'numpy.float64'>. Missing 10 values.
    dtype of age is <class 'numpy.int64'>. Missing 0 values.
    dtype of village is <class 'numpy.int64'>. Missing 0 values.
    dtype of folnum is <class 'numpy.int64'>. Missing 0 values.
    dtype of grc97 is <class 'numpy.int64'>. Missing 0 values.
    dtype of sc97 is <class 'numpy.float64'>. Missing 3872 values.



```python
#Map some variables for convenience
df['poor'] = df['poor'].map({'pobre': 1, 'no pobre': 0})
df['progresa'] = df['progresa'].map({'0': 0, 'basal': 1})

#Make a demographic data table
demo = df.drop(axis= 1, columns = ['year','folnum', 'village'])
cols = demo.columns.sort_values(ascending= True)
demo = demo[cols]
```


```python
#Make a table with mean, std, median for each variable
mean = []
std = []
median = []

for col in cols:
    mean.append(demo[col].mean())
    std.append(demo[col].std())
    median.append(demo[col].median())
    
summ = pd.DataFrame([mean, std, median], columns = cols, index = ['mean', 'std', 'median'])
summ
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>dist_cap</th>
      <th>dist_sec</th>
      <th>fam_n</th>
      <th>grc</th>
      <th>grc97</th>
      <th>hohage</th>
      <th>hohedu</th>
      <th>hohsex</th>
      <th>hohwag</th>
      <th>indig</th>
      <th>min_dist</th>
      <th>poor</th>
      <th>progresa</th>
      <th>sc</th>
      <th>sc97</th>
      <th>sex</th>
      <th>welfare_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>mean</th>
      <td>11.366460</td>
      <td>147.674452</td>
      <td>2.418910</td>
      <td>7.215715</td>
      <td>3.963537</td>
      <td>3.705372</td>
      <td>44.436717</td>
      <td>2.768104</td>
      <td>0.925185</td>
      <td>586.985312</td>
      <td>0.298324</td>
      <td>103.447520</td>
      <td>0.846498</td>
      <td>0.615663</td>
      <td>0.819818</td>
      <td>0.813922</td>
      <td>0.512211</td>
      <td>690.346564</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.167744</td>
      <td>76.063134</td>
      <td>2.234109</td>
      <td>2.352900</td>
      <td>2.499063</td>
      <td>2.572387</td>
      <td>11.620372</td>
      <td>2.656106</td>
      <td>0.263095</td>
      <td>788.133664</td>
      <td>0.457525</td>
      <td>42.089441</td>
      <td>0.360473</td>
      <td>0.486441</td>
      <td>0.384342</td>
      <td>0.389172</td>
      <td>0.499854</td>
      <td>139.491130</td>
    </tr>
    <tr>
      <th>median</th>
      <td>11.000000</td>
      <td>132.001494</td>
      <td>2.279000</td>
      <td>7.000000</td>
      <td>4.000000</td>
      <td>4.000000</td>
      <td>43.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>111.228612</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>685.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Differences at baseline?

Next, I checked the baseline (1997) demographic characteristics **for the poor**  different in treatment and control groups.<br>
I used a T-Test (p = 0.05 as threshold) to determine whether there is a statistically significant difference in the average values of each of the variables in the dataset.




```python
#Get the average of each variable for treatment and control group, respectively. 
variables = ['age', 'dist_cap', 'dist_sec', 'fam_n', 'grc97', 'hohage', 'hohedu', 'hohsex', 'hohwag', 'indig','progresa', 'min_dist','sc97', 'sex', 'welfare_index']

base = df[(df['year'] == 97) &  (df['poor'] == 1)][variables].groupby(by = 'progresa').mean().transpose()
base['diff'] = base[1] - base[0]
```


```python
#show the difference in table

p_values = []

vals = ['age', 'dist_cap', 'dist_sec', 'fam_n', 'grc97', 'hohage', 'hohedu', 'hohsex', 'hohwag', 'indig', 'min_dist','sc97', 'sex', 'welfare_index']
for val in vals:
    p_values.append(round(
        scipy.stats.ttest_ind(
            a = df[(df['year'] == 97) & (df['poor'] == 1) & (df['progresa'] == 1)][val],
            b = df[(df['year'] == 97) & (df['poor'] == 1) & (df['progresa'] == 0)][val],
            equal_var = False, #I chose to do Welch's t-test, meaning do not assume that these two samples have the same variance.
            nan_policy = 'omit' #ignore null and compute
        )[1],4)
    )


base['p-value'] = np.array(p_values)
base.columns = ['Control','Treatment', 'Difference', 'p_value']
base[['Treatment','Control', 'Difference', 'p_value']] 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Treatment</th>
      <th>Control</th>
      <th>Difference</th>
      <th>p_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>10.716991</td>
      <td>10.742023</td>
      <td>-0.025032</td>
      <td>0.4784</td>
    </tr>
    <tr>
      <th>dist_cap</th>
      <td>150.829074</td>
      <td>153.769730</td>
      <td>-2.940656</td>
      <td>0.0011</td>
    </tr>
    <tr>
      <th>dist_sec</th>
      <td>2.453122</td>
      <td>2.507662</td>
      <td>-0.054540</td>
      <td>0.0427</td>
    </tr>
    <tr>
      <th>fam_n</th>
      <td>7.281327</td>
      <td>7.302469</td>
      <td>-0.021142</td>
      <td>0.4290</td>
    </tr>
    <tr>
      <th>grc97</th>
      <td>3.531599</td>
      <td>3.543050</td>
      <td>-0.011450</td>
      <td>0.6895</td>
    </tr>
    <tr>
      <th>hohage</th>
      <td>43.648828</td>
      <td>44.276918</td>
      <td>-0.628090</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>hohedu</th>
      <td>2.663139</td>
      <td>2.590348</td>
      <td>0.072791</td>
      <td>0.0104</td>
    </tr>
    <tr>
      <th>hohsex</th>
      <td>0.924656</td>
      <td>0.922947</td>
      <td>0.001709</td>
      <td>0.5721</td>
    </tr>
    <tr>
      <th>hohwag</th>
      <td>544.339544</td>
      <td>573.163558</td>
      <td>-28.824015</td>
      <td>0.0003</td>
    </tr>
    <tr>
      <th>indig</th>
      <td>0.325986</td>
      <td>0.332207</td>
      <td>-0.006222</td>
      <td>0.2459</td>
    </tr>
    <tr>
      <th>min_dist</th>
      <td>107.152915</td>
      <td>103.237854</td>
      <td>3.915060</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>sc97</th>
      <td>0.822697</td>
      <td>0.815186</td>
      <td>0.007511</td>
      <td>0.0965</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>0.519317</td>
      <td>0.505052</td>
      <td>0.014265</td>
      <td>0.0122</td>
    </tr>
    <tr>
      <th>welfare_index</th>
      <td>655.428377</td>
      <td>659.579100</td>
      <td>-4.150723</td>
      <td>0.0015</td>
    </tr>
  </tbody>
</table>
</div>



**Note**

- According to the result of t-test, there are variables whose p-value is below 0.05. If we set our thresholds at 0.05, we can reject the hypothesis that these two sample groups have no stasitically significant difference. In other words, the selection of these two groups could be biased.
- Since there could be statistically significant differences between these two groups, we can't use single difference between treatment and control, or pre-post comparison that require 0 difference at baseline.  
- We need to choose the method that allows us to assume the difference in the treatment and control groups at baseline.

---

# 2. Simple Difference

My goal is to estimate the causal impact of the PROGRESA program on the social and economic outcomes of individuals in Mexico.<br> 
We will focus on the impact of the program on school enrollment rates among the poor (those with poor== 1), <br>
since only the poor were eligible to receive PROGRESA assistance, and since a primary objective of the program was to increase school enrollment.

### Simple difference(1): Tabular Analysis & T-test

Let's begin by estimating the impact of Progresa using "simple differences." <br>
Simple difference is a very straight forward approach. We simple going to compare the outcomes of treatment and control group, or pre and post periods.<br>
This method is simple and easy to understand, but requires the strong assumption to generate unbiased estimate.<br>
Its underlying assumption is "Two groups would have the same **outcome** in the absence of treatment."<br>
As we already saw there could be a selection bias between treatment and control group, so we need to keep in mind that this method will lead to biased estimation.<br><br><br>


First, I calculated the average enrollment rate among **poor** households in the Treatment villages and the average enrollment rate among **poor** households in the control villages in 1998, and then used T-test to see its statistics significance.


```python
#Compute the difference 
ave_sc_t = df[(df['year'] == 98) & (df['poor'] == 1) & (df['progresa'] == 1)]['sc']
ave_sc_c = df[(df['year'] == 98) & (df['poor'] == 1) & (df['progresa'] == 0)]['sc']

p_val = scipy.stats.ttest_ind(a = ave_sc_t, b = ave_sc_c, equal_var= False, nan_policy='omit')[1]

print(f'Average enrollment rate of the treatment group {ave_sc_t.mean()}'  )
print(f'Average enrollment rate of the control group is {ave_sc_c.mean()}')
print(f'Difference is {ave_sc_t.mean() - ave_sc_c.mean()}')
print(f'P-value of the null hypothesis is {p_val}')
```

    Average enrollment rate of the treatment group 0.8464791213954308
    Average enrollment rate of the control group is 0.807636956730308
    Difference is 0.0388421646651228
    P-value of the null hypothesis is 2.9655072988948406e-16


*Note*

- p-value is nearly 0, way below 5%.
- This simple difference implies that progresa had a causal impact to the average enrollment rate on the treatment villages. (3.88%)

### Simple difference(2): Regression

Now, let's estimate the effects of Progresa on enrollment using a regression model, by regressing the 1998 enrollment rates **of the poor** on treatment assignment. 


```python
# Run Multiple Regression
vals = ['age', 'dist_cap', 'dist_sec', 'fam_n', 'grc', 'hohage', 'hohedu', 'hohsex', 'hohwag', 'indig', 'min_dist','progresa','sex','sc','welfare_index']
mul = df[(df['year'] == 98) & (df['poor'] == 1)][vals].dropna(axis = 0)

X = pd.get_dummies(mul.drop(axis = 1, columns = ['sc']), columns = ['indig','sex'], drop_first= True, dtype= 'int')
X = sm.add_constant(X)
y = mul['sc']
model = sm.OLS(y,X).fit()
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     sc   R-squared:                       0.304
    Model:                            OLS   Adj. R-squared:                  0.303
    Method:                 Least Squares   F-statistic:                     847.0
    Date:                Sat, 03 Feb 2024   Prob (F-statistic):               0.00
    Time:                        13:41:45   Log-Likelihood:                -6928.7
    No. Observations:               27200   AIC:                         1.389e+04
    Df Residuals:                   27185   BIC:                         1.401e+04
    Df Model:                          14                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const             1.6540      0.020     84.276      0.000       1.616       1.692
    age              -0.1000      0.001    -83.566      0.000      -0.102      -0.098
    dist_cap          0.0002   3.62e-05      5.247      0.000       0.000       0.000
    dist_sec         -0.0074      0.001     -8.537      0.000      -0.009      -0.006
    fam_n             0.0015      0.001      1.724      0.085      -0.000       0.003
    grc               0.0510      0.001     34.772      0.000       0.048       0.054
    hohage           -0.0003      0.000     -1.571      0.116      -0.001    7.49e-05
    hohedu            0.0041      0.001      4.790      0.000       0.002       0.006
    hohsex            0.0051      0.008      0.672      0.502      -0.010       0.020
    hohwag        -1.515e-06   2.77e-06     -0.547      0.584   -6.94e-06    3.91e-06
    min_dist          0.0004   6.22e-05      5.653      0.000       0.000       0.000
    progresa          0.0319      0.004      8.140      0.000       0.024       0.040
    welfare_index -1.809e-05   1.79e-05     -1.011      0.312   -5.32e-05     1.7e-05
    indig_1.0         0.0272      0.005      5.852      0.000       0.018       0.036
    sex_1.0           0.0336      0.004      8.855      0.000       0.026       0.041
    ==============================================================================
    Omnibus:                     3250.561   Durbin-Watson:                   1.733
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             4554.560
    Skew:                          -0.949   Prob(JB):                         0.00
    Kurtosis:                       3.645   Cond. No.                     1.08e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.08e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.


*Note*

- The coefficient of treatment is **0.0319**, this implies that if one poor household receives progresa treatment, then their likelihood of school enrollment will increse by 3.19%.
- For example, for every 10,000 poor children, **319** more children newly started schooling because of progresa.
- P-value of progresa coefficient is very small (less than 0.05), so we can reject the null hypothesis that this coefficient is likely to equal to zero.

---
# 3. Difference-in-Difference

Thus far, we have computed the effects of Progresa by estimating the difference in 1998 enrollment rates across villages.<br> 
An alternative approach would be to compute the treatment effect using a **Difference-In-Differences(DID)** framework.<br>
The DID method is an approach to estimate causal effects under the selection bias between treatment and control group.<br><br>

The underlying assumption of DID is, "there exist parallel trends over time in enrollment rates between treated and control villages."<br>
So it doesn't require the assumption Simple difference requires, which is "The counterfactual assumption is that, in the absence of treatment, the average school enrollment in the treatment group and the average school enrollment in the control groups would have been the same."<br>
Although we need pre-period data to execute this method, this advantage on assumption is critical about impact evaluation in reality,<br> 
especially because we already know there could be a selection bias between treatment and control group of this dataset.


### Difference-in-Difference(1): Tabular Analysis

Let's begin by estimating the average treatment effects of the program for poor households using data from 1997 and 1998.<br>
Specifically, I calculated the difference (between 1997 and 1998) in enrollment rates among poor households in treated villages; then compute the difference (between 1997 and 1998) in enrollment rates among poor households in control villages. 


```python
#Tabular Analysis for DID
post_t = df[(df['year'] == 98) & (df['poor'] == 1) & (df['progresa'] == 1)]['sc'].mean()
pre_t = df[(df['year'] == 97) & (df['poor'] == 1) & (df['progresa'] == 1)]['sc'].mean()
post_c = df[(df['year'] == 98) & (df['poor'] == 1) & (df['progresa'] == 0)]['sc'].mean()
pre_c = df[(df['year'] == 97) & (df['poor'] == 1) & (df['progresa'] == 0)]['sc'].mean()


print(f'A diff in treatment is {post_t - pre_t}')
print(f'A diff in control is {post_c - pre_c}')
print(f'A diff-diff is {(post_t - pre_t) - (post_c - pre_c)}')

table = pd.DataFrame([[pre_c, post_c],[pre_t , post_t]], columns = ['1997','1998'], index=['Control', 'Treatment'])
table
```

    A diff in treatment is 0.023782233992046597
    A diff in control is -0.007549046327276487
    A diff-diff is 0.031331280319323085





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1997</th>
      <th>1998</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Control</th>
      <td>0.815186</td>
      <td>0.807637</td>
    </tr>
    <tr>
      <th>Treatment</th>
      <td>0.822697</td>
      <td>0.846479</td>
    </tr>
  </tbody>
</table>
</div>



*Note*
- DID effect is 0.0313, almost the same as the previous one but it's slightly smaller.
- The underlying assumption is that treatment group and control group have the same trend in the abscence of the treatment.

### Difference-in-Difference(2): Regression

Now I use a regression specification to estimate the average treatment effects of the program in a difference-in-differences, for the poor households.


```python
#Differenc in Difference Method
vals = ['year','age', 'dist_cap', 'dist_sec', 'fam_n', 'grc', 'hohage', 'hohedu', 'hohsex', 'hohwag', 'indig', 'min_dist','progresa','sex','sc','welfare_index']
df_did = df[df['poor'] == 1][vals].dropna(axis= 0)
df_did['year'] = df_did['year'].map({98 : 1, 97 : 0})

#Make an intersection term
df_did['did'] = df_did['year'] * df_did['progresa']
X = sm.add_constant(df_did.drop(axis = 1, columns = ['sc']))

y = df_did['sc']
model = sm.OLS(y, X).fit()
print(f"DID estimator is {model.params['did']}")
print(model.summary())

```

    DID estimator is 0.02915726848791441
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                     sc   R-squared:                       0.311
    Model:                            OLS   Adj. R-squared:                  0.310
    Method:                 Least Squares   F-statistic:                     1631.
    Date:                Mon, 05 Feb 2024   Prob (F-statistic):               0.00
    Time:                        22:28:47   Log-Likelihood:                -15320.
    No. Observations:               57938   AIC:                         3.067e+04
    Df Residuals:                   57921   BIC:                         3.083e+04
    Df Model:                          16                                         
    Covariance Type:            nonrobust                                         
    =================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
    ---------------------------------------------------------------------------------
    const             1.6163      0.014    119.229      0.000       1.590       1.643
    year              0.0251      0.004      5.866      0.000       0.017       0.033
    age              -0.0983      0.001   -120.910      0.000      -0.100      -0.097
    dist_cap          0.0002   2.51e-05      7.329      0.000       0.000       0.000
    dist_sec         -0.0065      0.001    -10.589      0.000      -0.008      -0.005
    fam_n             0.0009      0.001      1.505      0.132      -0.000       0.002
    grc               0.0486      0.001     48.071      0.000       0.047       0.051
    hohage        -4.707e-05      0.000     -0.354      0.723      -0.000       0.000
    hohedu            0.0037      0.001      6.225      0.000       0.003       0.005
    hohsex           -0.0017      0.005     -0.324      0.746      -0.012       0.008
    hohwag          4.95e-07    1.9e-06      0.260      0.795   -3.24e-06    4.23e-06
    indig             0.0317      0.003      9.854      0.000       0.025       0.038
    min_dist          0.0003   4.31e-05      7.579      0.000       0.000       0.000
    progresa          0.0033      0.004      0.876      0.381      -0.004       0.011
    sex               0.0367      0.003     14.006      0.000       0.032       0.042
    welfare_index -1.631e-05   1.24e-05     -1.318      0.188   -4.06e-05    7.95e-06
    did               0.0292      0.005      5.390      0.000       0.019       0.040
    ==============================================================================
    Omnibus:                     6083.138   Durbin-Watson:                   1.514
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):             8175.414
    Skew:                          -0.887   Prob(JB):                         0.00
    Kurtosis:                       3.488   Cond. No.                     1.09e+04
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.09e+04. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
#Visualize it
const = 1.6163
year = 0.0251
progresa = 0.0033
did =  0.0292

x = np.array([0, 1])
y_t = const + year * x + progresa + did * x
y_c = const + year * x
plt.plot(x, y_t, color = 'red', label = "Treatment")
plt.plot(x, y_c, color = 'blue', label = "Control")
plt.legend()
```




    <matplotlib.legend.Legend at 0x15b9f3c50>




    
![png](Images/did_line.png)
    


*Note*
- The DID estimator of is 0.0292, meaning the treatment increases the likelihood of child's enrollment of each household by 2.92%.
- In other words, for every 10,000 children from poor households, 292 more children newly started schooling because of progresa.
- Compared to the result of simple difference (0.319), the DID estimator is smaller by 0.0027. This indicates **the simple difference overestimated the causal effect of treatment.**

# Conclusion

- As long as standing with the assumption that the treatment group and control group have the same trend in the absence of the treatment, the double differences method can estimate an unbiased causal impact.
- Thus, according to DID regression, we can claim that the Progresa had a causal impact on the enrollment rates of poor households, *by 2.92%.*
- Other methods, such as simple differences, have serious flaws in their underlying assumptions and thus overestimated the causal impact up *3.19%*
- It is really important to be aware of the underlying assumption of the method we use, and the limitation of perfect random assignment!


