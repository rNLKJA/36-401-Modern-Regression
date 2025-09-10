36-401, Chapter 2: Simple Linear Regression
================

## Basic Properties of the $\beta$

In this chapter, we’ll still consider the simple linear regression
model, which involves parameters $\beta_0$, $\beta_1$, and $\sigma_2$.
We’ll focus on the least squares estimators $\beta_0$ and $\beta_1$, and
study properties about these estimators. We’ll also consider
consequences of having to estimate the variance $\sigma_2$.

First, let’s recall the three basic assumptions of simple linear
regression: 1. Mean-Zero Noise: $\mathbb{E}(\epsilon_i | X_i) = 0$ for
all $i$ 2. Constant Variance (“homoskedasticity”)
$\text{Var}(\epsilon_i | X_i) = \sigma_2$ for all $i$ 3. Uncorrelated
Noise: The $\epsilon_i$ are uncorrelated, i.e.,
$\text{Cov}(\epsilon_i, \epsilon+j | X+i) = 0$ for all $i \ne j$.

As we’ve seen, the least squares estimators are:

$$
\hat\beta_1 = \frac{\sum_i(Y_i - \bar Y)(X_i - \bar X)}{\sum_i(X_i - \bar X)^2} = \frac{s_{xy}}{s_{xx}}
$$

$$
\hat\beta_0 = \bar Y - \hat\beta_1 \bar X
$$

Under the above assumptions, we can find the following properties for
$\hat\beta_0$ and $\hat\beta_1$ hold:

------------------------------------------------------------------------

1.  Linearity  
    $\hat\beta_1 = \sum_{i=1}^n w_i Y_i,\quad w_i = \frac{X_i - \bar X}{\sum_{j=1}^n (X_j - \bar X)^2},\ \sum_i w_i = 0$  
    $\hat\beta_0 = \sum_{i=1}^n a_i Y_i,\quad a_i = \frac{1}{n} - \bar X w_i,\ \sum_i a_i = 1$

2.  Unbiasedness (conditioning on the fixed design $X_1,\dots,X_n$)  
    $\mathbb{E}(\hat\beta_1 \mid X) = \beta_1,\qquad \mathbb{E}(\hat\beta_0 \mid X) = \beta_0$

3.  Variances  
    $\text{Var}(\hat\beta_1 \mid X) = \frac{\sigma^2}{s_{xx}},\qquad s_{xx} = \sum_{i=1}^n (X_i - \bar X)^2$  
    $\text{Var}(\hat\beta_0 \mid X) = \sigma^2\left(\frac{1}{n} + \frac{\bar X^2}{s_{xx}}\right)$

4.  Covariance  
    $\text{Cov}(\hat\beta_0,\hat\beta_1 \mid X) = -\,\frac{\bar X \sigma^2}{s_{xx}}$

5.  Correlation  
    $\text{Corr}(\hat\beta_0,\hat\beta_1 \mid X) = - \frac{\bar X}{\sqrt{ s_{xx}/n + \bar X^2 }}$

6.  Sampling distributions

    - Without assuming normal errors: they are unbiased with the above
      variances; asymptotically normal (under mild conditions).  
    - If $\epsilon_i \sim \mathcal{N}(0,\sigma^2)$ i.i.d.:  
      $\hat\beta_1 \mid X \sim \mathcal{N}\!\left(\beta_1,\frac{\sigma^2}{s_{xx}}\right),\quad \hat\beta_0 \mid X \sim \mathcal{N}\!\left(\beta_0,\sigma^2\left(\frac{1}{n} + \frac{\bar X^2}{s_{xx}}\right)\right)$  
      Jointly bivariate normal with the stated covariance.

7.  Estimation of $\sigma^2$  
    Residuals $e_i = Y_i - \hat\beta_0 - \hat\beta_1 X_i$  
    $\text{SSE} = \sum_{i=1}^n e_i^2,\qquad \hat\sigma^2 = \frac{\text{SSE}}{n-2}$
    (unbiased)

8.  Standard errors (plug in $\hat\sigma^2$)  
    $\text{SE}(\hat\beta_1) = \sqrt{\frac{\hat\sigma^2}{s_{xx}}},\qquad \text{SE}(\hat\beta_0) = \sqrt{ \hat\sigma^2\left(\frac{1}{n} + \frac{\bar X^2}{s_{xx}}\right) }$

9.  t-statistics (normal error assumption)  
    $t_j = \frac{\hat\beta_j - \beta_j^{(0)}}{\text{SE}(\hat\beta_j)} \sim t_{n-2}\quad (j=0,1)$

10. Confidence intervals (level $1-\alpha$)  
    $\hat\beta_j \pm t_{1-\alpha/2,\ n-2}\ \text{SE}(\hat\beta_j),\quad j=0,1$

11. Gauss–Markov (efficiency)  
    $\hat\beta_0,\hat\beta_1$ are BLUE: Best (minimum variance) Linear
    Unbiased Estimators under the three core assumptions.

12. Prediction / mean response at a new $x_0$  
    Mean estimator: $\hat\mu(x_0) = \hat\beta_0 + \hat\beta_1 x_0$  
    $\text{Var}(\hat\mu(x_0) \mid X) = \sigma^2\left(\frac{1}{n} + \frac{(x_0 - \bar X)^2}{s_{xx}}\right)$  
    Prediction variance for a future $Y_0$:  
    $\text{Var}(Y_0 - \hat\mu(x_0) \mid X) = \sigma^2\left(1 + \frac{1}{n} + \frac{(x_0 - \bar X)^2}{s_{xx}}\right)$

------------------------------------------------------------------------

To prove the above results, it is useful to note the following fact:
$\hat\beta_1$ can be written as a linear combination of the $Y_i$ in the
following way.

$$
\hat\beta_1 = \sum^n_{i=1}k_iY_i,\quad \text{where }k_i = \frac{X_i - \bar X}{\sum^n_{i=1}(X_j - \bar X)^2} 
$$ **Exercise 0.1**. Prove that the variance of $\hat\beta_1$ takes the
stated from.

Goal: Prove that $\text{Var}(\hat\beta_1 \mid X) = \sigma^2 / s_{xx}$,
where $s_{xx} = \sum_{i=1}^n (X_i - \bar X)^2$.

### Step 1: Start from the model

$$
Y_i = \beta_0 + \beta_1 X_i + \epsilon_i,\quad \mathbb{E}(\epsilon_i \mid X)=0,\quad \text{Var}(\epsilon_i \mid X)=\sigma^2,\quad \text{Cov}(\epsilon_i,\epsilon_j \mid X)=0\ (i\ne j)
$$

### Step 2: Write $\hat\beta_1$ in a form using only errors

Recall $$
\hat\beta_1 = \frac{\sum_{i=1}^n (X_i - \bar X)(Y_i - \bar Y)}{s_{xx}}
$$ Substitute $Y_i = \beta_0 + \beta_1 X_i + \epsilon_i$ and note
$\bar Y = \beta_0 + \beta_1 \bar X + \bar\epsilon$ with
$\bar\epsilon = \frac{1}{n}\sum_{i=1}^n \epsilon_i$: $$
Y_i - \bar Y = \beta_1(X_i - \bar X) + (\epsilon_i - \bar\epsilon)
$$ Then $$
\hat\beta_1 = \frac{\sum (X_i - \bar X)\big[\beta_1(X_i - \bar X) + (\epsilon_i - \bar\epsilon)\big]}{s_{xx}}
= \beta_1 + \frac{\sum (X_i - \bar X)(\epsilon_i - \bar\epsilon)}{s_{xx}}
$$ But $$
\sum_{i=1}^n (X_i - \bar X)\bar\epsilon = \bar\epsilon \sum_{i=1}^n (X_i - \bar X) = 0
$$ Hence $$
\hat\beta_1 = \beta_1 + \frac{\sum_{i=1}^n (X_i - \bar X)\epsilon_i}{s_{xx}}
$$

### Step 3: Variance

Conditioning on the design (the $X_i$ treated as constants): $$
\text{Var}(\hat\beta_1 \mid X) = \text{Var}\left(\frac{\sum (X_i - \bar X)\epsilon_i}{s_{xx}} \mid X\right)
= \frac{1}{s_{xx}^2} \text{Var}\left(\sum (X_i - \bar X)\epsilon_i \mid X\right)
$$ Errors are uncorrelated with common variance $\sigma^2$, so $$
\text{Var}\left(\sum (X_i - \bar X)\epsilon_i \mid X\right)
= \sum (X_i - \bar X)^2 \sigma^2 = \sigma^2 s_{xx}
$$ Therefore $$
\text{Var}(\hat\beta_1 \mid X) = \frac{\sigma^2 s_{xx}}{s_{xx}^2} = \frac{\sigma^2}{s_{xx}}
$$

### Step 4: Alternative “weights” view

Define weights $$
w_i = \frac{X_i - \bar X}{s_{xx}},\quad \hat\beta_1 = \sum_{i=1}^n w_i Y_i
$$ Then $$
\text{Var}(\hat\beta_1 \mid X) = \sum_{i=1}^n w_i^2 \sigma^2 = \sigma^2 \sum \frac{(X_i - \bar X)^2}{s_{xx}^2} = \frac{\sigma^2}{s_{xx}}
$$ Same result.

That completes the proof.

------------------------------------------------------------------------

An important quantity for inference will be the standard error. The
stan- dard error of an estimator is the square root of its variance. As
we saw above, the variance may have to be estimated, such that the
standard error will also have to be estimated. Thus, we have:

$$
\text{SE}(\hat\beta) = \sqrt{\text{Var}(\hat\beta_1)}\quad \text{and } \hat{\text{SE}}(\hat\beta_1) = \sqrt{\hat{\text{Var}}(\hat\beta_1)} 
$$

The estimated SEs are displayed in `summary` output in R.

Exercise 0.2. Let’s consider the BEA example from previous chapters;
`summary()` output is shown below. State and interpret the standard
errors.

``` r
bea_url <- "https://www.stat.cmu.edu/~cshalizi/TALR/data/bea-2006.csv"
bea <- read.csv(bea_url, na.strings = c("NA", "", "."))
bea_fit <- lm(pcgmp ~ log(pop), data = bea)

summary(bea_fit)
```

    ## 
    ## Call:
    ## lm(formula = pcgmp ~ log(pop), data = bea)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -21572  -4765  -1016   3686  40207 
    ## 
    ## Coefficients:
    ##             Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept) -23306.2     4957.1  -4.702 3.67e-06 ***
    ## log(pop)      4449.8      390.9  11.383  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7929 on 364 degrees of freedom
    ## Multiple R-squared:  0.2625, Adjusted R-squared:  0.2605 
    ## F-statistic: 129.6 on 1 and 364 DF,  p-value: < 2.2e-16

You fit: $$
\text{pcgmp}_i = \beta_0 + \beta_1 \log(\text{pop}_i) + \epsilon_i
$$

R output (relevant parts): - Estimate (Intercept): $-23306.2$, Std.
Error: $4957.1$ - Estimate $\beta_1$ (log(pop)): $4449.8$, Std. Error:
$390.9$ - Residual standard error (RSE): $7929$ on 364 df

### What the standard errors mean

A standard error (SE) is the estimated standard deviation of the
sampling distribution of the estimator, reflecting how much the estimate
would typically fluctuate over repeated samples with the same design.

1.  Slope SE $390.9$:
    - Interpretation: If we repeatedly sampled regions (or
      jurisdictions) from the same super-population and refit the model
      each time, the slope estimates for $\beta_1$ would typically vary
      by about $391$ around the true $\beta_1$.
    - Practical interpretation of the slope itself: A 1-unit increase in
      $\log(\text{pop})$ (i.e., multiplying population by
      $e \approx 2.718$) is associated with an average increase of about
      $4449.8$ currency units in per-capita GMP.
    - Often more interpretable: effect of doubling population. A
      doubling corresponds to adding $\log 2 \approx 0.693$: $$
      \text{Estimated increase for doubling} = 4449.8 \times 0.693 \approx 3081
      $$ Standard error for that contrast: $$
      390.9 \times 0.693 \approx 271
      $$ So doubling population is associated with an estimated increase
      of about $3080$ (SE $\approx 270$) in per-capita GMP.
2.  Intercept SE $4957.1$:
    - The intercept is the expected per-capita GMP when
      $\log(\text{pop})=0\Rightarrow \text{pop}=1$. This population
      value is far outside the realistic range (an “extrapolation”), so
      while mathematically the SE tells us the precision of that
      extrapolated baseline, substantively the intercept has little
      direct meaning here.
    - Still, the SE indicates that across repeated samples
      (hypothetically including such tiny populations), the fitted
      intercept would vary by about $5000$ units.

### (Optional) 95% confidence intervals

Using large degrees of freedom, $t_{0.975,364} \approx 1.97$.

- Slope: $$
  4449.8 \pm 1.97 \times 390.9 \approx 4449.8 \pm 770 \Rightarrow (3680,\ 5220)\ \text{(approx)}
  $$ Interpretation: We are about 95% confident that the true increase
  in per-capita GMP per 1-unit increase in $\log(\text{pop})$ lies
  between roughly $3.7\text{k}$ and $5.2\text{k}$.

- Intercept: $$
  -23306.2 \pm 1.97 \times 4957.1 \approx -23306.2 \pm 9770 \Rightarrow (-33000,\ -13500)\ \text{(approx)}
  $$ Again, limited substantive value due to extrapolation.

### Residual standard error (context)

The residual standard error $7929$ estimates $\sigma$, the typical
unexplained deviation of per-capita GMP from the regression line, after
accounting for $\log(\text{pop})$. Compared with the slope magnitude
($\approx 4450$ per log-unit), it suggests meaningful—but not
overwhelming—explanatory power (consistent with R-squared
$\approx 0.26$).

### Summary Interpretation

- The slope is estimated very precisely (SE much smaller than the
  estimate), giving a large t-statistic ($11.38$) and extremely small
  p-value.
- Population size (on the log scale) is a statistically and practically
  meaningful predictor of per-capita GMP.
- The intercept’s SE is large relative to its (extrapolated) context,
  reinforcing that intercept interpretation should be cautious.

------------------------------------------------------------------------

## Adding on the Normality Assumption

So far, we have not made any distributional assumptions. Given
additional distributional assumptions, we will have additional
properties that will be useful for inference. We’ll focus our discussion
on inference for $\beta_1$. iid

**Exercise 0.3**. Assume $Y_i =\beta_0 + \beta_1X_i + \epsilon_i$, where
$\epsilon_i ∼ N(0, \sigma_2)$. Because $\beta_1 = \sum^n_{i=1}k_iY_i$,
what can we say about the distribution of $β_1$ in this case?

From now on, when we assume $\epsilon_i ∼ N(0, \sigma_2)$, we will call
the resulting model the normal simple linear regression model.

A key result: Under the normal simple linear regression model, $$
(\hat\beta_1 - \beta_1) / \hat{\text{SE}}(\hat\beta_1) = (\hat\beta_1 - \beta_1)/\bigg(\frac{\hat\sigma^2}{\sum_i(X_i-\bar X)^2}\bigg)^{1/2}\sim t_{n-2}
$$

Note that the ratio $\beta_1/SE(\beta_1)$ is named the “t value” in the
R output.

**Exercise 0.4**. Comment on the practical interpretation of the “t
value.”

When the degrees of freedom is small, the t distribution has heavier
tails than a standard Normal. Meanwhile, as the degrees of freedom
becomes large, the t distribution looks more and more like a standard
Normal.

In what follows, we’ll consider confidence intervals based on
t-distribution quantiles. These will be very similar to Normal quantiles
for large samples.

**Exercise 0.5**. Use the above “key result” to show that $$
\beta_1 \pm t_{1-\alpha/2,n-2}\hat{\text{SE}}(\hat\beta_1)
$$

is a $100(1-\alpha)\%$ confidence interval for $\beta_1$.

(Here $t_{1-\alpha/2,n-2}$) refers to other $1-\alpha$ quantile of the
$t_{n-2}$ distribution. It can be calculated in R with
`qt(1 - alpha/2, df = n - 2)`).

**Example 0.1**. Here we construct 95% confidence intervals for the
linear regression used in the BEA example.

First, we can look up the appropriate $t$ quantile for $\alpha=0.05$,
and find $\hat\beta_1$ and $\hat{\text{SE}(\beta_1)}$:

``` r
#compute t quantile
t_quant <- qt(1 - 0.05/2, df = nrow(bea) - 2)
#obtain beta-hat
betaHat = coef(bea_fit)["log(pop)"]
#obtain estimated SE
betaHat.se = coef(summary(bea_fit))["log(pop)", "Std. Error"]
#manual calculation of CI:
c(betaHat - betaHat.se*t_quant, betaHat + betaHat.se*t_quant)
```

    ## log(pop) log(pop) 
    ## 3681.029 5218.486

``` r
#using confint() for CI:
confint(bea_fit)
```

    ##                  2.5 %     97.5 %
    ## (Intercept) -33054.300 -13558.099
    ## log(pop)      3681.029   5218.486

------------------------------------------------------------------------

## Reporting and Interpreting Estimates

With confidence intervals, we now have what we need to report our re-
gression estimates with uncertainty. We must communicate the uncertainty
clearly to readers so they can understand our results.

### Interpreting $\hat\beta_0$ and $\hat\beta_1$

To report estimates, we must first be able to interpret what they mean.
Consider simple linear regression with the model $$
Y_i = \beta_0 + \beta_1X_i+\epsilon_i
$$ We use least squares to obtain $\hat\beta_0$ and $\hat\beta_1$ for a
particular sample of data.

**Exercise 0.6.** Interpret $\hat\beta_0$ and $\hat\beta_1$

------------------------------------------------------------------------

### Reporting Estimates with Uncertainty

Uncertainty is important, because simply reporting “$\beta_1 = 0.42$”
may conceal a great deal. Results should always be reported with
confidence intervals, or at least SEs, so readers can see the scale of
uncertainty.

**Example 0.2**. Suppose we surveyed CMU students to ask (a) how many
hours they sleep per night and (b) their GPA. We fit a regression using
hours of sleep as the covariate and GPA as the outcome, hoping to under-
stand how sleep habits relate to grades.

**Exercise 0.7.** Suppose we obtain $\hat\beta_1 = 0.4$. Interpret this
result in context, and show why the size of the confidence interval
matters a lot.

**Exercise 0.8.** Suppose instead that we obtained size of the
confidence interval matters a lot. $\hat\beta_1 = 0.002$. Show why the
size of the confidence interval matters a lot.

------------------------------------------------------------------------

## Hypothesis Testing for $\beta_1$

Review: There are five key components to a statistical hypothesis
test: 1. Null hypothesis $H_0$: Tentative assumption that an effect or
parameter is “null” (or equal to zero). We’ll assess to what extent the
data are consistent with $H_0$. For example: $H_0:\beta_1 = 0$. 2.
Alternative hypothesis HA: Characteristic about an effect or parameter
we assume if we reject the null hypothesis. For example: HA :
$\beta_1\ne 0$. 3. Test statistic: Measures how consistent the data are
with $H_0$. Ideally, (1) the more “false” $H_0$ becomes, the more the
test statistic changes; and (2) we know its distribution when H0 is
true. 4. Rejection region: Range of test statistic values for which we
reject $H_0$. 5. Significance level α: Determines size of rejection
region and frequency we’re willing to falsely reject when $H_0$ is true.

**Exercise 0.9.** To test H0 : $\beta_1 = 0$, a natural test statistic
is $T = \hat\beta_1 / \hat{\text{SE}(\hat\beta_1)}$. Why? Furthermore,
what’s the practical value of testing $H_0:\beta_0 = 0$?

Recall that we need to consider separately the one-sided and two-sided
hypothesis tests. In this case,

| If testing… | reject $H_0$ if … |
|----|----|
| $H0 : \beta_1 = 0$ versus $H_A : \beta_1 > 0$ | $T > t_{1−α,n−2}$ |
| $H0 : \beta_1 = 0$ versus $H_A : \beta_1 < 0$ | $T < −t_{1−α,n−2}$ |
| $H0 : \beta_1 = 0$ versus $H_A : \beta_1\ne0$ | $T > t_{1−α/2,n−2}$ or $T <−t_{1−α/2,n−2}$ |

Importantly, the p-value can be calculated using the appropriate tail
probability. We reject $H_0$ if $p < α$.

**Exercise 0.10**. Depict the appropriate tail probability to be
calculated in each of the three possibilities given above. Furthermore,
clarify what distribution you use to compute the tail probability.
