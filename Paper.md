\documentclass[11pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage{amsmath, amssymb, amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{float}
\usepackage{enumitem}
\setlist{nosep}
\hypersetup{colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue}

\title{Gibbs Sampling for Bayesian Alpha Signal Combination}
\author{Vidhi Jain}
\date{STAT 31511: Monte Carlo Simulations, University of Chicago}

\begin{document}
\maketitle

\begin{abstract}
Quantitative researchers often combine several noisy signals to forecast future returns. A classical approach is to estimate one vector of signal weights by least squares or to assign equal weights, but these approaches hide the uncertainty in the estimated coefficients. This paper studies Bayesian alpha signal combination as an application of Gibbs sampling. We formulate signal combination as a Bayesian linear regression problem with a Gaussian likelihood, Gaussian prior on signal weights, and inverse-gamma prior on residual variance. Under this conjugate specification, the full conditional distributions are available in closed form, making the Gibbs sampler simple to implement and interpret. A synthetic alpha experiment is used to illustrate the sampler, posterior coefficient uncertainty, posterior predictive intervals, and a simple out-of-sample sign strategy. The goal is not to propose a production trading strategy, but to show how Gibbs sampling can support probabilistic signal weighting in a setting that resembles systematic quant research.
\end{abstract}

\section{Introduction}

A central problem in systematic investment research is deciding how to combine multiple noisy predictors of returns. A researcher may have candidate signals based on momentum, short-term reversal, volatility expansion, volume shocks, or moving-average rules. Some of these signals may contain predictive information, while others may be redundant or pure noise. A natural first model is a linear forecast of the next-period return as a weighted average of these signals. The main statistical question is then how to estimate the signal weights and how much confidence to place in each weight.

This project studies this problem from the perspective of Monte Carlo methods. In particular, it uses Gibbs sampling to draw from the posterior distribution of the signal weights and residual variance. The project is organized like an application-focused extension of the Gibbs sampling chapter in the lecture notes. The general theory of Gibbs sampling isn't repeated, instead the full conditional distributions for a Bayesian regression model is derived and then the resulting algorithm in a small alpha-signal experiment is implemented. 

The academic goal is to demonstrate the mechanics of Gibbs sampling in a model where all conditional distributions are tractable. The applied goal is to show why posterior sampling may be useful in quant research. A point estimate answers the question ``what is the best single estimate of each signal weight?'' A Bayesian sampler gives a richer answer: it produces a distribution over plausible weights, revealing which signals are stable, which are uncertain, and how much residual noise remains in the return forecast.

\section{Signal-Combination Model}

Let $y_t$ denote the next-period return that we want to forecast, and let
\[
    x_t = (x_{t1},\ldots,x_{tk})^\top
\]
be a vector of $k$ standardized signals observed at time $t$. In the numerical example, the signals are named momentum, reversal, volatility breakout, volume shock, RSI, and moving-average crossover. These names are used to keep the example close to financial research practice, but the data are simulated so that the true coefficients are known.

The model is the Bayesian linear regression
\begin{equation}
    y_t = \alpha + x_t^\top \beta + \epsilon_t,
    \qquad \epsilon_t \sim N(0,\sigma^2),
    \label{eq:linear}
\end{equation}
where $\beta \in \mathbb{R}^k$ is the vector of signal weights. Define the design matrix $X \in \mathbb{R}^{T\times p}$, where $p=k+1$ because the first column is an intercept. Let
\[
    \theta = (\alpha,\beta_1,\ldots,\beta_k)^\top.
\]
Then the model can be written compactly as
\begin{equation}
    y \mid \theta,\sigma^2 \sim N(X\theta,\sigma^2 I_T).
\end{equation}

The prior is chosen to be conjugate:
\begin{align}
    \theta \mid \sigma^2 &\sim N(0,\sigma^2 V_0), \\
    \sigma^2 &\sim \mathrm{IG}(a_0,b_0).
\end{align}
Here $V_0$ is a prior covariance matrix. In the implementation, $V_0=cI_p$ for a positive constant $c$. Smaller $c$ implies stronger shrinkage of signal weights toward zero. This is useful in alpha research because many candidate signals are weak and may look predictive in sample only by chance.

\section{Posterior Distribution}

The posterior distribution is proportional to likelihood times prior:
\begin{equation}
    \pi(\theta,\sigma^2\mid y,X)
    \propto
    \pi(y\mid \theta,\sigma^2,X)
    \pi(\theta\mid \sigma^2)
    \pi(\sigma^2).
\end{equation}
Although one can derive the joint posterior analytically in this conjugate case, Gibbs sampling remains pedagogically useful because it illustrates the key idea of sampling a complicated joint distribution through simpler conditional distributions.
A useful interpretation is the Bayesian bias--variance trade-off embedded in the prior covariance $V_0$. If $V_0$ is diffuse, the posterior behaves similarly to ordinary least squares, allowing the data to dominate. If $V_0$ is tighter, coefficients are shrunk toward zero, which is desirable in alpha research where many apparent predictors are weak or unstable. This connects directly to the practical quant problem of avoiding overfitting when evaluating many candidate signals.

From a Monte Carlo perspective, this model is intentionally chosen because the full conditional distributions are available in closed form. In non-conjugate financial models, such as stochastic volatility or regime-switching specifications, exact conditional sampling is often impossible and one must combine Gibbs steps with Metropolis updates. The present model therefore serves as a clean benchmark for understanding Gibbs mechanics before moving to harder applications.


The Gaussian likelihood contributes the term
\begin{equation}
    \exp\left\{-\frac{1}{2\sigma^2}(y-X\theta)^\top(y-X\theta)\right\}.
\end{equation}
The Gaussian prior for $\theta$ contributes
\begin{equation}
    \exp\left\{-\frac{1}{2\sigma^2}\theta^\top V_0^{-1}\theta\right\}.
\end{equation}
Combining these terms and completing the square gives the conditional distribution of $\theta$ given $\sigma^2$.

Let
\begin{align}
    V_n &= (V_0^{-1}+X^\top X)^{-1}, \\
    m_n &= V_n X^\top y.
\end{align}
Then
\begin{equation}
    \theta \mid \sigma^2,y,X \sim N(m_n,\sigma^2 V_n).
    \label{eq:theta_cond}
\end{equation}
This conditional is a multivariate normal distribution. It is easy to sample from using a Cholesky factorization of $V_n$.

The conditional distribution of $\sigma^2$ follows from collecting the powers and exponential terms in $\sigma^2$. Conditional on $\theta$,
\begin{equation}
    \sigma^2 \mid \theta,y,X \sim \mathrm{IG}\left(
    a_0+\frac{T+p}{2},\,
    b_0+\frac{1}{2}\left[(y-X\theta)^\top(y-X\theta)+\theta^\top V_0^{-1}\theta\right]
    \right).
    \label{eq:sigma_cond}
\end{equation}
Equations \eqref{eq:theta_cond} and \eqref{eq:sigma_cond} define the Gibbs sampler.

\section{Gibbs Sampling Algorithm}

The Gibbs sampler alternates between drawing the coefficient vector and drawing the variance. Each update is exact from the corresponding full conditional distribution.

\begin{algorithm}[H]
\caption{Gibbs sampler for Bayesian alpha signal combination}
\begin{algorithmic}[1]
\State Input: design matrix $X$, return vector $y$, prior parameters $V_0,a_0,b_0$, number of iterations $N$.
\State Initialize $\theta^{(0)}$ and $(\sigma^2)^{(0)}$.
\For{$n=1,\ldots,N$}
    \State Draw $\theta^{(n)} \sim N(m_n, (\sigma^2)^{(n-1)} V_n)$.
    \State Compute $S^{(n)}=(y-X\theta^{(n)})^\top(y-X\theta^{(n)})+(\theta^{(n)})^\top V_0^{-1}\theta^{(n)}$.
    \State Draw $(\sigma^2)^{(n)} \sim \mathrm{IG}\left(a_0+(T+p)/2,\, b_0+S^{(n)}/2\right)$.
\EndFor
\State Discard burn-in draws and use the remaining draws to estimate posterior summaries.
\end{algorithmic}
\end{algorithm}

This algorithm is particularly simple because the full conditionals are standard distributions. The implementation does not use a black-box probabilistic programming package. It directly implements the two conditional updates above, which makes the connection to the Monte Carlo method transparent.


\subsection{Convergence Diagnostics and Monte Carlo Quality}

A Markov chain Monte Carlo method is only useful if the chain reaches its stationary distribution and mixes adequately across the posterior support. In this conjugate regression setting, mixing is generally strong because the posterior is smooth and uni-modal, but diagnostics still matter. The notebook evaluates trace plots for representative coefficients and the residual variance to verify that the chain stabilizes after burn-in and does not display persistent drift.

Autocorrelation is another practical consideration. Consecutive Gibbs draws are dependent because each draw conditions on the previous state. Thinning is therefore used in the implementation to reduce storage burden and serial dependence, though modern practice often emphasizes effective sample size rather than aggressive thinning. For this project, the goal is to improve interpretability rather than computational efficiency, so a conservative burn-in and thinning scheme is acceptable.

A more formal empirical study could compute effective sample size, Monte Carlo standard errors, or Gelman-Rubin diagnostics across multiple chains. These are omitted here to preserve focus, but they would be natural additions in a larger Bayesian workflow.

\section{Numerical Experiment}

The experiment uses simulated data with $T=900$ observations and $k=6$ candidate signals. The signal vector is sampled from a correlated Gaussian distribution and standardized. The next-period return is generated as
\begin{equation}
    y_t=x_t^\top \beta_\star+\epsilon_t,
    \qquad \epsilon_t\sim N(0,0.015^2).
\end{equation}
The true coefficient vector is
\begin{equation}
    \beta_\star=(0.0045,-0.0030,0.0025,0,0.0015,0)^\top.
\end{equation}
Thus, momentum, reversal, volatility breakout, and RSI have true predictive content, while volume shock and moving-average crossover are noise signals. This design allows us to check whether posterior inference separates useful signals from noisy ones.

The first 600 observations are used for training and the remaining 300 observations are used for out-of-sample evaluation. The Gibbs sampler is run for 12,000 iterations, with 3,000 burn-in iterations and thinning by 3. The posterior draws are used to compute coefficient means, credible intervals, posterior predictive means, and simple trading signals.

\section{Results}

Figure \ref{fig:trace} shows trace plots for three selected coefficients. The draws fluctuate around stable regions after burn-in, which is consistent with the sampler exploring its stationary distribution. In this conjugate model, the sampler is well behaved because the full conditional distributions are easy to sample from.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.86\textwidth]{alpha_gibbs_trace.png}
    \caption{Trace plots for selected posterior signal weights. The paths show saved Gibbs draws after burn-in and thinning.}
    \label{fig:trace}
\end{figure}

Figure \ref{fig:coef} displays the posterior mean and 95\% credible interval for each signal coefficient. The strongest signals have intervals shifted away from zero, while the no-content signals are closer to zero. This is the main advantage of the Bayesian approach in this example: the output is not only a ranking of signals, but also a measure of uncertainty around each signal's weight.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.86\textwidth]{alpha_gibbs_coefficients.png}
    \caption{Posterior coefficient estimates for the six alpha signals. Intervals crossing zero correspond to weaker evidence of predictive content.}
    \label{fig:coef}
\end{figure}

Figure \ref{fig:pred} shows the posterior mean forecast and a 90\% posterior band for the first part of the test period. The predictive mean is much less volatile than the realized returns, which is typical in return forecasting. Financial returns contain substantial idiosyncratic noise, so even useful signals may explain only a small part of realized next-period returns.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.86\textwidth]{alpha_gibbs_predictive.png}
    \caption{Posterior predictive mean and uncertainty band for out-of-sample returns.}
    \label{fig:pred}
\end{figure}

To convert forecasts into a simple strategy, define the position
\begin{equation}
    q_t=\mathrm{sign}\left(\mathbb{E}[y_{t+1}\mid y,X,x_t]\right).
\end{equation}
The strategy return is $q_t y_{t+1}$. This backtest is intentionally simple and does not include transaction costs. It is included only to illustrate how posterior signal combination can feed into a decision rule. Figure \ref{fig:bt} compares this Bayesian sign strategy with equal signal weighting and ordinary least squares.

\begin{figure}[H]
    \centering
    \includegraphics[width=0.86\textwidth]{alpha_gibbs_backtest.png}
    \caption{Out-of-sample cumulative returns for simple sign strategies. The purpose is illustrative rather than a claim of live tradability.}
    \label{fig:bt}
\end{figure}


\subsection{Prior Sensitivity}

Because Bayesian inference depends on prior assumptions, it is worth asking whether the qualitative conclusions are robust to moderate prior changes. In this implementation, the coefficient prior is centered at zero with diagonal covariance, reflecting a belief that most alpha signals are weak ex ante. This is consistent with empirical finance intuition: many candidate predictors fail out of sample.

If the prior variance were substantially increased, posterior estimates would move closer to unregularized regression coefficients and credible intervals would widen. If the prior were tightened, weaker signals would be more aggressively shrunk toward zero. The strongest signals in the synthetic experiment should remain directionally stable under reasonable prior perturbations, while marginal signals would be more sensitive. This sensitivity itself is informative because it reveals which predictors rely heavily on modeling assumptions. 

\section{Interpretation}

The Gibbs sampler provides a distribution over signal weights rather than a single estimate. This is important in alpha research because the signal-to-noise ratio in returns is often low. A signal may have a positive point estimate but a wide credible interval, indicating that its apparent effect may not be stable. Conversely, a signal with a posterior interval concentrated away from zero may deserve more attention in follow-up research.

The posterior also supports uncertainty-aware forecasting. Instead of forming only $\hat{y}_{t+1}=x_t^\top \hat{\beta}$, we obtain a posterior distribution of $x_t^\top \beta$. In a richer trading application, this uncertainty could be used to scale positions down when the forecast is uncertain, to filter trades when posterior mass is near zero, or to compare signals across regimes. These extensions would require additional modeling choices, but they follow naturally from the Bayesian framework.

The experiment is deliberately stylized. Because the data are simulated, the true signal weights are known. This is useful for validating the sampler, but real market data would introduce nonstationarity, heavy tails, transaction costs, look-ahead bias concerns, and changing signal efficacy. A production quant-research workflow would need rolling estimation, robust validation, and careful cost modeling. The purpose of this project is narrower: to demonstrate how Gibbs sampling can be used as a Monte Carlo tool for posterior inference in a signal-combination model.


Compared with frequentist point estimation, the Bayesian approach offers several conceptual advantages for signal research. First, uncertainty quantification is built directly into inference rather than added post hoc. Second, shrinkage arises naturally through prior specification, which is particularly useful when the number of candidate features grows relative to the sample size. Third, posterior predictive distributions align well with decision-making under uncertainty, since a trading system rarely benefits from treating a noisy alpha estimate as certain.

However, Bayesian methods also introduce modeling choices that must be justified. Priors can materially affect inference in small samples, computational cost grows in more complex models, and interpretability can suffer if hierarchical structures become too elaborate. These trade-offs are important in production quant environments where speed, robustness, and auditability matter.

\section{Discussion and Bibliography}

This project connects the Gibbs sampler to a practical problem in quantitative finance. The lecture notes present Gibbs sampling as a Markov chain Monte Carlo method that updates one coordinate or block at a time using full conditional distributions. Bayesian linear regression is a natural example because the conditionals are standard distributions and the algorithm can be implemented directly. The alpha-signal framing adds an applied interpretation: the sampled coefficients are posterior beliefs about the usefulness of each predictor.

Several extensions would make the model more realistic. First, one could replace the Gaussian residual model with a $t$ - distribution to handle fat-tailed returns. Second, one could use hierarchical shrinkage priors so that weak signals are pulled more aggressively toward zero. Third, one could allow time-varying coefficients and use sequential Monte Carlo rather than a static Gibbs sampler. Finally, one could apply the method to a cross-section of equities or futures contracts rather than a single simulated return series.

\begin{thebibliography}{9}

\bibitem{notes}
D. Sanz-Alonso and O. Al-Ghattas. \emph{A First Course in Monte Carlo Methods}. University of Chicago lecture notes.

\bibitem{geman}
S. Geman and D. Geman. Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images. \emph{IEEE Transactions on Pattern Analysis and Machine Intelligence}, 1984.

\bibitem{bishop}
C. Bishop. \emph{Pattern Recognition and Machine Learning}. Springer, 2006.

\end{document}
