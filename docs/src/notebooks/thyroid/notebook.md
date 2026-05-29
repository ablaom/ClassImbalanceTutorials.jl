```@meta
EditURL = "notebook.jl"
```

# An Introduction to Machine Learning in Quantitative Medicine using Julia

> Anthony D. Blaom, May 2026

This tutorial presents an end-to-end application of machine learning to the problem of
creating an automated test for the presence of some disease. Specifically, we compare
several tests for hypothyroidism based on machine learning models, trained using a
well studied Garavan Institute dataset [@Quinlan_86].

We assume readers are familiar with basic statistics and in particular: (i) the
distinction between in-sample and out-of-sample estimation of a model's
performance; (ii) the idea overfitting; and (iii) cross validation.

The principal software tool will be the [MLJ.jl](https://juliaml.ai) machine learning
toolbox, written for the Julia programming language [@Blaom_etal_20]. For resources on
learning Julia, installing Julia packages, etc, the reader is referred to the Julia
[website](https://julialang.org).

## Overview

### Class imbalance and performance measures

In medical binary classification tasks, such as disease screening and toxicity
detection, the training data is typically unbalanced, with many more negative outcomes
than positive ones. Additionally, naive measures of task "success", such as the
proportion of correct predictions, are inappropriate, and we should turn to more
nuanced metrics.  These issues are explicitly addressed here.

### The model building workflow

The Garavan Institute dataset is actually several datasets, with different amounts of
preprocessing. We work directly with the full "allhypo" dataset, which is unprocessed
and includes missing values. By **features** (also known as covariates or inputs), we
will mean potential predictors of disease, like gender or blood test results, that are
recorded in the training dataset, and available for applying our test on new
subjects. The core algorithm that learns how to distinguish between "positive" and
"negative" test outcomes is known as a **supervised learner**. Examples you may have
encountered before are logistic regression and decision trees. However, for a complete
solution, we must wrap our learner in several other workflows to be detailed in this
tutorial:

- **missing value imputation**

- **outlier detection** and removal

- **coercion** of feature data types, to ensure correct "scientific" interpretation by
  MLJ models

- **One-hot encoding**, which transforms categorical features (such
  as the source of a referall) to continuous ones (needed in training many models)

- Training of supervised models that report feature importances, with a view to applying
  **recursive feature elimination**, which can improve outcomes for other supervised
  models

- Constructing model **pipelines**, which combine data pre-processing with a supervised
  learner, to streamline workflows and mitigate **data leakage**

- Estimation of the predictive **performance** of model pipelines, and associated
- **uncertainties**

- **Tuning** (optimization) of user-defined hyperparameters to further boost performance

- **Oversampling** training data to correct for class imbalance

- **Comparison** of a battery of supervised learning models

- Plotting **precision-recall curves** for additional insight

- **Thresholding:** Transforming a probabilistic predictor into a predictor of
  point "positive"/"negative" outcomes, by determining an appropriate threshold for the
  "positive" class probability.

- Testing the final model on a holdout test set

## Julia setup

> Launching a specified Julia version and activating an appropriate set of third-party
> packages.

We suppose the user is interacting with Julia using the commandline interface known as
the **REPL**. This is typically launched by running the `julia` command from a terminal
window, which delivers a `julia>` prompt, after which you can enter all the commands in
this notebook.  Before doing so, **ensure version 1.12** or higher is used by running
`juliaup default 1.12` or similar from your operating system terminal.

Check the Julia version by entering `versioninfo()` at the `julia>` prompt. Quit the
REPL by entering `exit()` or Ctrl-D.

You can also interact with Julia using a interactive development environment such as
VSCode or emacs, or using notebook-style interfaces such as Juptyer (python-driven),
Pluto, or Neptune (pure Julia). These are not discussed here.

### Package management

A **package environment** is a set of specific third party packages known to be mutually
compatible, and compatible with the version of Julia you are using. Typically, each
project you work on will have it's own environment, which you will want to share with
coworkers to ensure workflow reproducibility. Although you can start to learn Julia
without worrying too much about package management, we recommend you invest the effort
to understand Julia package packagement before carrying out serious work, including this
tutorial.

To create a package environment compatible with the code in this tutorial, you will need
to:

1. Clone the GitHub repository at
  "https://github.com/ablaom/ClassImbalanceTutorials.jl/tree/master".

2. In a fresh Julia REPL, edit the right-hand side of following definition to point to
  the root directory of your clone: `path_to_clone =
  "path/to/ClassImbalanceTutorials.jl/tree/master/"`.

3. Still in the REPL, run `using Pkg; Pkg.activate(path_to_local); Pkg.resolve();
  Pkg.instantiate()`.

**Going deeper.** (These sections can be skipped on a first reading.) In Step 3, Julia
looks for a file named `Project.toml` in `path_to_local`, which contains a list of any
packages we want to use together, and any constraints on the verions we want to
enforce. The file also includes pointers to any other Project.toml files (under
`[workspaces]`) whose requirements we additionally want to include. The `resolve`
command determines exact versions of all packages that simultaneously meet the
requirements, and these versions are stored in a file called `Manifest.toml` at the top
level. The `instantiate` command downloads those packages to your computer, in a hidden
file named `.julia` in your home directory.

If you encounter issues, you can try manually installing the packages you need in a new
environment, created with `Pkg.activate("my_env")`, based on the output printed below:

````@julia
using Pkg
Pkg.status()
````

````
Status `~/work/ClassImbalanceTutorials.jl/ClassImbalanceTutorials.jl/docs/src/notebooks/thyroid/Project.toml`
  [1520ce14] AbstractTrees v0.4.5
  [336ed68f] CSV v0.10.16
  [a93c6f00] DataFrames v1.8.2
  [7806a523] DecisionTree v0.12.4
  [f6006082] EvoTrees v0.18.5
  [c709b415] Imbalance v0.1.7
  [add582a8] MLJ v0.23.2
  [a7f614a8] MLJBase v1.13.0
  [c6f25543] MLJDecisionTreeInterface v0.4.4
  [094fc8d1] MLJFlux v0.6.7
  [61c7150f] MLJLIBSVMInterface v0.2.2
  [6ee0df7b] MLJLinearModels v0.10.1
  [eff96d63] Measurements v2.14.1
  [636a865e] NearestNeighborModels v0.2.3
  [262411bb] OutlierDetection v0.4.2
  [51249a0a] OutlierDetectionNeighbors v0.1.5
  [91a5bcdd] Plots v1.41.6
  [860ef19b] StableRNGs v1.0.4
  [2913bbd2] StatsBase v0.34.10
  [f43a241f] Downloads v1.7.0

````

For example, to add MLJ v0.23.2, do `Pkg.add("MLJ@0.23.2").

**Going deeper.** In this case you get an *exact* replica of the top-level dependencies
used to create and test this tutorial. In Julia one can also pass around an exact
version of *all* packages (not just top-level dependencies) using the Manifest.toml, but
unfortunately there can be small differences in the manifest across platforms, so this
only works reliably if you stay with the same platform (e.g., silicon Mac).

See the [package management documentation](https://pkgdocs.julialang.org/v1/) for
more on package management.

## Downloading a dataset

Here's how we dowload our thyroid dataset:

## Models and machines

> An introduction to the principal actors in an MLJ workflow

The word "model", ubiquitous in statistics and machine learning, has a specific meaning
in MLJ documentation and in this notebook. A **model** is an object encapsulating a
machine learning algorithm, such as least-squares fitting of a polynomial, or logistic
regression, together with parameters that must be specified before the algorithm can be
executed, such as the degree of the polynomial. These parameters are the model
**hyperparameters**. In statistics a "model" generally includes the *learned
parameters*, like the coefficients of the polynomial, but in MLJ these are external to
the model.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

