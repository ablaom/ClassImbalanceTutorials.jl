# # An Introduction to Machine Learning in Quantitative Medicine using Julia

# This tutorial presents an end-to-end application of machine learning to the problem of
# creating an automated test for the presence of some disease. Specifically, we compare
# several tests for hypothyroidisem based on machine learning models, trained using the
# well studied Garavan Institute database [@Quinlan_86].

# We assume readers are familiar with basic statistics and in particular: (i) the
# distinction between in-sample and out-of-sample estimation of a model's
# performance; (ii) the idea overfitting; and (iii) cross validation.

# The principal software tool will be the [MLJ.jl](https://juliaml.ai) machine learning
# toolbox, written for the Julia programming language [@Blaom_etal_20]. For resources on
# learning Julia, installing Julia packages, etc, the reader is referred to the Julia
# [website](https://julialang.org).


# ## Overview

# ### Class imbalance and performance measures

# In medical binary classification tasks, such as disease screening and toxicity
# detection, the training data is typically unbalanced, with many more negative outcomes
# than positive ones. Additionally, naive measures of task "success", such as the
# proportion of correct predictions, are inappropriate, and we should turn to more
# nuanced metrics.  These issues are explicitly addressed here.

# ### The model building workflow

# The Thyroid Database includes several datasets, with a variety of preprocessing. We work
# with directly with the full "allhypo" dataset, which is unprocessed and includes missing
# values. By **features** (a.k.a., covariates, inputs), we mean potential predictors of
# disease, like gender or blood test results, that are recorded in the training dataset,
# and available for executing our test on new subjects. The core algorithm that learns how
# to distinguish between "positive" and "negative" test outcomes is known as a
# **supervised learner**. By **features** (a.k.a., covariates, inputs) we mean things like
# gender or blood test results, which are predictors of disease and available for
# executing our test.  Examples you may have encountered before are logistic regression
# and decision trees. However, for a complete solution, we must wrap our learner in
# several other workflows to be detailed in this tutorial:

# - **missing value imputation**

# - **outlier detection* and removal

# - **coercion** of feature data types, to ensure correct "scientific" interpretation by
#   MLJ models

# - **One-hot encoding**, which transforms categorical features (such
#   as the source of a referall) to continuous ones (needed in training many models)

# - Training of supervised models that report feature importances, with a view to applying
#   **recursive feature elimination**, which can improve outcomes for other supervised
#   models

# - Constructing model **pipelines**, which combine data pre-processing with a supervised
a a#   learner, to streamline workflows and mitigate **data leakage**

# - Estimation of the **performance** of model pipelines, and associated **uncertainties**

# - **Tuning** (optimization) of user-defined hyperparameters to further boost performance

# - **Oversampling** training data to correct for class imbalance

# - **Comparison** of a battery of supervised learning models

# - Plotting precision-recall curves for additional insight

# - **Thresholding:** Transforming a probabilistic predictor into a predictor of
#   "positive"/"negative" outcomes, by determining an appropriate threshold for the
#   "positive" class

# - Testing the final model on a holdout test set

# ## Julia setup

# > Launching a specified Julia version and activating an appropriate set of third-party
# > packages.

# We suppose the user is interacting with Julia using the commandline interface known as
# the **REPL**. This is typically launched by running the `julia` command from a terminal
# window, which delivers a `julia>` prompt, after which you can enter all the commands in
# this tutorial.  Before doing so, ensure version 1.12 is used by running `juliaup default
# 1.12` from your operating system terminal.

# Check the Julia version by entering `versioninfo()` at the `julia>` prompt. Quit the
# REPL by entering `exit()` or Ctrl-D.

# You can also interact with Julia using a interactive development environment such as
# VSCode or emacs, or using notebook-style interfaces such as Juptyer (python-driven) or
# Pluto (pure Julia). These are not discussed here.

# A package **environment** is a set of third party packages, and a version number for
# each package that ensures they are mutually compatible. Typically each project you work
# on will have it's own environment. In this case we'll download and activate the
# environment used to develop this tutorial:





# ## Downloading a dataset

# Here's how we dowload our thyroid dataset

# # Models and machines

# > An introduction to the principal actors in an MLJ workflow

# The word "model", ubiquitous in statistics and machine learning, has a specific meaning
# in MLJ documentation and in this tutorial. A **model** is an object encapsulating a
# machine learning algorithm, such as least-squares fitting of a polynomial, or logistic
# regression, together with parameters that must be specified before the algorithm can be
# executed, such as the degree of the polynomial. These parameters are the model
# **hyperparameters**. In statistics a "model" generally includes the *learned
# parameters*, like the coefficients of the polynomial, but in MLJ these are external to
# the model.
