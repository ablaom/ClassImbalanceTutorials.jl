# # An Introduction to Machine Learning in Quantitative Medicine using Julia

# > Anthony D. Blaom, May 2026

# This tutorial presents a detailed, end-to-end application of machine learning to the
# problem of creating an automated test for the presence of some disease. Specifically, we
# compare several tests for hypothyroidism based on machine learning models, trained using
# a well studied Garavan Institute dataset [@Quinlan_86].

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
# than positive ones. Additionally, naive measures of task success, such as the
# proportion of correct predictions, are inappropriate; we should turn to more
# nuanced metrics.  These issues are explicitly addressed here.

# ### The model building workflow

# The Garavan Institute dataset is actually several datasets, with different amounts of
# preprocessing. We work directly with the full "allhypo" dataset, which is unprocessed
# and includes missing values. By **features** (also known as covariates or inputs), we
# will mean potential predictors of disease, like gender or blood test results, that are
# recorded in the training dataset, and available for applying our test to new
# subjects. The core algorithm that learns how to distinguish between "positive" and
# "negative" test outcomes is known as a **supervised learner**. Examples you may have
# encountered before are logistic regression and decision trees. However, for a complete
# solution, we must wrap our learner in several other workflows to be detailed in this
# tutorial:

# - **missing value imputation**

# - **outlier detection** and removal

# - **coercion** of feature data types, to ensure correct "scientific" interpretation by
#   MLJ models

# - **one-hot encoding**, which transforms categorical features (such
#   as the source of a referall) to continuous ones (needed in training many models)

# - training of supervised models that report feature importances, with a view to applying
#   **recursive feature elimination**, which can improve outcomes for other supervised
#   models

# - constructing model **pipelines**, which combine data pre-processing with a supervised
#   learner, to streamline workflows and mitigate **data leakage**

# - constructing approximate confidence intervals expected **predictive
#   performance** of model pipelines

# - **tuning** (optimization) of user-defined hyperparameters to further boost performance

# - **oversampling** training data to correct for class imbalance

# - **comparison** of several supervised learning models (model selection)

# - plotting **precision-recall curves** for additional insight

# - **thresholding:** Transforming a probabilistic predictor into a predictor of
#   point "positive"/"negative" outcomes, by determining an appropriate threshold for the
#   "positive" class probability.

# - **testing** the final model on a holdout test set

# ## Julia setup

# > Launching a specified Julia version and activating an appropriate set of third-party
# > packages.

# We suppose the user is interacting with Julia using the commandline interface known as
# the **REPL**. This is typically launched by running the `julia` command from a terminal
# window, which delivers a `julia>` prompt, after which you can enter all the commands in
# this tutorial.  Before doing so, **ensure version 1.12** or higher is used by running
# `juliaup default 1.12` or similar from your operating system terminal.

# Check the Julia version by entering `versioninfo()` at the `julia>` prompt. Quit the
# REPL by entering `exit()` or Ctrl-D.

# You can also interact with Julia using a interactive development environment such as
# VSCode or emacs, or using notebook-style interfaces such as Juptyer (python-driven),
# Pluto, or Nee. We do not discuss these here, except to say that Pluto's "reactive"
# feature is not well-suited to machine learning applications with long training times.

# ### Package management

# A **package environment** is a set of specific third party packages known to be mutually
# compatible, and compatible with the version of Julia you are using. Typically, each
# project you work on will have it's own environment, which you will want to share with
# coworkers to ensure workflow reproducibility. Although you can start to learn Julia
# without worrying too much about package management, we recommend you invest the effort
# to understand Julia package packagement before carrying out serious work, including this
# tutorial.

# To create a package environment compatible with the code in this tutorial, you will need
# to:
#
# 1. Clone the GitHub repository at [https://github.com/ablaom/ClassImbalanceTutorials.jl/tree/master](https://github.com/ablaom/ClassImbalanceTutorials.jl/tree/master).
#
# 2. In a fresh REPL session, edit the right-hand side of following definition to point to the root directory of your clone: `path_to_clone = "path/to/ClassImbalanceTutorials.jl/tree/master/"`.
#
# 3. Still in the REPL, run `using Pkg; Pkg.activate(path_to_local); Pkg.resolve(); Pkg.instantiate()`.

# **Going deeper.**¹ In Step 3, Julia looks for a file named `Project.toml` in
# `path_to_local`, which contains a list of packages we want to use together, and any
# constraints on the verions we want to enforce. The file also includes pointers to any
# other Project.toml files (under `[workspaces]`) whose requirements we additionally want
# to include. The `resolve` command determines exact versions of all packages that
# simultaneously meet the requirements, and these versions are stored in a file called
# `Manifest.toml` at the top level. The `instantiate` command downloads those packages to
# your computer, in a hidden folder named `.julia` in your home directory.

# If you encounter issues, you can try manually installing the packages you need in a new
# environment, created with `Pkg.activate("my_env")`, based on the output printed below:

using Pkg
Pkg.status()

# For example, to add MLJ v0.23.2, do `Pkg.add("MLJ@0.23.2")`.

# **Going deeper.** In this case you get an *exact* replica of the top-level dependencies
# used to create and test this tutorial. In Julia one can pass around an exact version of
# *all* packages (not just top-level dependencies) using the Manifest.toml file,
# but unfortunately there can be small differences across platforms, so this only works
# reliably if you are sticking with the same platform (e.g., silicon Mac).

# See the [package management documentation](https://pkgdocs.julialang.org/v1/) for
# more on package management.

# ### Making our workflow reproducible

# As many algorithms depend on pseudo-random number generators (RNG's), and as the
# behaviour of the default RNG can depend on the Julia version, we take some steps in the
# tutorial to allow us to generate an identical RNG, stable across Julia versions, every
# time we need one (although it does not have optimal performance).

import StableRNGs
const SEED = 234
stable_rng() = StableRNGs.StableRNG(SEED)

# Here's a demonstation:

rng = stable_rng()
rand(rng) # this call mutates `rng` state

#-

rand(rng)

#-

rng = stable_rng() # reset the rng
rand(rng)          # same result as above

# ## Downloading a dataset

# > Grabbing a dataset from the internet and manifesting the data in a convenient in-memory form

using Downloads
import CSV        # for parsing CSV files
import DataFrames # for manipulating in-memory tabular data

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"*
    "thyroid-disease/allhypo.data"
path = Downloads.download(url)

# Entering these lines of code downloads the dataset to our computer's disk. This file is
# a comma-separated data file, but without a header. The header can be gleaned from
# [https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disea/allhypo.names,which](https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disea/allhypo.names,which),
# which has further information about the data set. We manually provide the header:

header = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source", "diagnosis",
]
data = CSV.read(path, DataFrames.DataFrame; header, missingstring="?");

# This last line reads the data into a dataframe, an in-memory table for which we have
# many convenient methods for manipulating, provided by the DataFrames.jl package. Let's
# inspect the first three rows:

first(data, 3)

# We can see there is a problem with the `diagnosis` column:

data.diagnosis[15:21] # rows 15 through 21

# Each entry has a trailing string which corresponds to a patient identifier we won't
# need. We introduce a function to to trim such an entry, and at the same time collapse
# all hypothyroidisem variants into a single `"positive"` class:

function fix(class)
    trimmed = split(class, ".|") |> first
    if trimmed == "negative"
        return trimmed
     end
    return "positive"
end
fix("compensated hypothyroid.|3169") # demo

# Applying it to every entry in the `diagnosis` column:

data.diagnosis = fix.(data.diagnosis);
first(data.diagnosis, 3)

#-

using MLJ
schema(data)

#-

describe(data)

# Models and machines

# > An introduction to the principal actors in an MLJ workflow

# The word "model", ubiquitous in statistics and machine learning, has a specific meaning
# in MLJ documentation and in this notebook. A **model** is an object encapsulating a
# machine learning algorithm, such as least-squares fitting of a polynomial, or logistic
# regression, together with parameters that must be specified before the algorithm can be
# executed, such as the degree of the polynomial. These parameters are the model
# **hyperparameters**. In statistics a "model" generally includes the *learned
# parameters*, like the coefficients of the polynomial, but in MLJ these are external to
# the model.

# ---

# ¹Read "Going deeper" sections if you feel you need more detail to fill out your mental
# model of what is going in the tutorial. You may want to skip them on a first reading.
