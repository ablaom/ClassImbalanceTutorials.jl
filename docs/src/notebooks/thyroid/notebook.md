```@meta
EditURL = "notebook.jl"
```

# An Introduction to Machine Learning in Quantitative Medicine using Julia

> Anthony D. Blaom, May 2026

This tutorial presents a detailed, end-to-end application of machine learning to the
problem of creating an automated test for the presence of some disease. Specifically, we
compare several tests for hypothyroidism based on machine learning models, trained using
a well studied Garavan Institute dataset [@Quinlan_86].

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
than positive ones. Additionally, naive measures of task success, such as the
proportion of correct predictions, are inappropriate; we should turn to more
nuanced metrics.  These issues are explicitly addressed here.

### The model building workflow

The Garavan Institute dataset is actually several datasets, with different amounts of
preprocessing. We work directly with the full "allhypo" dataset, which is unprocessed
and includes missing values. By **features** (also known as covariates or inputs), we
will mean potential predictors of disease, like gender or blood test results, that are
recorded in the training dataset, and available for applying our test to new
subjects. The core algorithm that learns how to distinguish between "positive" and
"negative" test outcomes is known as a **supervised learner**. Examples you may have
encountered before are logistic regression and decision trees. However, for a complete
solution, we must wrap our learner in several other workflows to be detailed in this
tutorial:

- **missing value imputation**

- **outlier detection** and removal

- **coercion** of feature data types, to ensure correct "scientific" interpretation by
  MLJ models

- **one-hot encoding**, which transforms categorical features (such
  as the source of a referall) to continuous ones (needed in training many models)

- training of supervised models that report feature importances, with a view to applying
  **recursive feature elimination**, which can improve outcomes for other supervised
  models

- constructing model **pipelines**, which combine data pre-processing with a supervised
  learner, to streamline workflows and mitigate **data leakage**

- constructing approximate confidence intervals expected **predictive
  performance** of model pipelines

- **tuning** (optimization) of user-defined hyperparameters to further boost performance

- **oversampling** training data to correct for class imbalance

- **comparison** of several supervised learning models (model selection)

- plotting **precision-recall curves** for additional insight

- **thresholding:** Transforming a probabilistic predictor into a predictor of
  point "positive"/"negative" outcomes, by determining an appropriate threshold for the
  "positive" class probability.

- **testing** the final model on a holdout test set

## Julia setup

> Launching a specified Julia version and activating an appropriate set of third-party
> packages.

We suppose the user is interacting with Julia using the commandline interface known as
the **REPL**. This is typically launched by running the `julia` command from a terminal
window, which delivers a `julia>` prompt, after which you can enter all the commands in
this tutorial.  Before doing so, **ensure version 1.12** or higher is used by running
`juliaup default 1.12` or similar from your operating system terminal.

Check the Julia version by entering `versioninfo()` at the `julia>` prompt. Quit the
REPL by entering `exit()` or Ctrl-D.

You can also interact with Julia using a interactive development environment such as
VSCode or emacs, or using notebook-style interfaces such as Juptyer (python-driven),
Pluto, or Nee. We do not discuss these here, except to say that Pluto's "reactive"
feature is not well-suited to machine learning applications with long training times.

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

1. Clone the GitHub repository at [https://github.com/ablaom/ClassImbalanceTutorials.jl/tree/master](https://github.com/ablaom/ClassImbalanceTutorials.jl/tree/master).

2. In a fresh REPL session, edit the right-hand side of following definition to point to the root directory of your clone: `path_to_clone = "path/to/ClassImbalanceTutorials.jl/tree/master/"`.

3. Still in the REPL, run `using Pkg; Pkg.activate(path_to_local); Pkg.resolve(); Pkg.instantiate()`.

**Going deeper.**¹ In Step 3, Julia looks for a file named `Project.toml` in
`path_to_local`, which contains a list of packages we want to use together, and any
constraints on the verions we want to enforce. The file also includes pointers to any
other Project.toml files (under `[workspaces]`) whose requirements we additionally want
to include. The `resolve` command determines exact versions of all packages that
simultaneously meet the requirements, and these versions are stored in a file called
`Manifest.toml` at the top level. The `instantiate` command downloads those packages to
your computer, in a hidden folder named `.julia` in your home directory.

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
  [2913bbd2] StatsBase v0.34.11
  [f43a241f] Downloads v1.7.0

````

For example, to add MLJ v0.23.2, do `Pkg.add("MLJ@0.23.2")`.

**Going deeper.** In this case you get an *exact* replica of the top-level dependencies
used to create and test this tutorial. In Julia one can pass around an exact version of
*all* packages (not just top-level dependencies) using the Manifest.toml file,
but unfortunately there can be small differences across platforms, so this only works
reliably if you are sticking with the same platform (e.g., silicon Mac).

See the [package management documentation](https://pkgdocs.julialang.org/v1/) for
more on package management.

### Making our workflow reproducible

As many algorithms depend on pseudo-random number generators (RNG's), and as the
behaviour of the default RNG can depend on the Julia version, we take some steps in the
tutorial to allow us to generate an identical RNG, stable across Julia versions, every
time we need one (although it does not have optimal performance).

````@julia
import StableRNGs
const SEED = 234
stable_rng() = StableRNGs.StableRNG(SEED)
````

````
stable_rng (generic function with 1 method)
````

Here's a demonstation:

````@julia
rng = stable_rng()
rand(rng) # this call mutates `rng` state
````

````
0.48542906587478507
````

````@julia
rand(rng)
````

````
0.09049710430456415
````

````@julia
rng = stable_rng() # reset the rng
rand(rng)          # same result as above
````

````
0.48542906587478507
````

## Downloading a dataset

> Grabbing a dataset from the internet and manifesting the data in a convenient in-memory form

````@julia
using Downloads
import CSV        # for parsing CSV files
import DataFrames # for manipulating in-memory tabular data

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"*
    "thyroid-disease/allhypo.data"
path = Downloads.download(url)
````

````
"/tmp/jl_YLDGxw/allhypo.data"
````

Entering these lines of code downloads the dataset to our computer's disk. This file is
a comma-separated data file, but without a header. The header can be gleaned from
[https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disea/allhypo.names,which](https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disea/allhypo.names,which),
which has further information about the data set. We manually provide the header:

````@julia
header = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_medication",
    "sick", "pregnant", "thyroid_surgery", "I131_treatment", "query_hypothyroid",
    "query_hyperthyroid", "lithium", "goitre", "tumor", "hypopituitary", "psych",
    "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured", "TT4",
    "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured", "TBG",
    "referral_source", "diagnosis",
]
data = CSV.read(path, DataFrames.DataFrame; header, missingstring="?");
````

This last line reads the data into a dataframe, an in-memory table for which we have
many convenient methods for manipulating, provided by the DataFrames.jl package. Let's
inspect the first three rows:

````@julia
first(data, 3)
````

```@raw html
<div><div style = "float: left;"><span>3×30 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">age</th><th style = "text-align: left;">sex</th><th style = "text-align: left;">on_thyroxine</th><th style = "text-align: left;">query_on_thyroxine</th><th style = "text-align: left;">on_antithyroid_medication</th><th style = "text-align: left;">sick</th><th style = "text-align: left;">pregnant</th><th style = "text-align: left;">thyroid_surgery</th><th style = "text-align: left;">I131_treatment</th><th style = "text-align: left;">query_hypothyroid</th><th style = "text-align: left;">query_hyperthyroid</th><th style = "text-align: left;">lithium</th><th style = "text-align: left;">goitre</th><th style = "text-align: left;">tumor</th><th style = "text-align: left;">hypopituitary</th><th style = "text-align: left;">psych</th><th style = "text-align: left;">TSH_measured</th><th style = "text-align: left;">TSH</th><th style = "text-align: left;">T3_measured</th><th style = "text-align: left;">T3</th><th style = "text-align: left;">TT4_measured</th><th style = "text-align: left;">TT4</th><th style = "text-align: left;">T4U_measured</th><th style = "text-align: left;">T4U</th><th style = "text-align: left;">FTI_measured</th><th style = "text-align: left;">FTI</th><th style = "text-align: left;">TBG_measured</th><th style = "text-align: left;">TBG</th><th style = "text-align: left;">referral_source</th><th style = "text-align: left;">diagnosis</th></tr><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;"></th><th title = "Union{Missing, Int64}" style = "text-align: left;">Int64?</th><th title = "Union{Missing, InlineStrings.String1}" style = "text-align: left;">String1?</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "Union{Missing, Float64}" style = "text-align: left;">Float64?</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "Union{Missing, Float64}" style = "text-align: left;">Float64?</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "Union{Missing, Float64}" style = "text-align: left;">Float64?</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "Union{Missing, Float64}" style = "text-align: left;">Float64?</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "Union{Missing, Float64}" style = "text-align: left;">Float64?</th><th title = "InlineStrings.String1" style = "text-align: left;">String1</th><th title = "Missing" style = "text-align: left;">Missing</th><th title = "InlineStrings.String7" style = "text-align: left;">String7</th><th title = "InlineStrings.String31" style = "text-align: left;">String31</th></tr></thead><tbody><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: right;">41</td><td style = "text-align: left;">F</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">t</td><td style = "text-align: right;">1.3</td><td style = "text-align: left;">t</td><td style = "text-align: right;">2.5</td><td style = "text-align: left;">t</td><td style = "text-align: right;">125.0</td><td style = "text-align: left;">t</td><td style = "text-align: right;">1.14</td><td style = "text-align: left;">t</td><td style = "text-align: right;">109.0</td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: right;">missing</td><td style = "text-align: left;">SVHC</td><td style = "text-align: left;">negative.|3733</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: right;">23</td><td style = "text-align: left;">F</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">t</td><td style = "text-align: right;">4.1</td><td style = "text-align: left;">t</td><td style = "text-align: right;">2.0</td><td style = "text-align: left;">t</td><td style = "text-align: right;">102.0</td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: right;">missing</td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: right;">missing</td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: right;">missing</td><td style = "text-align: left;">other</td><td style = "text-align: left;">negative.|1442</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: right;">46</td><td style = "text-align: left;">M</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">f</td><td style = "text-align: left;">t</td><td style = "text-align: right;">0.98</td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: right;">missing</td><td style = "text-align: left;">t</td><td style = "text-align: right;">109.0</td><td style = "text-align: left;">t</td><td style = "text-align: right;">0.91</td><td style = "text-align: left;">t</td><td style = "text-align: right;">120.0</td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: right;">missing</td><td style = "text-align: left;">other</td><td style = "text-align: left;">negative.|2965</td></tr></tbody></table></div>
```

We can see there is a problem with the `diagnosis` column:

````@julia
data.diagnosis[15:21] # rows 15 through 21
````

````
7-element Vector{InlineStrings.String31}:
 "negative.|2043"
 "compensated hypothyroid.|3169"
 "negative.|2755"
 "negative.|1010"
 "negative.|803"
 "negative.|2297"
 "negative.|3564"
````

Each entry has a trailing string which corresponds to a patient identifier we won't
need. We introduce a function to to trim such an entry, and at the same time collapse
all hypothyroidisem variants into a single `"positive"` class:

````@julia
function fix(class)
    trimmed = split(class, ".|") |> first
    if trimmed == "negative"
        return trimmed
     end
    return "positive"
end
fix("compensated hypothyroid.|3169") # demo
````

````
"positive"
````

Applying it to every entry in the `diagnosis` column:

````@julia
data.diagnosis = fix.(data.diagnosis);
first(data.diagnosis, 3)
````

````
3-element Vector{AbstractString}:
 "negative"
 "negative"
 "negative"
````

````@julia
using MLJ
schema(data)
````

````
┌───────────────────────────┬─────────────────────────┬─────────────────────────
│ names                     │ scitypes                │ types                  ⋯
├───────────────────────────┼─────────────────────────┼─────────────────────────
│ age                       │ Union{Missing, Count}   │ Union{Missing, Int64}  ⋯
│ sex                       │ Union{Missing, Textual} │ Union{Missing, String1 ⋯
│ on_thyroxine              │ Textual                 │ String1                ⋯
│ query_on_thyroxine        │ Textual                 │ String1                ⋯
│ on_antithyroid_medication │ Textual                 │ String1                ⋯
│ sick                      │ Textual                 │ String1                ⋯
│ pregnant                  │ Textual                 │ String1                ⋯
│ thyroid_surgery           │ Textual                 │ String1                ⋯
│ I131_treatment            │ Textual                 │ String1                ⋯
│ query_hypothyroid         │ Textual                 │ String1                ⋯
│ query_hyperthyroid        │ Textual                 │ String1                ⋯
│ lithium                   │ Textual                 │ String1                ⋯
│ goitre                    │ Textual                 │ String1                ⋯
│ tumor                     │ Textual                 │ String1                ⋯
│ hypopituitary             │ Textual                 │ String1                ⋯
│ psych                     │ Textual                 │ String1                ⋯
│ ⋮                         │ ⋮                       │ ⋮                      ⋱
└───────────────────────────┴─────────────────────────┴─────────────────────────
                                                    1 column and 14 rows omitted

````

````@julia
describe(data)
````

```@raw html
<div><div style = "float: left;"><span>30×7 DataFrame</span></div><div style = "clear: both;"></div></div><div class = "data-frame" style = "overflow-x: scroll;"><table class = "data-frame" style = "margin-bottom: 6px;"><thead><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;">Row</th><th style = "text-align: left;">variable</th><th style = "text-align: left;">mean</th><th style = "text-align: left;">min</th><th style = "text-align: left;">median</th><th style = "text-align: left;">max</th><th style = "text-align: left;">nmissing</th><th style = "text-align: left;">eltype</th></tr><tr class = "columnLabelRow"><th class = "stubheadLabel" style = "font-weight: bold; text-align: right;"></th><th title = "Symbol" style = "text-align: left;">Symbol</th><th title = "Union{Nothing, Float64}" style = "text-align: left;">Union…</th><th title = "Any" style = "text-align: left;">Any</th><th title = "Union{Nothing, Float64}" style = "text-align: left;">Union…</th><th title = "Any" style = "text-align: left;">Any</th><th title = "Int64" style = "text-align: left;">Int64</th><th title = "Type" style = "text-align: left;">Type</th></tr></thead><tbody><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">1</td><td style = "text-align: left;">age</td><td style = "text-align: left;">51.8442</td><td style = "text-align: left;">1</td><td style = "text-align: left;">54.0</td><td style = "text-align: left;">455</td><td style = "text-align: right;">1</td><td style = "text-align: left;">Union{Missing, Int64}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">2</td><td style = "text-align: left;">sex</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">F</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">M</td><td style = "text-align: right;">110</td><td style = "text-align: left;">Union{Missing, String1}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">3</td><td style = "text-align: left;">on_thyroxine</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">4</td><td style = "text-align: left;">query_on_thyroxine</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">5</td><td style = "text-align: left;">on_antithyroid_medication</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">6</td><td style = "text-align: left;">sick</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">7</td><td style = "text-align: left;">pregnant</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">8</td><td style = "text-align: left;">thyroid_surgery</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">9</td><td style = "text-align: left;">I131_treatment</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">10</td><td style = "text-align: left;">query_hypothyroid</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">11</td><td style = "text-align: left;">query_hyperthyroid</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">12</td><td style = "text-align: left;">lithium</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">13</td><td style = "text-align: left;">goitre</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">14</td><td style = "text-align: left;">tumor</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">15</td><td style = "text-align: left;">hypopituitary</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">16</td><td style = "text-align: left;">psych</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">17</td><td style = "text-align: left;">TSH_measured</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">18</td><td style = "text-align: left;">TSH</td><td style = "text-align: left;">4.67215</td><td style = "text-align: left;">0.005</td><td style = "text-align: left;">1.4</td><td style = "text-align: left;">478.0</td><td style = "text-align: right;">284</td><td style = "text-align: left;">Union{Missing, Float64}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">19</td><td style = "text-align: left;">T3_measured</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">20</td><td style = "text-align: left;">T3</td><td style = "text-align: left;">2.02497</td><td style = "text-align: left;">0.05</td><td style = "text-align: left;">2.0</td><td style = "text-align: left;">10.6</td><td style = "text-align: right;">585</td><td style = "text-align: left;">Union{Missing, Float64}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">21</td><td style = "text-align: left;">TT4_measured</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">22</td><td style = "text-align: left;">TT4</td><td style = "text-align: left;">109.072</td><td style = "text-align: left;">2.0</td><td style = "text-align: left;">104.0</td><td style = "text-align: left;">430.0</td><td style = "text-align: right;">184</td><td style = "text-align: left;">Union{Missing, Float64}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">23</td><td style = "text-align: left;">T4U_measured</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">24</td><td style = "text-align: left;">T4U</td><td style = "text-align: left;">0.997912</td><td style = "text-align: left;">0.31</td><td style = "text-align: left;">0.98</td><td style = "text-align: left;">2.12</td><td style = "text-align: right;">297</td><td style = "text-align: left;">Union{Missing, Float64}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">25</td><td style = "text-align: left;">FTI_measured</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">t</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">26</td><td style = "text-align: left;">FTI</td><td style = "text-align: left;">110.788</td><td style = "text-align: left;">2.0</td><td style = "text-align: left;">107.0</td><td style = "text-align: left;">395.0</td><td style = "text-align: right;">295</td><td style = "text-align: left;">Union{Missing, Float64}</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">27</td><td style = "text-align: left;">TBG_measured</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">f</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String1</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">28</td><td style = "text-align: left;">TBG</td><td style = "font-style: italic; text-align: left;"></td><td style = "font-style: italic; text-align: left;"></td><td style = "font-style: italic; text-align: left;"></td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: right;">2800</td><td style = "text-align: left;">Missing</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">29</td><td style = "text-align: left;">referral_source</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">STMW</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">other</td><td style = "text-align: right;">0</td><td style = "text-align: left;">String7</td></tr><tr class = "dataRow"><td class = "rowLabel" style = "font-weight: bold; text-align: right;">30</td><td style = "text-align: left;">diagnosis</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">negative</td><td style = "font-style: italic; text-align: left;"></td><td style = "text-align: left;">positive</td><td style = "text-align: right;">0</td><td style = "text-align: left;">AbstractString</td></tr></tbody></table></div>
```

Models and machines

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

¹Read "Going deeper" sections if you feel you need more detail to fill out your mental
model of what is going in the tutorial. You may want to skip them on a first reading.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

