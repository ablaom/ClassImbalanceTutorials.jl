include("notebook.jl")

"""
    intervals_overlap(a::Measurement, b::Measurement; n_sigma::Real=1.0) -> Bool

Determines whether the confidence intervals of two `Measurement` instances overlap.
The confidence interval width is controlled by `n_sigma` (default is 1.0 for 1-sigma).
"""
function intervals_overlap(a::Measurement, b::Measurement; n_sigma::Real=1.0)
    # Extract nominal values
    val_a = Measurements.value(a)
    val_b = Measurements.value(b)

    # Calculate the half-width (radius) of each confidence interval
    radius_a = Measurements.uncertainty(a) * n_sigma
    radius_b = Measurements.uncertainty(b) * n_sigma

    # The intervals overlap if the distance between centers
    # is less than or equal to the sum of the radii
    return abs(val_a - val_b) <= (radius_a + radius_b)
end

# these results are hard-wired into the section "Preview of the performance of the best
# model", but calculate in the last section:
svm_row = filter(performance_summary) do row
    row.tag == "SVM"
end
svm_training_precision_at_95 = svm_row.PrecisionAtFixedRecall[1]
@test round(Measurements.value(svm_training_precision_at_95), sigdigits=3) == 0.927
@test round(Measurements.uncertainty(svm_training_precision_at_95), sigdigits=2) == 0.024
# somehow couldn't nail down the RNG; there's probably a default_rng() in play here:
@test round(test_recall_svm, sigdigits=2) in [0.97, 0.98, 0.99]
@test round(test_precision_svm, sigdigits=2) in [0.86, 0.87, 0.88]

# In the section commenting on results (in Section 11), some rankings of the models are
# discussed:
@testset "rankings of precision at 0.95 recall" begin
    sort!(performance_summary, :PrecisionAtFixedRecall, rev=true)
    @test performance_summary.tag ==  [
        "SVM",
        "gradient booster",
        "decision tree",
        "decision tree, tuned",
        "KNN",
        "linear (logistic)",
        "SVM, balanced",
        "KNN, balanced",
        "neural network",
        "dummy",
    ]
end
@testset "top three models for average precision" begin
    sort!(performance_summary, :AveragePrecision, rev=true)
    @test Set(performance_summary.tag[1:3]) ==
        Set(["dummy", "decision tree", "decision tree, tuned"])
end
@testset "oversampling reduced variance of svm average precision"  begin
    i = findfirst(==("SVM"), performance_summary.tag)
    j = findfirst(==("SVM, balanced"), performance_summary.tag)
    @test intervals_overlap(
        performance_summary.BalancedAccuracy[i],
        performance_summary.BalancedAccuracy[j],
    ) # svm and balanced svm overlap
end

@testset "top two models for balanced accuracy" begin
    sort!(performance_summary, :BalancedAccuracy, rev=true)
    @test  Set(performance_summary.tag[1:2]) ==
        Set(["decision tree, tuned", "decision tree"])
end
@testset "oversampling boosts balanced accuracy" begin
    i = findfirst(==("SVM"), performance_summary.tag)
    j = findfirst(==("SVM, balanced"), performance_summary.tag)
    k = findfirst(==("KNN"), performance_summary.tag)
    m = findfirst(==("KNN, balanced"), performance_summary.tag)
    @test i > j
    @test k > m
    @test intervals_overlap(
        performance_summary.BalancedAccuracy[i],
        performance_summary.BalancedAccuracy[j],
    ) # svm and balanced svm overlap
    @test !intervals_overlap(
        performance_summary.BalancedAccuracy[k],
        performance_summary.BalancedAccuracy[m],
    ) # knn and balanced knn don't overlap
end

# And at the end of the section on RFE,
two = (DataFrames.DataFrame(describe.(evaluations))).PrecisionAtFixedRecall
@test first(two) < last(two)

# probability thresholds for the tree are reported in the section on thresholding to be
# these:
@test thresholds == [1.0, 0.0]

# in last section, we implicitly use:
@test test_precision_tree > test_precision_svm

true
