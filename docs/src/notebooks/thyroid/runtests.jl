include("notebook.jl")

# these results are hard-wired into the section "Preview of the performance of the best
# model":

# @test round(test_recall, sigdigits=2) == 0.98
# @test round(test_precision, sigdigits=2) == 0.92

# And at the end of the section on RFE,

two = (DataFrames.DataFrame(describe.(evaluations))).PrecisionAtFixedRecall
@test first(two) < last(two)
