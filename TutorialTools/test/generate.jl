using Test
using TutorialTools
using Suppressor

tutorial_root = joinpath(@__DIR__, "dummy_tutorials")
good_tutorial = joinpath(tutorial_root, "good_tutorial")
bad_tutorial = joinpath(tutorial_root, "bad_tutorial")
tutorial_without_script = joinpath(tutorial_root, "tutorial_without_script")
tutorial_without_tests = joinpath(tutorial_root, "tutorial_without_tests")
docs = joinpath(@__DIR__)

###################################
# REMOVE @suppress WHEN DEBUGGING #
###################################

@testset "generate(…)" begin
    @test @suppress generate(good_tutorial, docs) == joinpath(good_tutorial, "tutorial.md")
    @test @suppress isempty(generate(bad_tutorial, docs))
    @test @suppress generate([good_tutorial, tutorial_without_tests], docs) ==
        [
            joinpath(good_tutorial, "tutorial.md"),
            joinpath(tutorial_without_tests, "tutorial.md"),
        ]
    @test @suppress !generate([good_tutorial, bad_tutorial], docs)
    @test_throws(
        TutorialTools.ERR_BAD_SCRIPT(joinpath(tutorial_without_script, "tutorial.jl")),
        generate(tutorial_without_script, docs),
    )
    @test_logs(
        (:info, TutorialTools.INFO_MISSING_TESTS("tutorial_without_tests")),
        generate(tutorial_without_tests, docs),
    )
    firstline = open(readline, joinpath(good_tutorial, "tutorial.md"))
    @test firstline == "```@meta"
    firstline = open(readline, joinpath(tutorial_without_tests, "tutorial.md"))
    @test firstline == "```@meta"
end

# remove the generated markdown files:

for dir in [good_tutorial, tutorial_without_tests]
    rm(joinpath(dir, "tutorial.md"))
end

true
