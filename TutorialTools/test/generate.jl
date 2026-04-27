using Test
using TutorialTools
using Suppressor

tutorial_root = joinpath(@__DIR__, "dummy_tutorials")
good_tutorial = joinpath(tutorial_root, "good_tutorial")
bad_tutorial = joinpath(tutorial_root, "bad_tutorial")
tutorial_without_script = joinpath(tutorial_root, "tutorial_without_script")
tutorial_without_tests = joinpath(tutorial_root, "tutorial_without_tests")

@testset "generate(…)" begin
    @test @suppress generate(good_tutorial)
    @test @suppress !generate(bad_tutorial)
    @test @suppress !generate([good_tutorial, bad_tutorial])
    @test_throws(
        TutorialTools.ERR_BAD_SCRIPT(joinpath(tutorial_without_script, "tutorial.jl")),
        generate(tutorial_without_script),
    )
    @test_logs(
        (:info, TutorialTools.INFO_MISSING_TESTS("tutorial_without_tests")),
        generate(tutorial_without_tests),
    )
end

true
