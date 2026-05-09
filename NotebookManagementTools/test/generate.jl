using Test
using NotebookManagementTools
using Suppressor

tutorial_root = joinpath(@__DIR__, "dummy_tutorials")
good_tutorial = joinpath(tutorial_root, "good_tutorial")
bad_tutorial = joinpath(tutorial_root, "bad_tutorial")
tutorial_without_script = joinpath(tutorial_root, "tutorial_without_script")
tutorial_without_tests = joinpath(tutorial_root, "tutorial_without_tests")
docs = joinpath(@__DIR__)
set_path_to_docs("docs")

###################################
# REMOVE @suppress WHEN DEBUGGING #
###################################

@testset "generate(…)" begin
    @test @suppress generate(good_tutorial) == joinpath(good_tutorial, "notebook.md")
    @test @suppress isempty(generate(bad_tutorial, docs))
    @test @suppress generate([good_tutorial, tutorial_without_tests], docs) ==
        [
            joinpath(good_tutorial, "notebook.md"),
            joinpath(tutorial_without_tests, "notebook.md"),
        ]
    @test @suppress isnothing(generate([good_tutorial, bad_tutorial]))
    @test_throws(
        NotebookManagementTools.ERR_BAD_SCRIPT(joinpath(tutorial_without_script, "notebook.jl")),
        generate(tutorial_without_script, docs),
    )
    @test_logs(
        (:info, NotebookManagementTools.INFO_MISSING_TESTS("tutorial_without_tests")),
        generate(tutorial_without_tests),
    )
    firstline = open(readline, joinpath(good_tutorial, "notebook.md"))
    @test firstline == "```@meta"
    firstline = open(readline, joinpath(tutorial_without_tests, "notebook.md"))
    @test firstline == "```@meta"
end

# remove the generated markdown files:

for dir in [good_tutorial, tutorial_without_tests]
    rm(joinpath(dir, "notebook.md"))
end

true
