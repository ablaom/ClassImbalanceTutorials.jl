ERR_BAD_SCRIPT(file) = ErrorException(
    "$file appears to be missing, or is not a regular file. "
)

INFO_MISSING_TESTS(name) = "No runtests.jl file found. Skipping tests for $name. "

"""
    generate(tutorial_dir; tests=true)

If `tests==true`, then, in a new Julia process, run the code appearing in the file
"runtests.jl" appearing in the directory `tutorial_dir`, which should additionally contain
the julia script `tutorial.jl` for some tutorial. The code is run after activating and
instantiating the Julia package environment specified by the Manifest.toml file contained
in the same directory.

!!! note

The test file `runtests.jl` should begin with `include("tutorial.jl")`.

In any case, attempt to generate a new markdown file from "tutorial.jl" using Literate.jl
and the same project.

Returns `true` if the Julia process executes successfully, and returns `false` otherwise.

"""
function generate(tutorial_dir; tests=true)
    tutorial_name = splitpath(tutorial_dir) |> last

    script_file = joinpath(tutorial_dir, "tutorial.jl")
    isfile(script_file) || throw(ERR_BAD_SCRIPT(script_file))

    test_file = joinpath(tutorial_dir, "runtests.jl")
    test_file_exists = isfile(test_file)
    test_file_exists || @info(INFO_MISSING_TESTS(tutorial_name))

    success = true
    if tests && test_file_exists
        cmd = `julia
               --startup-file=no
               --color=yes
               --project=$tutorial_dir -e '
                   # Next if-end block makes sure the standard library is available.
                   # When calling generate(…) from /test/runtests.jl the standard library
                   # is mysteriously disappearing from LOAD_PATH here so that Pkg is not
                   # available.

                   if !("@stdlib" in LOAD_PATH)
                       push!(LOAD_PATH, "@stdlib")
                   end

                   using Pkg
                   using Test
                   Pkg.instantiate();
                   @testset "'$tutorial_name'" begin
                       include("'$test_file'")
                       @test true
                   end'`
        try
            run(cmd)
        catch excptn
            success = false
            excptn isa ProcessFailedException || rethrow(excptn)
        end
    end
    return success
end

"""
    generate(tutorial_dirs::AbstractVector; tests=true)

Attempts to generate each tutorial with directory in the provided list
`tutorial_dirs`. Returns `true` if all tutorials are successfully tested and generated.

"""
generate(tutorial_dirs::AbstractVector; kwargs...) =
    all(dir -> generate(dir; kwargs...), tutorial_dirs)
