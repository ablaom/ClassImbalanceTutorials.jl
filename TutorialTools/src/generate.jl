ERR_BAD_SCRIPT(file) = ErrorException(
    "$file appears to be missing, or is not a regular file. "
)

INFO_MISSING_TESTS(name) = "No runtests.jl file found. Skipping tests for $name. "

"""
    generate(tutorial_dir, docs_project=tutorial_dir; tests=true)

Attempt to generate a new markdown file from the file named "tutorial.jl" and contained in
`tutorial_dir`, using Literate.jl and the project at `tutorial_dir`, using a version of
Literate.jl consistent with the project specified at `docs_project` (which is pushed to
`LOAD_PATH` to make it available). Generally, `docs_project` and `tutorial_dir` should be
part of the same Julia package workspace. For example, your directory structure might look
like this:

```
├── Project.toml
├── docs
│   ├── make.jl
│   ├── Project.toml
│   └── src
│       └── tutorials
│           ├── tutorial1
│           │   ├── Project.toml
│           │   ├── runtests.jl
│           │   └── tutorial.jl
│           └── tutorial2
│               ├── Project.toml
│               ├── runtests.jl
│               ├── tutorial.jl
│               └── tutorial.md
```

where every "Project.toml" file is part of a single (nested) workspace. In that case you
might have `tutorial_dir="docs/src/tutorials/tutorial1"` and `docs_project="docs"`.

In development, you can omit `doc_project`, provided you add `Literate` to the project at
`tutorial_dir`, but it is not recommended that you permanently commit such additions.

# Testing

If `tests==true`, then, *before* attempting markdown generation, run, in a new Julia
process, the code appearing in the file "runtests.jl" appearing in the directory
`tutorial_dir`, which should additionally contain the julia script `tutorial.jl` for some
tutorial. The code is run after activating and instantiating the Julia project in
`tutorial_dir` (which happens anyway, if `tests=false`).

!!! note

    The test file `runtests.jl` should begin with "using Test; include("tutorial.jl")" and
    end with "true".



Returns the path to the generated markdown, unless the Julia process exits abnormally, in
which case an empty string is returned.

"""
function generate(tutorial_dir, docs_project=tutorial_dir; tests=true)
    tutorial_name = splitpath(tutorial_dir) |> last

    script_file = joinpath(tutorial_dir, "tutorial.jl")
    markdown_file = joinpath(tutorial_dir, "tutorial.md")
    isfile(script_file) || throw(ERR_BAD_SCRIPT(script_file))

    test_file = joinpath(tutorial_dir, "runtests.jl")
    test_file_exists = isfile(test_file)
    test_file_exists || @info(INFO_MISSING_TESTS(tutorial_name))

    success = true

    testing = tests && test_file_exists

    # Since we get Literate to do the code execution, it's output needs vanilla code
    # fencing:
    literate_config = "codefence" => Pair("````@julia", "````" )

    # Note the use of '$ ⋯ ' to interpolate into a Julia command. Naive use of
    # $ doesn't work.
    cmd = `julia
               --startup-file=no
               --color=yes
               --project=$tutorial_dir -e '
                   # Next if-end block makes sure the standard library is available.
                   # When calling generate(…) from /test/runtests.jl in TutorialTools pkg,
                   # the standard library
                   # is mysteriously disappearing from LOAD_PATH here so that Pkg is not
                   # available.

                   if !("@stdlib" in LOAD_PATH)
                       push!(LOAD_PATH, "@stdlib")
                   end

                   # Add packages from the docs project to the load path; needed to
                   # use Literate:
                   push!(LOAD_PATH, "'$docs_project'")

                   using Pkg
                   Pkg.instantiate();
                   using Test
                   using Literate

                   if '$testing'
                       @testset "'$tutorial_name'" begin
                           include("'$test_file'")
                           @test true
                       end
                   end

                   @info "Generating markdown for '$tutorial_name'... "

                   Literate.markdown(
                       "'$script_file'",
                       "'$tutorial_dir'",
                       execute=true,
                       config=Dict("codefence" => Pair("\`\`\`\`@julia", "\`\`\`\`" )),
                   )'`

    try
        run(cmd)
    catch excptn
        success = false
        excptn isa ProcessFailedException || rethrow(excptn)
    end

    return success ? markdown_file : ""
end

"""
    generate(tutorial_dirs::AbstractVector, docs_project; tests=true)

Attempts to generate each tutorial with directory in the provided list
`tutorial_dirs`. Returns re-generated markdown file paths, or `false` if there is a failed
tutorial test or markdown generation failure.

"""
function generate(tutorial_dirs::AbstractVector, docs_project; kwargs...)
    generated_files =  map(tutorial_dirs) do dir
        generate(dir, docs_project; kwargs...)
    end
    "" in generated_files && return false
    filter(generated_files) do file
        !isempty(file)
    end
end
