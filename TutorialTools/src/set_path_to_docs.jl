# # LOGGING

const ERR_PATH_TO_DOCS = ErrorException(
    "No path to the documents folder has been set. This  must be "*
        "a folder, typically \"/docs\",  containing a Project.toml file with a "*
        "[deps] entry for \"Literate\". Run "*
        "`set_path_to_docs(path)` go set it. "
)


# # METHODS

"""
    path_to_docs()

Return the path to a Project.toml file with a [deps] entry for "Literate". Use
[`set_path_to_docs`](@ref) to set or change.

"""
function path_to_docs()
    out = PATH_TO_DOCS[]
    isempty(out) && throw(ERR_PATH_TO_DOCS)
    return out
end

"""
    set_path_to_docs(path)

Point `TutorialsTools` to the location of a directory containing a Project.toml file with
a [deps] entry for "Literate". Typically, this is the absolute path to a repository "docs"
directory.

```julia-repl
julia> pwd()
"/Users/anthony/GoogleDrive/Julia/ClassImbalanceTutorials.jl"

julia> setpath("~/GoogleDrive/Julia/ClassImbalanceTutorials.jl/docs/")
```

Use [`path_to_docs()`](@ref) to inspect the path once set.

"""
set_path_to_docs(path) = (PATH_TO_DOCS[] = expanduser(path))
