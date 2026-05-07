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

Return the path to a Project.toml file with a [deps] entry for \"Literate\". Use
[`setpath`](@ref) to set or change.

"""
function path_to_docs()
    out = PATH_TO_DOCS[]
    isempty(out) && throw(ERR_PATH_TO_DOCS)
    return out
end

"""
    set_path_to_docs(path)

Point `TutorialsTools` to the location of the doc to be modified. Ordinarily,
this is the absolute path to the subdirectory `/src/doc` of a local clone of
MLJModels.jl.

```julia-repl
julia> pwd()
"/Users/anthony/GoogleDrive/Julia/MLJ/MLJModels.jl"

julia> setpath("~/GoogleDrive/Julia/MLJ/MLJModels.jl/src/doc")
```

"""
set_path_to_docs(path) = (PATH_TO_DOCS[] = expanduser(path))
