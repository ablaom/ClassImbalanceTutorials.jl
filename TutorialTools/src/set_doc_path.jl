# # LOGGING

const ERR_DOC_PATH = ErrorException(
    "No path to the documents folder has been set. This  must be "*
        "a folder, typically \"/docs\",  containing a Project.toml file with a "*
        "[deps] entry for \"Literate\". Run "*
        "`set_doc_path(path)` go set it. "
)


# # METHODS

"""
    doc_path()

Return the path to a Project.toml file with a [deps] entry for \"Literate\". Use
[`setpath`](@ref) to set or change.

"""
function doc_path()
    out = DOC_PATH[]
    isempty(out) && throw(ERR_DOC_PATH)
    return out
end

"""
    set_doc_path(path)

Point `MLJModelRegistryTools` to the location of the doc to be modified. Ordinarily,
this is the absolute path to the subdirectory `/src/doc` of a local clone of
MLJModels.jl.

```julia-repl
julia> pwd()
"/Users/anthony/GoogleDrive/Julia/MLJ/MLJModels.jl"

julia> setpath("~/GoogleDrive/Julia/MLJ/MLJModels.jl/src/doc")
```

"""
set_doc_path(path) = (DOC_PATH[] = expanduser(path))
