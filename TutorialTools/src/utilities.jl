"""
    dirs_containing([paths]; root=pwd())

Return a list of those subdirectores of `root` that contain one or more of the files
specified by `paths`.

"""
function dirs_containing(paths=nothing; root=TUTORIAL_ROOT)
    all_tutorial_dirs = filter(isdir, readdir(root, join=true))
    isnothing(paths) && return all_tutorial_dirs
    filter(all_tutorial_dirs) do dir
        any(path -> startswith(path, dir), paths)
    end |> unique
end

"""
    dirs(; root=pwd())

Return *all* top-level subdirectories of `root`.

"""
dirs(; kwargs...) = dirs_containing(; kwargs...)
