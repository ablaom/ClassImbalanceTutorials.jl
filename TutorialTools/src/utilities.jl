"""
    dirs_containing([paths]; root=pwd())

Return a list of those subdirectores of `root` that contain one or more of the files
specified by `paths`. If `paths` is unspecified, return *all* subdirectories of `root`.

"""
function dirs_containing(paths=nothing; root=TUTORIAL_ROOT)
    all_tutorial_dirs = filter(isdir, readdir(root, join=true))
    isnothing(paths) && return all_tutorial_dirs
    filter(all_tutorial_dirs) do dir
        any(path -> startswith(path, dir), paths)
    end |> unique
end
