"""
    NotebookManagementTools

Module providing tools for:

- Generating markdown from Literate.jl-compliant julia scripts, including testing of the
  same.

- Automating this generation and testing in continuous integration, for GitHub
  repositories providing collections of tutorials, either stand-alone or part of
  documentation of some Julia package(s) provided by the repository.

The tools are meant to be useful in both local development and final deployment. A
repository of this kind will be structured using Julia's "workspace" package management
structure. For a stand-alone collection of tutorials, this might look something like this:

```
├── Project.toml     # <---- root project
├── docs
│   ├── make.jl
│   ├── Project.toml # <--- project for generating documentation (including "Literate")
│   └── src
│       └── notebooks
│           ├── tutorial1
│           │   ├── Project.toml  # <--- tutorial-specific project
│           │   ├── runtests.jl
│           │   └── notebook.jl
│           └── tutorial2
│               ├── Project.toml
│               ├── runtests.jl
│               ├── notebook.jl
│               └── notebook.md
```

Here each "Project.toml" file is part of a single (nested) workspace.

# Tools

- [`generate`](@ref): The main tool, used to generate markdown from annotated julia script.

- [`set_path_to_docs`](@ref): For pointing the generator to an appropriate version of
  Literate.jl, the engine for the markdown generation.

- [`path_to_docs`](@ref): For inspecting the above.

- [`dirs`](@ref): Used in continuous integration to list tutorial directories.

- [`dirs_containing`](@ref): Used in continuous integration to find tutorial directories
  containing files known to have changed in a pull-request.

"""
module NotebookManagementTools

include("utilities.jl")
include("set_path_to_docs.jl")
include("generate.jl")
include("init.jl")

export generate, dirs_containing, dirs, set_path_to_docs, path_to_docs

end # module
