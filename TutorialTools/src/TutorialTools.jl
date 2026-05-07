"""
    TutorialTools

Module providing tools for:

- Generating markdown from Literate.jl-compliant julia scripts, including testing of the
  same.

- Automating this generation and testing in continuous integration, for GitHub
  repositories providing collections of tutorials, either standalone or part of
  documentatio of some Julia package(s) provided by the repository.

The tools are meant to be useful in both local development and final deployment. A
repository of this kind will be structured using Julia's "workspace" package management
structure. For a standalone collection of tutorials, this might look something like this:

```
├── Project.toml     # <---- root project
├── docs
│   ├── make.jl
│   ├── Project.toml # <--- project for generating documentation (including "Literate")
│   └── src
│       └── tutorials
│           ├── tutorial1
│           │   ├── Project.toml  # <--- tutorial-specific project
│           │   ├── runtests.jl
│           │   └── tutorial.jl
│           └── tutorial2
│               ├── Project.toml
│               ├── runtests.jl
│               ├── tutorial.jl
│               └── tutorial.md
```

Here each "Project.toml" file is part of a single (nested) workspace.

# Tools

- [`set_doc_path`](@ref)
- [`doc_path`](@ref)
- [`generate`](@ref)
- [`dirs`](@ref)
- [`dirs_containing`](@ref)

""" module TutorialTools

include("utilities.jl")
include("set_doc_path.jl")
include("generate.jl")
include("init.jl")

export generate, dirs_containing, dirs, set_doc_path, doc_path

end # module
