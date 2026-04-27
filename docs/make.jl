using Documenter

const  REPO = Remotes.GitHub("JuliaAI", "ThyroidTutorial.jl")

makedocs(
    modules=[],
    format=Documenter.HTML(
        prettyurls = true,
        collapselevel = 1,
    ),
    pages=[
        "Home" => "index.md",
        "Developing a test for Hypothyroidism" => "tutorials/thyroid/tutorial.md",
    ],
    sitename="ThyroidTutorial.jl",
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("JuliaAI", "ThyroidTutorial.jl"),
)

deploydocs(
    devbranch="dev",
    push_preview=false,
    repo="github.com/JuliaAI/ThyroidTutorial.jl.git",
)
