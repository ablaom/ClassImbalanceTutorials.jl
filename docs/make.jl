using Documenter

const REPO_NAME = "ClassImbalanceTutorials.jl"
const  REPO = Remotes.GitHub("JuliaAI", REPO_NAME)

makedocs(
    modules=Module[],
    format=Documenter.HTML(
        prettyurls = true,
        collapselevel = 1,
    ),
    pages=[
        "Home" => "index.md",
        "Test for Hypothyroidism" => "tutorials/thyroid/tutorial.md",
        "Test for Diabetes" => "tutorials/diabetes/tutorial.md",
    ],
    sitename=REPO_NAME,
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("JuliaAI", REPO_NAME),
)

deploydocs(
    devbranch="dev",
    push_preview=false,
    repo="github.com/JuliaAI/$REPO_NAME.git",
)
