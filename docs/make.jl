using Documenter

const REPO_NAME = "ClassImbalanceTutorials.jl"
const  REPO = Remotes.GitHub("ablaom", REPO_NAME)

makedocs(
    modules=Module[],
    format=Documenter.HTML(
        prettyurls = true,
        collapselevel = 1,
        inventory_version = "1",
    ),
    pages=[
        "Home" => "index.md",
        "Test for Hypothyroidism" => "tutorials/thyroid/tutorial.md",
        "Test for Diabetes" => "tutorials/diabetes/tutorial.md",
    ],
    sitename=REPO_NAME,
    warnonly = [:cross_references, :missing_docs],
    repo = Remotes.GitHub("ablaom", REPO_NAME),
)

deploydocs(
    devbranch="dev",
    push_preview=false,
    repo="github.com/ablaom/$REPO_NAME.git",
)
