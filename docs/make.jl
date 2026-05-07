using Documenter

const REPO_NAME = "ClassImbalanceTutorials.jl"
const  REPO = Remotes.GitHub("ablaom", REPO_NAME)

using TutorialTools

makedocs(
    modules=Module[TutorialTools,],
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
    repo = Remotes.GitHub("ablaom", REPO_NAME),
)

deploydocs(
    devbranch="master", # deployment to gh-pages only happens when this is the target
    push_preview=false,
    repo="github.com/ablaom/$REPO_NAME.git",
)
