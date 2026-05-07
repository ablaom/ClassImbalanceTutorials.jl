using Documenter

const REPO_NAME = "ClassImbalanceTutorials.jl"
const  REPO = Remotes.GitHub("ablaom", REPO_NAME)

using TutorialTools

makedocs(
    modules=Module[TutorialTools,],
    format = Documenter.HTML(
        collapselevel = 1,
        assets = [
            "assets/favicon.ico",
            asset(
                "https://fonts.googleapis.com/css2?family="*
                    "Lato:ital,wght@0,100;0,300;0,400;0,700;0,900;"*
                    "1,100;1,300;1,400;1,700;1,900&"*
                    "family=Montserrat:ital,wght@0,100..900;1,100..900&display=swap",
                class = :css,
            ),
            asset(
                "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/"*
                    "5.15.4/css/all.min.css",
                class = :css,
            ),
        ],
        repolink = "https://github.com/ablaom/ClassImbalanceTutorials.jl",
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
