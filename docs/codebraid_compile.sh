## Compiles the JMP into a pdf using `codebraid`

PARAMS=(
pandoc 
-f markdown
-t beamer
--overwrite
-o estimator_presentation.pdf
estimator_presentation.cbmd
)

#--filter pandoc-mermaid
#--filter pandoc-xnos
#--include-in-header=template.tex 

codebraid ${PARAMS[@]}