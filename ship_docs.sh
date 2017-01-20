#!/bin/bash

# run 'make html' in the docs folder
printf "\nBuilding docs\n\n"
make -Cdocs html

printf "\nRemoving old new-docs local branch\n\n"
git branch -D new-docs

# -f b/c built docs are otherwise ignored
printf "\nCreating new-docs local branch\n\n"
git checkout -b new-docs
printf "\nAdding built html to new-docs branch\n\n"
git add -f docs/build/html
printf "\nCommitting built docs\n\n"
git commit -m"built updated docs"

printf "\nCreating new branch from docs/build/html and pushing to docs remote\n\n"
git filter-branch -f --prune-empty --subdirectory-filter docs/build/html new-docs
git push -u docs HEAD:new-docs

cd ../docs/

git checkout master

git branch -D new-docs

git fetch origin
git checkout new-docs


printf "\nRemoving old gh-pages from docs repo\n\n"
git push origin :gh-pages

printf "\nPushing updated gh-pages branch to docs repo\n\n"
git push -u origin HEAD:gh-pages

printf "\nRemoving old new-docs from docs repo\n\n"
git push origin :new-docs

cd ../PRMS-Python

git checkout -f master
