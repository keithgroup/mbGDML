#!/usr/bin/env bash
# Push HTML files to gh-pages automatically.

# Fill this out with the correct org/repo
ORG=keithgroup
REPO=mbGDML
# This probably should match an email for one of your users.
EMAIL=aalexmmaldonado@gmail.com

set -e

# Clone the gh-pages branch outside of the repo and cd into it.
cd ..
git clone -b gh-pages "https://aalexmmaldonado:$GITHUB_TOKEN@github.com/$ORG/$REPO.git" gh-pages
cd gh-pages
git remote set-url origin "https://aalexmmaldonado:$GITHUB_TOKEN@github.com/$ORG/$REPO.git"

# Update git configuration so I can push.
if [ "$1" != "dry" ]; then
    # Update git config.
    git config user.name "Travis Builder"
    git config user.email "$EMAIL"
fi

# Copy in the HTML.  You may want to change this with your documentation path.
cp -a ../$REPO/docs/html/. .

# Add and commit changes.
git add -A .
git commit -m "[ci skip] Autodoc commit for $COMMIT."
if [ "$1" != "dry" ]; then
    # -q is very important, otherwise you leak your GH_TOKEN
    git push -q origin gh-pages
fi