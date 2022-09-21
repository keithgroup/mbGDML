#!/usr/bin/env bash
# Build sphinx documentation with multiversion.
# Will not build current branch documentation; only main and tagged commits.
cd "${0%/*}"
rm -rf ./html/
sphinx-multiversion -nT ./source/ ./html/
touch ./html/.nojekyll

# Create html redirect to main
echo "<head>" > ./html/index.html
echo "  <meta http-equiv='refresh' content='0; URL=https://www.aalexmmaldonado.com/reptar/main/index.html'>" >> ./html/index.html
echo "</head>" >> ./html/index.html
