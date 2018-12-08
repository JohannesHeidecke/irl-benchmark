echo "Deploying documentation to github pages..."

# These make sure the script will stop if something fails:
set -e
set -x

# Build Sphinx to html
make html

# Move the html to gh-pages branch
ghp-import -n -p docs/_build/html/

echo "Successfully deployed!"
