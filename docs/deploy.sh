echo "Deploying documentation to github pages..."

# These make sure the script will stop if something fails:
set -e
set -x

make html
ghp-import -n -p docs/_build/html/

echo "Successfully deployed!"
