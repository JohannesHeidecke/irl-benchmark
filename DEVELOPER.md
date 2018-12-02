# Developer documentation


### Welcome!
If you are reading this document then you are interested in contributing to the IRL Benchmark project -- very happy to have you here! All contributions are welcome: ideas, documentation, code, patches, bug reports, etc.

### Code review and git 

We use pull requests and code reviews.
Recommended workflow:

1. Make sure your local master branch is up to date with remote master. (`git pull --rebase`)
2. Work directly on master
3. When you are done, commit your changes
4. Push your changes to remote branchname (`git push origin HEAD:branchname`)
5. Open a pull request on github
6. Make changes to your files based on reviews. Push these changes to the same branch, (`git push origin HEAD:branchname`)
7. When everything is working, "Squash and merge" your pull request on Github and celebrate the merge!
8. Click "Delete branch" on github to keep it clean (can be restored later)

### Deploying changes in documentation

Documentation is compiled from `.rst` files in `docs` and the python inline docstrings, and
then served with github pages.

To deploy any changes just run the deploy script in the `docs`:

`bash deploy.sh`

However, please wait for someone to approve your pull request with the changes
before you deploy the changes in documentation.
