# See: https://gist.github.com/jaekookang/5968ac6394ff9548456a4be92f7079fb

git add .
git status # to see what changes are going to be commited
git commit -m 'update before syncing gh-pages'
git push

git checkout gh-pages # go to the gh-pages branch
git rebase master # bring gh-pages up to date with master
git push origin gh-pages # commit the changes
git checkout master # return to the master branch