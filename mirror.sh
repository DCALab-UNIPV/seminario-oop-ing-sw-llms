# Serve solo a me
TARGET="/home/aiman/code/seminario-oop-ing-sw-llms"
# mkdir "$TARGET/"
rm -rf "$TARGET/*"
git ls-files | xargs cp --parents -t "$TARGET"
cd "$TARGET/"
git add *
git add -u
git commit -m ...
git push