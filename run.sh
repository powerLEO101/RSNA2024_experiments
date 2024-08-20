git add -A .
git commit -m "$1"
git push origin main
ssh 100.82.29.87 -t 'cd /media/workspace/RSNA2024_experiments; git pull; cat jobs.txt >> ../jobs.txt; exit'