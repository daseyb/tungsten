
ssh dseybhos@dseyb.host.dartmouth.edu "mkdir -p ~/public_html/results/experiments/$1"
ssh dseybhos@dseyb.host.dartmouth.edu "mkdir -p ~/public_html/results/experiments/$1/images"
rsync ./data/experiments/$1.json dseybhos@dseyb.host.dartmouth.edu:~/public_html/results/experiments/$1/$1.json
rsync $DATA_DIR/sweeps/$1/*.exr dseybhos@dseyb.host.dartmouth.edu:~/public_html/results/experiments/$1/images
ssh dseybhos@dseyb.host.dartmouth.edu "chmod 644 ~/public_html/results/experiments/$1/$1.json"
ssh dseybhos@dseyb.host.dartmouth.edu "chmod 644 ~/public_html/results/experiments/$1/images/*"
