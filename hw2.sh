# TODO: create shell script for running the testing code of the baseline model
wget 'https://www.dropbox.com/s/nioi25wsrfjncb6/model_best.pth.tar?dl=1'
mv model_best.pth.tar?dl=1 model_best.pth.tar
python3 test.py --dir_img $1 --save_img $2