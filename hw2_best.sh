# TODO: create shell script for running the testing code of your improved model
wget 'https://www.dropbox.com/s/k0pdds3ji9ce9xm/model_best_impr.pth.tar?dl=1'
mv model_best_impr.pth.tar?dl=1 model_best_impr.pth.tar

python3 test2.py --dir_img $1 --save_img $2
