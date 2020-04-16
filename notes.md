# environment setup
    source ~/.local/bin/virtualenvwrapper.sh
    source ~/.bashrc
    mkvirtualenv dlaz -p python3
    workon dlaz
    #export TMPDIR=/mnt/disks/data/tmp
    pip install numpy
    pip install tensorflow==2.0.0 
    pip install scikit-learn
    pip install matplotlib
    pip install spyder
    pip install Theano
    pip install keras
    pip install pandas
    pip install pillow
# Data location
     /mnt/disks/data/deep_learning_A-Z
     
# on google cloud machine
    cd  /mnt/disks/data/deep_learning_A-Z
    source /usr/local/bin/virtualenvwrapper.sh
    source ~/.bashrc