- [Pinspire](#pinspire)
  * [1. requirements (python=3.6)](#1-requirements--python-36-)
  * [2. file organization](#2-file-organization)
  * [3. running in local](#3-running-in-local)
  * [4. model introduction](#4-model-introduction)


# [Pinspire](http://www.ppinspire.com/)

a personalized search engine that makes recommendations on image search based on collections of images

## 1. requirements (python=3.6)
clone the repository and conda virtual environment is recommended. check the *requirements.txt* for dependencies

```
conda create --name pinterest3 python=3.6
conda activation pinterest3
pip install -r requirements.txt

# download the chromedriver for selenium (chromedriver >= 2.46)
git clone https://github.com/Yingru/pinspire.git
cd src/
wget https://chromedriver.storage.googleapis.com/2.46/chromedriver_linux64.zip
unzip chromedriver_linux64.zip

# additional libary is required if you would like to depolit the webapp
conda install flask gunicorn Pillow inflection
```
**Note**: if you would like to use the package in a server, you need to use the selenium with headless chrome.
- [inorder to use selenium with headless chrome (tested on Ubuntu-14)](https://stackoverflow.com/questions/49323099/webdriverexception-message-service-chromedriver-unexpectedly-exited-status-co)

```
sudo apt-get update -f
sudo apt-get install -y chromium-browser  # make sure to use chrome version > 72.0
sudo apt-get install libgconf-2-4
./chromedriver --version
```


## 2. file organization
```
    pinspire/
    |
    |--- run.py      # main file to launch the app
    |--- board/      # destination to store user's board information (images, descriptions)
    |       |--- $boardFolder
    |--- pureMix/    # webapp interface
    |       |--- __init__.py
    |       |--- view.py
    |       |--- static/
    |       |--- templates/
    |--- src/        # source files for scrapping, analyzing the images of the board 
    |       |--- chromedriver  
    |       |--- __init__.py
    |       |--- grepInfo.py    
    |       |--- classifier.py
    |       |--- colors.py
    |--- weights/    # weights files for store the weights of the CNN classifiers: content, pattern, artStyle
```

## 3. running in local 
### 3.1 one key command for launching the webapp and testing the interactive UI
- `./run.py`
- open your browser and go to your server. If you launch it locally, go to `http://0.0.0.0:5000`

### 3.2 use pre-trained model for scraping/classifying the images
```
cd src/
# grepping images from Pinterest and store them locally (input: pinterest username, pinterest boardname)
python grepInfo.py --user $user --board $board

# classifying the images with different classifiers
python classifier.py --classifier (content[default], pattern, artStyle)
```

## 4. model introduction
Please check the [Wiki](https://github.com/Yingru/pinspire/wiki) for more detailed explanation of the model
