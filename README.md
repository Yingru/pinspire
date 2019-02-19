- [Pinspire](#pinspire)
  * [1. requirements (python=3.6)](#1-requirements--python-36-)
  * [2. file organization:](#2-file-organization-)
  * [3. For running in local](#3-for-running-in-local)
  * [4. Model introduction](#4-model-introduction)


# Pinspire

a personalized search engine that makes recommendations on image search based on collections of images

## 1. requirements (python=3.6) 

clone the repository locally and conda virtual environment is recommended
```
conda install pandas numpy scipy matplotlib keras tensorflow seaborn  beautifulsoup4 ipython lxml requests selenium  nltk

pip install Unidecode wget 

conda install flask gunicorn Pillow inflection

# inorder to use selenium 
https://stackoverflow.com/questions/49323099/webdriverexception-message-service-chromedriver-unexpectedly-exited-status-co
sudo apt-get install -y chromium-browser 

./chromedriver --version 
sudo apt-get install libgconf-2-4

```

## 2. file organization:
```
    -- run.py  # main file to launch the app
    -- board/  # destination to store user's board information (images, titles)
        -- $boardFolder
    -- pureMix/ # webapp interface
        -- __init__.py
        -- view.py 
        -- static/
        -- templates/
    -- src/  # source file for scrapping, machine learning the images
        -- __init__.py
        -- grepInfo.py
        -- classifier.py
        -- colors.py
    -- weights/ # weights files for three classifiers: content, pattern, artStyle
```

## 3. For running in local
```
cd src/
python grepInfo.py --user $user --board $board
python classifier.py --classifier (content[default], pattern, artStyle)
```

## 4. Model introduction
