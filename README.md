# Image Reference Sorter Tool - Powered by "AI"

## Who is this for?
- Designers, Artists and anyone who works with a lot of reference images that need to be categorized.
- I am an artist myself and I download a lot of reference photos, having them in folders is a chore. Being a Deep learning engineer has it's perks and so I made this litle program to help me automatically sort my photos

## What is this?
- It can take a folder of unsorted images and plop them into sorted folders. It uses a simple neural network that I trained on the following classes
- Supports the following categories for now with a good bit of accuracy
    - 'Airplanes', 'Animals', 'Armor', 'Birds', 'Desert', 'Fields', 'Flowers', 'Food', 'Forest', 'Jewellery', 'Mech', 'Modern Architecture', 'Mountain', 'Old Architecture', 'People', 'Plants', 'Ships', 'Statues', 'Utensils', 'Vehicles', 'Water', 'Weapons', 'modern clothes', 'vintage clothes'
- If this gains traction, this will become a proper tool

## Do I need XYZ
- Only a decent computer to run these
- Install everything as below
- Some patience if these things seem complicated

## How to use it?
- Clone this repository
    - Use git clone if you know how, or,
    - Click the green "Code" button. Click "Download ZIP" and extract it
- Install python 
    - [site](https://www.python.org/downloads/)
    - Mac/Linux have it preinstalled generally
- Install [fastai](https://docs.fast.ai/) on your machine
- Open the file predictions.py
    - Change the folders as required that say the word CHANGE
    - Save the file
- Open a terminal/command prompt (google it if you do not know how)
- cd to the directory you saved the repository in
    - You can drag and drop the folder into the terminal
- Type python3 predictions.py
- Cheer

## FAQ
- How much will I need to pay?
    - Nothing really. I do not intend to make this paid. Ever. 
    - I would appreciate some shares though. 
- Why is it so complicated to run ugh?
    - This is an early beta. If this gains any traction, then I will make it more user friendly and maybe even add a little interface
- Can I have my own custom categories?
    - Yes of course, but
    - For now, you would need some programming experience + a powerful computer
        - I have both. Now what. Well open trainer.ipynb and scroll through it. You can change the folders as above and add your own custom classes.
        - Oh no?
            - Don't worry. Either ask me for more classes. Or make this blow up so I have the motivation to make an interface
- I am a programmer, can I contribute?
    - Hell yes. Please do. File an issue/fork it etc