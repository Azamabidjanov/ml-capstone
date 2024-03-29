# Machine-Learning Capstone Project: Recycle Rush

## Authors: Natalie Kalin, Armando Ocampo, Jay Harrison, Azam Abidjanov

## Summary

We designed and implemented a project that uses a CNN to classify goods as recyclable (paper, plastic, metal, glass, cardboard) or as trash. We ended up achieving an accuracy of approximately 92% using a FastAI transfer learning model, and 40% with a model we trained from scratch. Though this is a proof-of-concept in the form of a website, this would ultimately be useful on a factory line that could help determine goods that are actually recyclable vs. trash.  

## Dataset
https://github.com/garythung/trashnet

## Saad's Website Intro Code

https://github.com/minds-mines/intro-ml-code-samples/tree/complete

## Development Environment - `pipenv`
We are using pipenv to keep our development/dependencies consistent.
Pipenv uses Pipfile/Pipfile.lock to ensure we are using the same version
of packages for our projects. This avoids "I swear it just worked on my machine!" type situations.
Whenever you are introducing a new package, please add it to the pipfile (instructions below).

## Pipenv installation
+ Clone the repo: `git clone`
+ In the repo directory, set your environment and dependencies: `pipenv install --dev`
+ Activate your environment: `pipenv shell`
+ Run python scripts from the activated shell: `python pull-data.py`

## Pipenv: Adding new packages
+ To add a new package (scipy for example) into the pipfile: `pipenv install scipy`
