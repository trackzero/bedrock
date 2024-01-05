# README.md

Starting some bedrock experiments...

## Environment Setup

This particular version is working with python 3.12. For a first time install, go to the directory, run `python -m venv venv` to create a virtual environment in your current location. Then activate it with `venv\Scripts\activate` (on Windows. `source venv/bin/activate` it on Linux/mac). Then `pip install -r requirements.txt`. Remember to deactivate it when you're done. Or don't, it's your computer.


 `pip install -r requirements.txt` to load the dependencies.

## image_generator.py

Generate an image with both Titan and Stable Diffusion. Results saved in /output

## bedrock_text_gen_runoff.py

Feed the same prompt to three different models to compare output.