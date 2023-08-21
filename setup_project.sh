#!/bin/bash

# This setups a project using the file structure suggested by genome au: 
# https://genome.au.dk/docs/best-practices/
# Please remember to update the readme.md file after you've created the project.
# You can also setup the project for several users, just manually add the user(s) 
# directories and run this script in that directory.

# Create a function that makes a directory and a README file inside it
make_dir() {
  mkdir -vp "$1" # Make the directory with the given name and print a message
}

# Get the name of the current folder
project_name=$(basename "$PWD")


# Create an array of subdirectory names
subdirs=(data steps results plots scripts docs)

# Loop over the subdirectory names and create them with the make_dir function
for subdir in "${subdirs[@]}"; do
  make_dir "$subdir"
done

# Create the other files in the main project directory
# remember to overwrite this with you anaconda environment. Alternatively you can create
# a requirements.txt file.
touch "./environment.yml"
touch "./NOTEBOOK.md"
touch "./README.md"
# touch "./.gitignore"

# Create the files in the docs subdirectory
# These are just examples, rename or change them.
touch "./docs/1-installing-dependencies.md"
touch "./docs/2-running-some-analysis.md"
touch "./docs/3-running-some-other-analysis.md"

# Write items to .gitignore
echo "/data" >> .gitignore


# Write some content to the README files using a here document
cat << EOF > "./README.md"
# $project_name

This is a project that does something awesome.

## Project structure

This project has the following structure:

- data: This directory contains the data files used for the analysis.
- steps: This directory contains the steps of the analysis.
- results: This directory contains the results of the analysis.
- plots: This directory contains the plots generated from the analysis.
- scripts: This directory contains the scripts used for the analysis.
- docs: This directory contains the documentation of the project.
- environment.yml: This file specifies the dependencies of the project.
- NOTEBOOK.md: This file contains some notes about the project.
- README.md: This file provides an overview of the project.

## How to run

To run this project, you need to do the following:

- Install the dependencies using "conda env create -f environment.yml".
- Activate the environment using "conda activate .".
- Run the scripts using "bash scripts/*.sh".
EOF

# Print a message to confirm the creation of the folder structure
echo "Folder structure created successfully for $project_name!"
