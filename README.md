# NLP-Measure-Text-Fluency  

## Team
* <b>Team name</b>: $pecial ¢haracters
* <b>Team number</b>: 27
* <b>Members</b>: 
    * Thota Gokul Vamsi (2019111009)  
    * Sai Akarsh C (2019111017)  

## Description 

This repository consists of the entire implementation of baselines, experiments and application which were implemented as part of a project of 'Introduction to NLP' course. It is regarding <b>measuring text fluency</b> and explores various features, language model architectures and classifiers, for the multi-label classification of text into labels such as - Fluent / Not Fluent / Neutral.

## Data

* The train, test and validation datasets which were used for the project are uploaded [here](https://drive.google.com/drive/folders/1evD24N9AAh7k3GNfJBhxVNrKTBy7GToi?usp=sharing).
* Note that this folder should be downloaded as a whole, and placed in the same folder which contains `src` folder, before any execution of below steps.

## Model checkpoints

* All the model checkpoints which were saved are uploaded [here](https://drive.google.com/drive/folders/16c04yf95_Ael0iXFmQuQKdhQm97SWWHE?usp=sharing). 
    * The `app_models` folder consists of the language model (BERT), the classifier (Random Forest), tokenizer and vocabulary which are required by the application to perform real time predictions.
    * The `language_models` folder consists of the RNN and LSTM language models, which were explored as a part of the experimentation.

## Execution

### Application

* To execute the application, firstly download the models from the above drive link, and move the folder `app_models` inside the `models` folder to the `app` folder inside the `src` directory. 

* Move to the `app` folder via `cd ./src/app`, and execute the `run.sh` script which performs the installation of required libraries. It can be modified as per requirement.

* Execute the app locally by running the command: `streamlit run main.py`. Open the URL provided by streamlit (or wait till it opens automatically on your browser).

* Follow the instructions on the dashboard, and enter the text for fluency estimation in the text area, and submit it when done. The prediction of the model is displayed on submitting.

### Other implementations

* The respective jupyter notebooks of different experimental implementations can be used normally, where individual cells / whole notebook can be executed as usual.

* Note that, for executing scripts associated with RNN and LSTM language models, the corresponding checkpoints inside `language_models` folder in above drive link should be downloaded and moved to respective folders. For example, to execute the notebook in `RNN-LM` folder, firstly the `RNN_LM.hdf5` file should be downloaded and moved to the `RNN-LM` folder. 
