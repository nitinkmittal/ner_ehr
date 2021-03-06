# Name Entity Recognition for Electronic Health Record

## Summary
Named Entity Recognition is a sub-task of information extraction that locates and classifies named entities present in unstructured text into predefined categories. Electronic Health Records EHRs contain the medical history of patients in the form of unstructured text. Additional processing is required to extract information like diseases, symptoms, drugs, etc which are entities of interest within EHRs. Structuring information from free-text EHRs can help pharmaceutical companies to minimize the time to develop new drugs and also to pre-identify the conditions and pre-conditions under which a drug might cause adverse reactions.


## About Data
The [dataset](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/) consists of 505 discharge summaries (EHRs) drawn from the MIMIC-III (Medical Information Mart for Intensive Care-III) clinical care database. *To obtain the dataset, user needs to create an account with DBMI Data Portal and submit a request for MIMIC-III*. Each EHR in the dataset was annotated by 2 independent annotators while a third annotator resolved conflicts. The dataset was already split into train and test sets. A total of 303 EHRs are present in the train set with 202 EHRs in the test set. Each EHR is associated with an annotated file containing entities associated with selected words or phrases within the EHRs. Available entities are: Drug, Strength, Form, Dosage,Frequency, Route, Duration, Reason and ADE (Adverse Drug Event). Validation data is generated from train set.

## Initial Results
![BiLSTM + CRF Confusion matrix](https://github.com/nitinkmittal/ner_ehr/blob/master/logs/ner_ehr_lstm_crf/version_7/plots/Confusion%20Matrix%20from%20best%20BiLSTM%20%2B%20CRF%20model%20on%20Test.jpeg)

## Instructions to setup environment
1. Install [miniconda](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh)
1. After installing, use command following commands
```bash
conda init
git clone https://github.com/nitinkmittal/ner_ehr.git
cd ner_ehr
conda env create -f environment.yml
conda activate ner_ehr
python src/setup.py develop
````

## To run scripts to generate tokens from raw Electronic Health Records(EHR) and associated annotations and train models refer [scripts/README.md](https://github.com/nitinkmittal/ner_ehr/tree/master/scripts#readme)
