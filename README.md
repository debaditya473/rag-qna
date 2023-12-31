﻿Answers to example questions in [answers.txt](https://github.com/debaditya473/rag-qna/blob/main/answers.txt)

The approach has been explained in this document: [Approach and Experiments](https://docs.google.com/document/d/1oHqxCV9dCWkMQ5rzzivAMfQhqCMNYAc4qMVp4h881jI/edit)
The notebook [approach.ipynb](https://github.com/debaditya473/rag-qna/blob/main/approach.ipynb) also has the approach displayed through code

## How to run the code:


The final pipeline used to generate the published results has been written and executed in the notebook [qna_pipeline.ipynb](https://github.com/debaditya473/rag-qna/blob/main/qna_pipeline.ipynb).

To run this notebook, first go to this link: https://drive.google.com/drive/folders/1TXtjcWZmw-3LNGSBjJxBYp59VTC1-DLm?usp=sharing,
and download the text-to-text generation model that I fine-tuned on our dataset for 10 epochs using Causal Language Modeling. 

The model to download is named ‘flan-t5-large-10_epochs.h5’


Also download the corpus or knowledge base that contains our dataset, from the same link. The corpus file to be downloaded is called ‘section_wise_data.txt‘.


Next, in the first line of cell no. 4 of the notebook (cell numbering starts from 1), replace the file_name variable with the path to the downloaded corpus ‘flan-t5-large-10_epochs.h5‘.


Now, in the 16th line of cell number 5, replace the string in 

```
gen_model = pickle.load(open(' ', 'rb'))
```

with the path to the downloaded fine-tuned text generation model ‘flan-t5-large-10_epochs.h5‘.


Next, run all cells of the notebook in sequential order, and the answers to the four questions will get printed as output of the last cell.


## How to run the fine-tuning notebook:


The text-to-text generation model I have used in our final model is a fine-tuned version of the ‘google/flan-t5-large’ model. It has been fine-tuned on our dataset consisting of the PDFs from https://www.cdc.gov/vaccines/hcp/acip-recs/vacc-specific/hpv.html .


The notebook used to fine-tune the model has also been provided in this submission. The notebook is named [finetuning.ipynb](https://github.com/debaditya473/rag-qna/blob/main/finetuning.ipynb).


To run it, first go to the link: [Finetuning Data](https://drive.google.com/drive/u/2/folders/1wwM0iLutsjpnY3djCqx9NV5xv-HY2np_), and download the three dataset files ‘finetuning_train.txt’, ‘finetuning_val.txt’ and ‘finetuning_test.txt’.


Now, in cell number 4 of the notebook, replace the data_files “train”, “val” and “test” paths with the paths to the above three downloaded files.


In cell number 12, set the training arguments as per your wish. In cell number 16, replace the filename with the path to where you want to store the fine-tuned model and the name of the model.


Now run all cells in sequential order. In cell number 15, you might have to visit this website: https://wandb.ai/authorize, sign in and copy the key displayed on the website to your terminal. This is used for logging the training runs.
