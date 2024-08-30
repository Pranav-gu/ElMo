# INLP Assignment 3
## Name - Pranav Gupta
## Roll No. - 2021101095

### Execution of files:
For constructing ELMO Forward and Backward Word Vectors and Pre-Trained BILSTM Model, run the following command - 
python3 ./ELMO.py

This will execute the Python Script and a Pre-Trained Word Embeddings Model will be saved in the current Working Directory.


Link for Initial Pre-Trained Glove Embeddings: [Glove Embeddings](https://iiitaphyd-my.sharepoint.com/:t:/g/personal/pranav_g_students_iiit_ac_in/EQ3yISHCV9tMtOyeGXTyXV0B-ZcSBTBnESshDra0_haCog?e=fdo1uf)

Link for Pre-trained BiLSTM ELMO Model: [Pre-Trained BiLSTM Model](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/pranav_g_students_iiit_ac_in/EUAyNgV2YstKoU42fBgjXiUBEbrfGsCdPMl34aPv80sFMw?e=lMcZSQ)


For performing Downstream Classification Task using any RNN(LSTM), run the following command - 
python3 ./classification.py

This will execute the Python Script and a Pre-Trained Word Embeddings Model will be saved in the current Working Directory.

For loading the Pre-Trained Model for Classification, use - torch.load('classifier.pt').

For loading the BiLSTM Model, use elmo.load_state_dict(torch.load('pretrained_elmo.zip', map_location=device)) where elmo is assumed to be an instance of the ELMO Class.




Link for Pre-Trained Model for Classification using ELMO Embeddings - [Pre-Trained Classification Model](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/pranav_g_students_iiit_ac_in/EXFyLocprCNLtJ9n_C05tvYBjPuZOQQ7EmQYe6H8LyNWig?e=WmhxHC)


The execution of these files will save a pre-trained model in form of .pt file extension can be reused later to perform down-stream classification task for any dataset using the given command for loading of pre-trained model.


### Implementation Assumptions:
Some small assumptions are made in the assignment as follows: 

1. It is assumed that the initial pre-trained embeddings used to improve the ELMO Embeddings are Glove Embeddings.

2. Only first 100000 N-grams are used for Training the Forward and Backward Embeddings due to Computational Issues.

3. 6-Sized N-Grams are taken for training of BiLSTM Model.

4. Maximum Sequence Length for Downstream Classification Task is taken out to be the Length of the Sentence having lengths greater than 95% of the sentences in the Training Dataset. That comes out to be 59.