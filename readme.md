# UKGE based on BERT  
Uncertain Knowledge Graphs Embedding is implemented by means of BERT pretrained natural language model. Exploring approximate inference on the embedding space. Considering the commonsense reasoning based on these works.

## environment
* python 3.6
* torch 1.6.0+cpu
* transformers 4.0.0
* others   

## Uncertain Knowledge Graphs datasets for benchmark
  | dataset | entities | relations | facts | average of confidence | standard deviation | ratio of facts/relations |             
  | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
  |CN15k| 15,000 | 36 | 234,675 | 0.627 | 0.234 | 15.6 |    
  |PPI5k|  4,999 |  7 | 271,666 | 0.415 | 0.213 | 54.3 |    

## run
    python.exe ./prediction.py
    Using Theano backend.
    Model: "bert_fine_tuning_UKGE"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm_1 (LSTM)                (None, 768)               4721664   
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 769       
    =================================================================
    Total params: 4,722,433
    Trainable params: 4,722,433
    Non-trainable params: 0
    _________________________________________________________________
    
## Under working
The project is under working. We are very sorry for any trouble to you. You are welcome to put forward your valuable opinions by comments.