# UKGE based on BERT and Bayesian Network  
Uncertain Knowledge Graphs Embedding is implemented by means of BERT pretrained natural language model. Exploring approximate inference on the embedding space. Considering the commonsense reasoning based on these works.

## Environment require
* python 3.9
* pytorch 2.0.0+cpu
* pretrained BERT models, such as 'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz", etc.

## Uncertain Knowledge Graphs datasets for benchmark
  In data directory, train.tsv is the training data, test.tsv the testing data and val.tsv the validation data.    
  
  | dataset | entities | relations | facts | average of confidence | standard deviation | ratio of facts/relations |             
  | ---- | ---- | ---- | ---- | ---- | ---- | ---- | 
  |CN15k| 15,000 | 36 | 234,675 | 0.627 | 0.234 | 15.6 |    
  |PPI5k|  4,999 |  7 | 271,666 | 0.415 | 0.213 | 54.3 |    

## Comparison 
The unit of all values on MSE and MAE is  10$^{-2}$. "-" means the corresponding data did not report in literatures.
  <table>
	<tr>
	    <th>dataset</th>
	    <th colspan = 3>CN15k</th>
	    <th colspan = 3>PPI5k</th>  
	</tr >
	<tr>
	    <td>metrics</td><td>MSE</td><td>MAE</td><td>epochs</td><td>MSE</td><td>MAE</td><td>epochs</td>
	</tr>
	<tr>
	    <td>URGE</td><td>10.32</td><td>22.72</td><td>-</td><td>1.44</td><td>6.00</td><td>-</td>
	</tr>
	<tr>
	    <td>UKGE$_{rect}$</td><td>8.61</td><td>19.90</td><td>>100</td><td>0.95</td><td>3.79</td><td>>100</td>
	</tr>
	<tr>
	    <td>UKGE$_{logi}$</td></td><td>9.86</td><td>20.74</td><td>>100</td><td>0.96</td><td>4.07</td><td>>100</td>
	</tr>
	<tr>
	    <td>UKG$_s$E</td></td><td>7.71</td><td>21.34</td><td>19</td><td>0.98</td><td>5.98</td><td>80</td>
	</tr>
	<tr>
	    <td>UKGE$_{bert}$</td></td><td>6.61</td><td>19.3</td><td>2</td><td>0.85</td><td>5.38</td><td>75</td>
	</tr>
	</table>
  

## Run
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
    ...
    
    Prediction for (HEAD, ?relation?, TAIL) RANKING:
    Head = ' car '
    Tail = ' people '
      
          relation candidate 1 = ('hasa', 0.8294004)
          relation candidate 2 = ('usedfor', 0.760266) *
          relation candidate 3 = ('createdby', 0.65154946)
          relation candidate 4 = ('definedas', 0.63723207)
          relation candidate 5 = ('atlocation', 0.6002464)
    
## Under working
The project is under working. We are very sorry for any trouble to you. You are welcome to put forward your valuable opinions by comments.