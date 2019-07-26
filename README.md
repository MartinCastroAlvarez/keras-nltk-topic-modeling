# Spain
Neural Network implementation using Python3 and Keras.
This app predicts the topic of any given text.

![image-alt](./valencia.jpg)

## References:
- [Keras Tutorial](https://towardsdatascience.com/text-classification-in-keras-part-1-a-simple-reuters-news-classifier-9558d34d01d3
)
- [Saving Keras Model](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)
- [How to make predictions with Keras](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)
- [Persisting the Tokenizer](https://intellipaat.com/community/491/keras-text-preprocessing-saving-tokenizer-object-to-file-for-scoring)
- [Predicting a new text](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/3.6-classifying-newswires.ipynb)

## Training
![epochs-alt](./models/8036_loss.png)

## Usage
Install the virtual environment.
```bash
virtualenv -p python3 .env
source .env/bin/activate
```
Execute the AI script to generate a prediction model:
```bash
python3 barcelona.py
```
The results will be available in the [models](./models) directory.  
Then, txecute the AI script to make a prediction:
```bash
python3 valencia.py 8036 "https://www.google.com/"
```
Pages will be cached in the [html](./html) directory.  
The results will be available in the [predictions](./predictions) directory.  
You should get something like this:
```bash
[http://www.ritualroasters.com/]
- trade: 0.41248124837875366
- acq: 0.30653807520866394
- earn: 0.038371115922927856
- crude: 0.03154604882001877
- grain: 0.018684780225157738
- sugar: 0.018534166738390923
- gold: 0.01808004267513752
- oilseed: 0.016360409557819366
- iron-steel: 0.013791962526738644
```
You may then generate a report using this script:
```bash
python3 madrid.py
```
You should get something like this:
```bash
earn                           41.382154922932386
acq                            33.2141318959184
crude                          18.85195611056406
trade                          17.117614451795816
money-fx                       11.601139926700853
grain                          11.376437215832993
interest                       9.942362987286828
money-supply                   7.787595164816594
gold                           7.470945799126639
ship                           7.069725468982824
sugar                          6.376406626854987
iron-steel                     4.9058672383714566
oilseed                        4.6498465190449
coffee                         4.411411816827695
```
