# Spain
Neural Network implementation using Python3 and Keras.
This app predicts the topic of any given text.

![image-alt](./valencia.jpg)

## References:
- [Keras Tutorial](# https://towardsdatascience.com/text-classification-in-keras-part-1-a-simple-reuters-news-classifier-9558d34d01d3
)
- [Saving Keras Model](https://machinelearningmastery.com/save-load-keras-deep-learning-models/)
- [How to make predictions with Keras](https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/)

## Training
![epochs-alt](./8000_loss.png)

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
Execute the AI script to make a prediction:
```bash
python3 valencia.py 7987 "https://www.google.com/"
```
