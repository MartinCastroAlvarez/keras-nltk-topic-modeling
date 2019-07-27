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
- [Tagging Text with SpaCy](https://spacy.io/)
- [Named Entity Recognition with SpaCy](https://medium.com/@manivannan_data/spacy-named-entity-recognizer-4a1eeee1d749)
- [Tagging People in a Text using NLTK](https://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list)
- [Stanford NER](https://stanfordnlp.github.io/CoreNLP/index.html#download)

## Performance
![epochs-alt](./models/8036_loss.png)

## Instructions

### Setup
Install the virtual environment.
```bash
virtualenv -p python3 .env
source .env/bin/activate
```
Download the SpaCy assets:
```bash
python3 -m spacy download en_core_web_sm
```
Download the NLTK assets in Python3:
```bash
python3
>>> import nltk
>>> nltk.download('averaged_perceptron_tagger')
>>> nltk.download('maxent_ne_chunker')
>>> nltk.download('words')
```
Download the Stanford NER library from [here](https://stanfordnlp.github.io/CoreNLP/index.html#download).

### Training
Execute the AI script to generate a prediction model:
```bash
python3 barcelona.py
```
The results will be available in the [models](./models) directory.  
It will generate something like this, depending on the accuracy of the model:
```bash
total 20144
-rw-rw-r--. 1 martin martin    23530 Jul 26 11:32 8036_loss.png
-rw-rw-r--. 1 martin martin     1464 Jul 26 11:32 8036_model.json
-rw-rw-r--. 1 martin martin      340 Jul 26 11:32 8036_tokenizer.pkl
-rw-rw-r--. 1 martin martin 20591008 Jul 26 11:32 8036_weights.h5
```

### Predicting
Execute the AI script to make a prediction:
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

### Reporting
You may then generate a report using this script:
```bash
python3 madrid.py
```
The report will be available [here](./madrid.txt).
You should get something like this:
```bash
[https://www.market-inspector.co.uk/blog/2018/04/best-coffee-blogs-2018]
- crude: 0.34337058663368225
- earn: 0.30171316862106323
- money-fx: 0.18945614993572235
- acq: 0.07513884454965591
- trade: 0.034326352179050446
- money-supply: 0.02358020655810833
- ship: 0.012165199033915997
- grain: 0.008449743501842022
- sugar: 0.004694306291639805
- gold: 0.003508565714582801
- veg-oil: 0.0006372999050654471
- interest: 0.0005667438381351531
- oilseed: 0.0004590681928675622
- cocoa: 0.00036324531538411975
- livestock: 0.0003095024439971894
- gnp: 0.0002835058548953384
- wheat: 0.0001192237323266454
- alum: 0.00011752318823710084
- iron-steel: 9.341971599496901e-05
- reserves: 8.244166383519769e-05
- dlr: 7.928269769763574e-05
- gas: 7.064364763209596e-05
- nat-gas: 5.216805948293768e-05
- meal-feed: 4.832234844798222e-05
- strategic-metal: 4.0608691051602364e-05
- copper: 4.051349242217839e-05
- bop: 3.771424962906167e-05
- pet-chem: 3.680576628539711e-05
- zinc: 3.5635421227198094e-05
- rubber: 2.2437330699176528e-05
- coffee: 2.1934592950856313e-05
- retail: 1.6022182535380125e-05
- cotton: 1.347179295407841e-05
- ipi: 9.784547728486359e-06
- orange: 9.196536666422617e-06
- carcass: 8.057489139901008e-06
- cpi: 4.591794095176738e-06
- housing: 3.7308027458493598e-06
- jobs: 3.6221856589691015e-06
- heat: 2.4453836431348464e-06
I0727 12:25:53.057993 140271176644224 valencia.py:277] Score calculated. | sf_score=12.070065908208077
[Score] 12.070065908208077
[Stanford People] ['Michael', 'Allen', 'Smith', 'Veneziano', 'Tanya', 'Newton', 'Resi', 'Jim', 'Seven', 'James', 'Hoffmann', 'James', 'Stephen', 'Leighton', 'Kenneth', 'Davids', 'Nick', 'Danijela', 'Dean']
[SpaCy Nouns] [...]
[SpaCy Verbs] [...]
[SpaCy People] []
[NLTK People] ['Coffee Blogs', 'Coffee Blogs', 'Market Inspector', 'Facebook Pixel Code', 'Le HTML5', 'Denmark', 'Sweden', 'Menu', 'Small Businesses Telephone Systems', 'Rental', 'Photocopier Suppliers', 'A3 Printers', 'Coffee Machines', 'Coffee Vending Machines', 'Machines', 'Coffee Machine Suppliers', 'Cash Registers', 'Rental', 'Vehicle Tracking', 'Fleet Management', 'Transport Management Systems', 'Blog', 'Blog', 'Coffee Blogs', 'Coffee Blogs', 'Coffee Badge', 'Market Inspector', 'Coffee Blogs', 'Best Guru Coffee Blogs', 'Best Unique Coffee Blogs', 'Best Review Coffee Blogs', 'Best Stylish Coffee Blogs', 'Award', 'Best Coffee', 'Best Coffee', 'Coffee Blogs', 'Coffee Logo', 'Coffee', 'Coffee', 'Coffee', 'Michael Allen Smith', 'Veneziano Coffee Logo', 'Veneziano Coffee Roasters', 'Veneziano Coffee Roasters', 'Coffee Roasters', 'Veneziano Coffee Roasters', 'Motto', 'Quills Coffee Logo', 'Coffee', 'Quills Coffee', 'Quills Coffee', 'Coffee', 'Blueprint Coffee Logo', 'Blueprint Coffee', 'Blueprint Coffee', 'Blueprint Coffee', 'Best Guru Coffee Blogs', 'Perfect Daily Grind Logo', 'Perfect Daily Grind', 'Perfect Daily Grind', 'Perfect Daily Grind', 'Barista Champion', 'Tanya Newton', 'Perfect Daily Grind', 'Bean Ground Logo', 'Bean Ground', 'Bean Ground', 'Bean Grounded', 'Mark', 'Bean Ground', 'Barista Hustle Logo', 'Barista Hustle', 'Barista Hustle', 'Barista Hustle', 'Barista Hustle', 'Coffee Logo', 'Coffee', 'Coffee', 'Jimseven Logo', 'Jim Seven', 'Jim Seven', 'James Hoffmann', 'James', 'Coffee', 'Jimseven', 'Unique Coffee Blogs', 'Coffee Museum Logo', 'Coffee Museum', 'Coffee Museum', 'Cátia Biscaia', 'Coffee Museum', 'Atlas Coffee Logo', 'Atlas Coffee Club', 'Atlas Coffee Club', 'Atlas Coffee Club', 'Horse Logo', 'Horse Coffee', 'Fair Trade', 'Horse Coffee', 'Home Grounds', 'Home Grounds', 'Logo', 'Coffee Loving Cardmakers', 'Best Review Coffee Blogs', 'Has Bean Coffee Logo', 'Has Bean Coffee', 'Has Bean', 'Stephen Leighton', 'Hasbean', 'Bean Coffee', 'Good Coffee', 'Good Coffee', 'Coffee Review Logo', 'Coffee Review', 'Coffee Review', 'Coffee Review', 'Kenneth Davids', 'Coffee Review', 'Coffee Detective Logo', 'Coffee Detective', 'Coffee Detective', 'Nick', 'Coffee Detective', 'Coffee Detective', 'Coffee Concierge Logo', 'Stylish Coffee Blogs', 'Logo', 'Yorkshire', 'Coffee Stylish Logo', 'Coffee Stylish', 'Coffee Stylish', 'Stylish', 'Coffee', 'Coffee', 'Coffee', 'Joyride Coffee', 'Best Coffee', 'Match', 'Quotes', 'Market Inspector Logo', 'Apply', 'Market Inspector Scholarships', 'Denmark', 'Sweden', 'Scroll', 'Change', 'Close']
[NLTK Organizations] ['ROBOTS', 'INDEX', 'FOLLOW', 'Best Coffee Blog', 'DeviceSpec', 'TRACKJS', 'HTML5', 'IE', 'pixelDepth', 'phoneNum', 'Customer', 'mobileMenuText', 'Telephone Systems', 'VoIP Phone', 'Telephone Systems', 'PBX System', 'Business Mobile Phones', 'Leasing', 'Business Printers', 'Hire', 'UK', 'Commercial Coffee Machines', 'ePOS Systems', 'POS Systems', 'Best Coffee Blog', 'Coffee Machines', 'INeedCoffee', 'INeedCoffee', 'CRS Logo', 'CRS', 'CRS', 'CRS', 'NGO', 'NGO', 'CRS', 'YouTube', 'Atlas Coffee Club', 'Kicking Horse', 'DIY', 'Cardmakers', 'HasBean', 'Good Coffee Me Logo', 'Good Coffee', 'Good Coffee', 'Coffee Review', 'Coffee Detective', 'Coffee', 'Coffee Concierge', 'Coffee Concierge', 'Coffee Concierge', 'Coffee Concierge', 'Harrogate', 'Harrogate', 'Harrogate', 'DIY', 'Senses Coffee Logo', 'Coffeetographer', 'Coffeetographer', 'Coffeetographer', 'Coffeetographer', 'JoyRide Logo', 'USA', 'Joyride Mission', 'Best Coffee Blogs', 'Best Coffee Blogs', 'LinkedIn', 'greenB', 'blueDB', 'greenB', 'blueDB', 'greenB', 'Long Lane', 'phoneNum', 'Privacy', 'Cookies Policy', 'Market', 'footerPartenrs', 'footerLang', 'getCookieValue', 'getCookieValue', 'cookieValue', 'getCookieValue', 'AWMCookieName', 'cookieValue', 'cookieIntervalId', 'cookieIntervalId', 'triggerGaEvent', 'modalBoxId', 'modalBoxId', 'closeModal', 'closeModal', 'clickCheckbox', 'triggerGaEvent', 'closeModal']
[NLTK Locations] ['United Kingdom', 'Norway', 'Finland', 'France', 'Believe', 'Veneziano', 'Melbourne', 'Australia', 'Melbourne', 'Brisbane', 'Sydney', 'Adelaide', 'Coffeeland', 'Action', 'Louisville', 'Good', 'Suffice', 'Add', 'Coffee', 'Coffee', 'Resi', 'Japanese', 'Coffee', 'Japan', 'Add', 'Japanese', 'Japan', 'Earth', 'Danijela', 'Coffee', 'Dean', 'United Kingdom', 'Norway', 'Finland', 'France']
```
