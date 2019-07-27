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
[http://www.ritualroasters.com/]
- trade: 0.47086700797080994
- acq: 0.2597869336605072
- earn: 0.04419371858239174
- crude: 0.03594855219125748
- grain: 0.019196435809135437
- gold: 0.016572486609220505
- oilseed: 0.016033483669161797
- sugar: 0.012674011290073395
- iron-steel: 0.012585194781422615
- livestock: 0.009270072914659977
- nat-gas: 0.009241299703717232
- veg-oil: 0.009032214991748333
- ship: 0.005692950449883938
- coffee: 0.005276921205222607
- copper: 0.004891602788120508
- interest: 0.004873819649219513
- bop: 0.004332657437771559
- rubber: 0.003639825852587819
- gas: 0.003595465561375022
- cotton: 0.003554644761607051
- pet-chem: 0.0034948273096233606
- cocoa: 0.0032427709084004164
- carcass: 0.0031560687348246574
- retail: 0.0029264551121741533
- silver: 0.002836642088368535
- alum: 0.002637687139213085
- gnp: 0.0024283200036734343
- wheat: 0.0022102834191173315
- ipi: 0.0022035357542335987
- zinc: 0.002202308736741543
- meal-feed: 0.002188275568187237
- strategic-metal: 0.0020325281657278538
- money-supply: 0.001999250380322337
- reserves: 0.001982838148251176
- dlr: 0.0018926480552181602
- cpi: 0.0018284650286659598
- wpi: 0.0016993408789858222
- orange: 0.0016445540823042393
- money-fx: 0.0014349583070725203
- tin: 0.0011600113939493895
I0727 12:20:29.508213 140398731085440 valencia.py:277] Score calculated. | sf_score=9.812694048844396
[Score] 9.812694048844396
[Stanford People] ['Sarah', 'KAVAN', 'AARON', 'VAN', 'DER', 'GROEN', 'ANDREW', 'GIBSON']
[SpaCy Nouns] ['san francisco', '/ menu', '/ store](http://ritual.myshopify.com/) / news / locations / brew', '[/ wholesale](https://ritual-wholesale-store.myshopify.com/', '[facebook](images', 'icon.png)](https://www.facebook.com/ritualroasters', '[instagram](images', '[twitter](images/menu-twitter-icon.png)](http://twitter.com/ritualcoffee', '[flikr](images', '/about-us.png', 'a cup', 'coffee', 'nothing', 'a caffeine delivery\nvehicle', 'it', 'it', 'it', 'you', 'the morning', 'you', 'a long afternoon', 'the last decade', 'things', 'a lot', 'a few people', 'the country', 'a cup', 'coffee', 'you', 'the beans', 'farmers', 'you', 'you', 'the beans', 'yourself', 'such a\nway', 'nothing', 'the coffee', 'you', 'the coffee', 'down-to-the-second precision', 'ritual', 'a pioneer', 'this delicious shift', 'coffee consciousness', 'we', 'our doors', 'valencia street', 'what', 'san francisco', 'our goal', 'our goal', 'the very best cup', 'coffee', 'period', 'we', 'the years', 'the care', 'attention', 'we', 'our process', 'every coffee', 'it', 'our coffee bars', 'your cup', 'we', 'coffee', 'we', 'it', 'pretty much everybody', 'who', 'a moment', 'a really, really\ngood cup', 'coffee', 'their lives', '### upcoming events', '*july', '\\- cupping', 'valencia', '1 pm', '*july', '* \\- public coffee tasting', 'ritual hayes valley', '12:30 pm', '[store](images', '[shop now](http://ritual.myshopify.com/', 'our team', 'june 18th', 'free coffee', 'work', 'us', 'ritual', 'we', 'these stellar positions', '**people operations manager', 'https://www.linkedin.com/jobs/view/1309243916/', 'we', 'an experienced leader', 'who', 'people', 'coffee', '"people operations', 'ritual', 'the leader', 'systems', 'the people', 'who', 'ritual the amazing company', 'it', 'we', 'we', 'a people operations\nmanager', 'the growth', 'the culture', 'we', 'the past 14 years', 'hiring', 'retention', 'communication', 'the\npeople operations manager', 'a full-time, salaried position', 'san\nfrancisco', 'the people operations manager', 'the owner', 'your resume', 'a cover letter', 'bullet points', 'you', 'a great people operations manager', '[s@noyes-works.com](mailto:s@noyes-works.com', 'priority application deadline', 'august 9th', '*baristas', 'hayes valley', 'napa', '* :\n<https://www.localwise.com', 'job/38260-barista/18701-ritual-coffee-roasters', 'san-', 'francisco', '>\n\nritual baristas', 'the experiences', "people's eyes", 'good coffee', 'they', 'people', 'our coffee', 'our producers', 'outstanding service', 'beverage preparation', 'our baristas', 'dozens', 'fresh coffees', 'the year', 'they', 'their\nskills', 'baristas', 'ritual', 'fun', 'opportunity', 'growth', 'all jobs', 'we', 'candidates', 'the outside', 'ritual', 'a competitive wage', 'a complete benefits package', 'medical insurance', 'dental insurance', 'commuter benefits', 'sick pay', '401k\nplan', 'a resume', 'person', 'email', '](/news', 'our-team', 'previous post', '[locations](images/locations.png)\n\n### mission', '[ 1026 valencia street', 'san francisco', 'monday - friday', '6am - 8pm', '7am - 8pm', '7am - 8pm', '4th', 'july', '7am- 6pm', '### haight', '1300 haight street', 'san francisco', '(https://goo.gl/maps/m3mv39axgvm)\n\nmonday - friday', '6:30am - 7pm', '7am - 7pm', '4th', 'july', 'flora grubb gardens', '1634 jerrold ave', 'san francisco', 'monday - thursday', '- 4pm', ': 10am - 5pm', 'july', '[ 2299 market st unit', 'a  \nsan francisco', '- 5pm', 'saturday - sunday', '8am - 5pm', 'july', ': 8am', 'public market', '610 first street  \nnapa', 'california', '(http://goo.gl/maps/dxfkm', 'every day', ': 7am - 8pm', '4th', 'july', '7am - 7pm\n\n### hayes valley', 'proxy  \n432b octavia', 'san francisco', '(http://goo.gl/maps/mqvqk)\n\n(almost) every day', ': 7am - 7pm', 'july 3rd', '3pm', 'brew', '[](images', 'images', 'images', '[](images', '(images', '[](images', 'images', 'images', '[](images', '(images', '[contact', '/contact-us.png', '**general information', '*  \n[info@ritualroasters.com](mailto:info@ritualroasters.com', 'ritual roastery', '1050 howard', '**wholesale inquiries', 'april', '[wholesale@ritualroasters.com](mailto:wholesale@ritualroasters.com', '**green coffee buyer', '**  \naaron van der groen', '**catering and event inquiries', 'catering', 'requests', 'our catering site', 'andrew gibson', '[events@ritualroasters.com](mailto:events@ritualroasters.com', '**mailorder inquiries', '*  \n[mailorder@ritualroasters.com](mailto:mailorder@ritualroasters.com', '[facebook](images', 'icon.png)](https://www.facebook.com/ritualroasters', '[instagram](images/instagram-icon.png)](http://instagram.com/ritualcoffee', '[twitter](images', 'twitter-icon.png)](http://twitter.com/ritualcoffee', '[flikr](images', 'flikr-', 'news', 'events', 'offers](images', '[](images', 'ritual](images', '[privacy policy](privacy', 'policy.php', '[terms', 'conditions](terms', 'and-', '[site', 'ccxxiids](http://www.ccxxiids.com/']
[SpaCy Verbs] ['make', 'be', 'do', 'matter', 'taste', 'have', 'slap', 'prop', 'have', 'change', 'scatter', 'discover', 'can', 'be', 'source', 'know', 'roast', 'be', 'remove', 'brew', 'have', 'be', 'open', 'start', 'call', 'be', 'craft', 'have', 'learn', 'lavish', 'be', 'include', 'taste', 'go', 'do', 'do', 'make', 'do', 'work', 'have', 'have', 'change', 'join', 'post', 'come', 'be', 'grow', 'have', 'be', 'look', 'love', 'oversee', 'will', 'develop', 'implement', 'find', 'hire', 'develop', 'celebrate', 'make', 'ritual', 'be', 'grow', 'need', 'continue', 'foster', 'have', 'create', 'be', 'base', 'apply', 'send', 'be', 'explain', 'be', 'sarah', 'be', 'project', 'valencia', 'ca', 'create', 'open', 'can', 'be', 'connect', 'exact', 'get', 'work', 'be', 'be', 'expect', 'train', 'improve', 'be', 'work', 'know', 'have', 'be', 'be', 'post', 'seek', 'offer', 'include', 'apply', 'submit', 'read', 'join', 'ca', 'saturday', 'sunday', 'ca', 'bayview', 'ca', 'sunday', 'close', 'ca', 'friday', 'oxbow', 'ca', 'close', 'a.m.', 'guide', 'click', 'ca', 'visit', 'reach', 'sign']
[SpaCy People] []
[NLTK People] ['Ritual Coffee Roasters', 'Ritual Coffee', 'Ritual', 'Valencia Street', 'People Operations Manager', 'Sarah', 'Baristas', 'Valencia', 'Ritual', 'San Francisco', 'San Francisco', 'Gardens', 'Jerrold Ave', 'San Francisco', 'Market St Unit', 'San Francisco', 'Oxbow Public Market', 'Street', 'San Francisco', 'CLICK']
[NLTK Organizations] ['DOCTYPE', 'ABOUT', 'CUPPING', 'RITUAL', 'PUBLIC', 'RITUAL', 'SHOP', 'JOIN', 'POSTED', 'Ritual', 'People Operations Manager', 'People Operations', 'Hayes Valley', 'Ritual', 'LOCATIONS', 'CASTRO', 'Almost', 'Proxy', 'Almost', 'BREW', 'GO', 'CONTACT', 'GENERAL', 'INFO', 'INFO', 'HOWARD', 'WHOLESALE', 'APRIL', 'WHOLESALE', 'WHOLESALE', 'GREEN', 'AARON', 'AARON', 'AARON', 'CATERING', 'RITUAL', 'ANDREW', 'GIBSON', 'EVENTS', 'MAILORDER', 'MAILORDER', 'MAILORDER', 'NEWSLETTER', 'RITUAL', 'PRIVACY', 'TERMS', 'CONDITIONS', 'SITE', 'CCXXIIDS']
```
