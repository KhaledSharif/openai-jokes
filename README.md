# OpenAI Jokes
### Training a character-level language model on a corpus of jokes.

#### How to obtain the training dataset

```
git clone https://github.com/taivop/joke-dataset.git
cd joke-dataset
export PATH_TO_JOKES=$PWD
```

#### How to train

```
git clone https://github.com/KhaledSharif/openai-jokes.git
cd openai-jokes
python3 lstm_text_generation.py --path=$PATH_TO_JOKES/reddit_jokes.json --batch_size=512 --epochs=1000
```

#### Training arguments

```
python3 lstm_text_generation.py --help
```

```
  --path PATH           Path to JSON file containing jokes
  --learning_rate LEARNING_RATE
                        Learning rate as a float
  --clipping_value CLIPPING_VALUE
                        Clipping value of the gradient as a float
  --number_of_layers NUMBER_OF_LAYERS
                        Number of layers in the LSTM, integer
  --lstm_size LSTM_SIZE
                        Number of neurons in each individual LSTM, integer
  --lstm_bidirectional LSTM_BIDIRECTIONAL
                        LSTM direction (uni or bi), boolean
  --batch_normalization BATCH_NORMALIZATION
                        Batch normalization (enabled or disabled), boolean
  --epochs EPOCHS       Number of training epochs, integer
  --batch_size BATCH_SIZE
                        Training batch size, integer
```

#### Output after 100 epochs

```
Epoch 100/100
833320/833320 [==============================] - 391s 469us/step - loss: 0.1055

----- generating text after epoch: 99
----- diversity: 0.2
----- generating with seed: "n had a right for privacy he said theref"
n had a right for privacy he said therefore her number she said got a new baby boy masking words wave horseage he says a happy but i haven't allowed to do a gorgeous but i really needs to go back to the doctor and says 'that's mord friend yes she's a shot | why are standing bakery karaaaaaaaaaaaaaaaaaaaaaaaaaaaamn! | i made a doctor did in a joke about poor dad still not | if you either music away for a minute? firm years and i take him
----- diversity: 0.5
----- generating with seed: "n had a right for privacy he said theref"
n had a right for privacy he said therefore he comes back in her she shoots the disground bread and close to their house who do you see? one is a shut of water | what do you call a megaphian for a listen in the deal is that i go home to come to see my own cowarderph? a rectomous was really bad and no punchline its like a partner but she wings up and fortunately asks her poor what's the most love advim he says back to me no idea now that
----- diversity: 1.0
----- generating with seed: "n had a right for privacy he said theref"
n had a right for privacy he said therefore her petacle say? he he was a huge liber's house | donald trump advance have a women perious long? none asked the way's the other punchister but i just wave the dish the blonde replies well everything that he's probably illegal they susped to come as a christmas tree's testicles for it while he was about to over his dick into a gate the temple says i was a doctor? damate who was throwing up i s
----- diversity: 1.2
----- generating with seed: "n had a right for privacy he said theref"
n had a right for privacy he said therefore her pay he decides he'd have a problem of the next hate good news and voodoo dick fix meme?! | i have to fix the iphone before i saw the poor woman he is bored but there is a very problem of minutes feet constation for the sterical of the story? robotoge it a newspaper are you aware of people i was over! | my wife and i asked my wife the ship! let's remain the fridge answer wall well son a wom
```
