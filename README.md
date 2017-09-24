# TensorflowGuitarTrainer
uses tensorflow sequince to sequence Models to learn musical concepts and guitar styles based on tablature
skimmed from the web.

## Running
run the data collection tool in guitarData.py before running the training algorithm in guitarTrain.py 

## Dependancies
* Tensorflow
* Beutiful Soup 4

## Challanges
### Styling
Guitar Tablature has no standard style so it was difficult to collect consistant data and lots of tablature had to be discarded
as a resualt. This hurt the training set but probably had little effect on the resaults of the training.
### Models
Initialy it seemed clear that a sequence to sequence model trained from bar to bar of the guitar music would make the most 
sense to generate novel music from given input. after training it became appearent that the previous bar of the tab had little
to no effect on the next bar. However, there was an overall pattern to the tablature, so it would go to show that an encoded 
value could have an effective baring to showing overall style of a piece.

## Acomplishments
Learned how a sequence to sequence recurent neural network works functionally.

## Notes
LSTM models are good for language comprehension so it would go to show that they would be good for comprehending musical
ideas. It turns out there is not enugh style carried in the data of the finger positioning on the guitar. It was interesting
to see that some chord shapes could be noticed after many training generations.
