This is a c++ wrapper for denosing auto-encoders. You can combine the single layer auto encoder to a multi layer version using some pre-train algorithms( Stacked Auto Encoders)
![ABC](http://deeplearning.stanford.edu/wiki/images/thumb/8/8d/STL_CombinedAE.png/500px-STL_CombinedAE.png) 


Wiki : http://deeplearning.stanford.edu/wiki/index.php/Autoencoders_and_Sparsity

Using this wrapper, you can train/save/load the model, and the model is saved as a standard json.
you can also get the hidden layer output for your next prediction module such as LR/BPR 

The denoising auto-encoder is a stochastic version of the auto-encoder. Intuitively, a denoising auto-encoder does two things: 
try to encode the input (preserve the information about the input), and try to undo the effect of a corruption process stochastically
applied to the input of the auto-encoder. The latter can only be done by capturing the statistical dependencies between the inputs.


More details here: http://deeplearning.net/tutorial/dA.html#da

How to use:

cd build
cmake ../
make

This lib rely on C++ Boost lib and gtest lib, so make sure you have installed them.

The input data format is from the Liblinear data format
http://www.csie.ntu.edu.tw/~cjlin/liblinear/

you can see examples in the data folder

# train the model

cd data
../build/AutoEncoder -m 1 -h 2 -v 4 -n 10000 ./test.encode ./model.json

# get the hidden layer output

cd data
../build/AutoEncoder -m 2 -h 2 -v 4 -n 10000 ./test.encode ./model.json ./test.encode.hidden

if your have some questions, send email : stiger104@126.com


