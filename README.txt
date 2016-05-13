
# train the model

cd data
../build/AutoEncoder -m 1 -h 2 -v 4 -n 10000 ./test.encode ./model.json

# get the hidden layer output

cd data
../build/AutoEncoder -m 2 -h 2 -v 4 -n 10000 ./test.encode ./model.json ./test.encode.hidden