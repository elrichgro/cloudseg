All the file and data is in /mnt/dsl3lab-scrach/llodrant

To update code just call `git pull`.

Install pip3 dependencies in your own profile with `pip3 install --user -r requirements.txt` 
Didn't want to bother with installing virtualenv.


Setup config-spaceml.json with your parameters (check nvidia-smi to see which GPUs are available and set them up).
Don't use more than 2.

run `./train.sh optimized_3`

profit 
