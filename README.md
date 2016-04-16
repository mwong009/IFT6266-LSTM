# IFT6266
IFT6266 git repository

Vocal Synthesis Project - IFT6266 H16

For best examples of audio reporduction see: audio_reconstuction.wav

*IMPORTANT*: this git repo had been recently reset, therefore changes made before march 31 will not be shown

# Files
main.py
main file to train and genrate audio

lstmp.py
2 Layer LSTM recurrent projection neural network (LSTMP)

convert.sh
For converting audio.wav to audio.mp4

# Notes
NLL = - log exp{0.5(x_t-\mu)^2/(2 * pi * sigma)}
