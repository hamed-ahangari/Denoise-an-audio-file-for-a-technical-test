# Denoise an audio file for a technical test
Denoise an audio file as a technical test during a job interview

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GV3mDzLHG-KrMRUNsVYCP0R73IHk6N2E?usp=sharing)

**Table of Contents**
- [Denoise an audio file for a technical test](#denoise-an-audio-file-for-a-technical-test)
  - [Task description](#task-description)
  - [Exploration of noises](#exploration-of-noises)
    - [Diagrams](#diagrams)
    - [Frequency-amplitude diagram](#frequency-amplitude-diagram)
  - [Reducing the noises](#reducing-the-noises)
  - [Code](#code)
    - [Notebook on Colab](#notebook-on-colab)

## Task description
While I was in a job interview for getting a position in the deep learning department of one of the leading AI startups in Iran, I received a task to denoise an audio file.
I was told to write a python code to try my best to reduce or remove noises within the given audio signal.

The audio file is a very short slice from an audiobook in Persian, with some presented artificial noises within its signal.

## Exploration of noises
My interviewer gave me neither details about the number of noises nor their types. So I need to explore that information at the very first step.
### Diagrams
The below are waveforms and frequency-amplitude diagrams of the given noisy audio signal.

![figure 1](https://raw.githubusercontent.com/hamed-ahangari/Denoise-an-audio-file-for-a-technical-test/main/Figure_1%20%28before%20denoising%29.png)

### Frequency-amplitude diagram
Also, the frequency-amplitude diagram is zoomed in for frequencies up to 1200 Hz. It was obvious to me that I am faced with three noises:
- two beep noises with ~440 Hz and ~1010 Hz frequencies
- and white noise with uniform distribution

![figure 2](https://raw.githubusercontent.com/hamed-ahangari/Denoise-an-audio-file-for-a-technical-test/main/Figure_2%20%28before%20denoising%29.png)

  

## Reducing the noises
At the first step, I used two butterfly bandstop filters to remove the beep noises. The result was very acceptable.

Secondly, in order to remove the white noise, I did sample a part of audio with a length of 0.1 sec that has nothing but noisy sound. Then I used the idea from [Audacity](https://wiki.audacityteam.org/wiki/How_Audacity_Noise_Reduction_Works) and [this blog post](https://timsainburg.com/noise-reduction-python.html#noise-reduction-python). The steps of that idea could be summarised as below:

1. An STFT is calculated over the noisy sample
2. Statistics are calculated over STFT of the noise
3. A threshold is calculated based upon the statistics of the noise
4. Also, STFT is calculated over the whole signal
5. A mask is determined by comparing the signal STFT to the threshold for each window
6. The mask is applied to the STFT of the signal
7. the result of the mask is inverted at the last step to obtain the denoised audio signal

Finally and after trying to remove beep noises and reduce white noise, the waveform looked like below.
![figure 3](https://raw.githubusercontent.com/hamed-ahangari/Denoise-an-audio-file-for-a-technical-test/main/Figure_3%20%28after%20denoising%29.png)

## Code
I have provided both [my python code](https://github.com/hamed-ahangari/Denoise-an-audio-file-for-a-technical-test/blob/main/%5BCODE%5D%20Test%20-%20Denoiser%20-%20Voice%20converter.py), written in an object-oriented manner, and an [IPython notebook](https://github.com/hamed-ahangari/Denoise-an-audio-file-for-a-technical-test/blob/main/%5BNOTEBOOK%5D%20Technical_test_Denoising_an_audio.ipynb).

### Notebook on Colab
You can run my codes on Google Colab via [this shared notebook](https://colab.research.google.com/drive/1GV3mDzLHG-KrMRUNsVYCP0R73IHk6N2E?usp=sharing).