## Multimodal Music Emotion Recognition using Convolutional Neural Network
### MUMT 621 Final Project

This project explores the performance of multimodal learning on Music Emotion Recognition on the [Multimodal MIREX-Like Emotion Dataset](http://mir.dei.uc.pt/downloads.html) proposed in 2013. Unimodal and bimodal methods on audio and lyrical features were built and compared. Both middle fusion and late fusion were applied for bimodal methods. This result demonstrates that the combination of two feature domains can improve the MER performance. 

- `dataset.ipynb` downloads and preprocesses the dataset.
- `audio_preprocess.py` extracts several audio features from the audio data.
- `audio.py` trains and evaluates the unimodal model based on audio.
- `l_crnn.py` trains and evaluates the unimodal model based on lyrics.
- `f_crnn.py` trains and evaluates the bimodal model with middle fusion.
- `fl_crnn.py` trains and evaluates the bimodal model with late fusion.
