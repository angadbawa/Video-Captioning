# Video-Captioning

Video Captioning is a sequential learning model that employs an encoder-decoder architecture. It accepts a video as input and produces a descriptive caption that summarizes the content of the video.

The significance of captioning stems from its capacity to enhance accessibility to videos in various ways. An automated video caption generator aids in improving the searchability of videos on websites. Additionally, it facilitates the grouping of videos based on their content by making the process more straightforward.

<h2 id="Dataset">Dataset</h2>

This project utilizes the <a href="https://opendatalab.com/MSVD">MSVD</a> dataset, which consists of 1450 training videos and 100 testing videos, to facilitate the development of video captioning models.

<h2 id="Setup">Setup</h2>

Clone the repository: <code>git clone https://github.com/angadbawa/Video-Captioning </code>

Video Caption Generator: <code>cd Video-Captioning</code>

Create environment: <code>conda create -n video_caption python=3.7</code>

Activate environment: <code>conda activate video_caption</code>

Install requirements: <code>pip install -r requirements.txt</code>

<h2 id="Usage">Usage</h2>

To utilize the pre-trained models, follow these steps:

1. Add a video to the "data/testing_data/video" folder.
2. Execute the "predict_realtime.py" file using the command: <code>python predict_realtime.py</code>.

For quicker results, extract the features of the video and save them in the "feat" folder within the "testing_data" directory.

To convert the video into features, run the "extract_features.py" file using the command: <code>python extract_features.py</code>.

For local training, run the "train.py" file. Alternatively, you can use the "Video_Captioning.ipynb" notebook.

<h2 id="Model">Model</h2>

<h3 id="Introduction">Introduction</h3>

The model presented here employs a Sequence-to-Sequence (Seq2Seq) architecture with Back-to-Back Long Short-Term Memory (LSTM) cells, utilizing pre-extracted Convolutional Neural Network (CNN) features from video frames. This report provides a comprehensive overview of the model components and justifications for their selection.

<h3 id="Dataset">Dataset</h3>

This project utilizes the MSVD dataset, which consists of 1450 training videos and 100 testing videos, to facilitate the development of video captioning models.

<h3 id="ModelArchitecture">Model Architecture</h3>

The model architecture consists of two main components - an encoder and a decoder.

<h4 id="Encoder">Encoder</h4>

- **CNN Feature Extraction**: Video frames are first converted into CNN features using a pre-trained VGG16 model. VGG16 is chosen for its balance between performance and simplicity. The dense features extracted from the videos serve as the input to the subsequent Seq2Seq model.
- **LSTM Encoder**: The extracted CNN features are then fed into an LSTM encoder. The LSTM layers capture temporal dependencies in the video frames and generate a fixed-size representation of the video content.

<h4 id="Decoder">Decoder</h4>

- **LSTM Decoder**: The LSTM decoder processes the encoded video representation and generates a sequence of words that form the caption. Back-to-Back LSTMs allow the model to capture both short and long-term dependencies in the sequential data.
- **Word Embeddings**: Word embeddings are used to represent each word in the generated sequence. The embedding matrix is learned during training, mapping words to continuous vector representations.

<h4 id="Seq2SeqModel">Sequence-to-Sequence Model</h4>

- The Seq2Seq architecture enables the model to handle input sequences of variable length and generate output sequences. It is particularly suitable for tasks where the input and output have different lengths, as in video captioning.
- The model is trained to minimize the cross-entropy loss between predicted and actual words in the captions. Adam optimizer is employed for efficient optimization.

<h4 id="ImplementingAttention">Implementing Attention</h4>

Due to the many limitations of Keras (such as the inability to return the states of each time step of LSTM, etc.), I decided to first pass the sequence of the encoder model output through permute and a softmax dense layer for attention (Decide which frame to focus on), convert the dimensions and then concatenate it with the decoder input and feed it into the decoder for training.

<h3 id="Justifications">Justifications</h3>

- **Pre-trained VGG16**: VGG16 is chosen for its simplicity and good performance in image classification tasks. The model has proven effective in extracting meaningful features from videos, making it suitable for video captioning where image content is crucial.
- **LSTMs**: LSTMs are selected for their ability to capture long-range dependencies in sequential data. Video frames inherently have temporal relationships, and LSTMs excel in modeling such relationships, making them suitable for video captioning.
- **Sequence-to-Sequence Model**: Seq2Seq models are well-suited for tasks where the input and output have varying lengths, as is the case in video captioning. This architecture enables the model to effectively encode the video content and generate captions of varying lengths.
- **Word Embeddings**: Word embeddings provide a continuous representation of words, preserving semantic relationships. Learning embeddings allows the model to capture the meaning of words in the context of the video content.

<h2 id="RunPreprocessing">Run Preprocessing</h2>

To run pre-processing:
<code>
python VideoCaptioningPreProcessing.py process_main --video_dest '/content/extracted_folder/YouTubeClips' --feat_dir '/content/extracted_folder/YouTubeClips/features/' --temp_dest '/content/extracted_folder/YouTubeClips/temp/' --img_dim 224 --channels 3 --batch_size=128 --frames_step 80
</code>

<h2 id="RunModel">Run Model</h2>

To run the model:
<code>
!python Video_seq2seq.py process_main --video_dest '/content/extracted_folder/YouTubeClips' --feat_dir '/content/extracted_folder/YouTubeClips/features/' --temp_dest '/content/extracted_folder/YouTubeClips/temp/' --img_dim 224 --channels 3 --batch_size=128 --frames_step 80
</code>
