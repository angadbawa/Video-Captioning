import functools
import operator
import os
import cv2
import time

import numpy as np
import extract_features

import config
import model


class CaptionVideo:
    """
    Real-time video description using a pre-trained model.
    """
    def __init__(self, config):
        self.latent_dim = config.latent_dim
        self.num_encoder_tokens = config.num_encoder_tokens
        self.num_decoder_tokens = config.num_decoder_tokens
        self.time_steps_encoder = config.time_steps_encoder
        self.max_probability = config.max_probability

        self.tokenizer, self.inf_encoder_model, self.inf_decoder_model = model.inference_model()
        self.save_model_path = config.save_model_path
        self.test_path = config.test_path
        self.search_type = config.search_type
        self.num = 0

    def greedy_search(self, loaded_array):
        """
        Greedy search to predict the caption.
        """
        inv_map = self.index_to_word()
        states_value = self.inf_encoder_model.predict(loaded_array.reshape(-1, 80, 4096))
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        final_sentence = ''
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1

        for _ in range(15):
            output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
            states_value = [h, c]
            output_tokens = output_tokens.reshape(self.num_decoder_tokens)
            y_hat = np.argmax(output_tokens)

            if y_hat == 0 or inv_map[y_hat] is None or inv_map[y_hat] == 'eos':
                break

            final_sentence += inv_map[y_hat] + ' '
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, y_hat] = 1

        return final_sentence

    def decode_sequence2bs(self, input_seq):
        """
        Beam search to predict the caption.
        """
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value, [], [], 0)
        return decode_seq

    def beam_search(self, target_seq, states_value, prob, path, lens):
        """
        Recursive beam search to predict the caption.
        """
        global decode_seq
        node = 2
        output_tokens, h, c = self.inf_decoder_model.predict([target_seq] + states_value)
        output_tokens = output_tokens.reshape(self.num_decoder_tokens)
        sampled_token_index = output_tokens.argsort()[-node:][::-1]
        states_value = [h, c]

        for i in range(node):
            sampled_char = self.tokenizer.index_word.get(sampled_token_index[i], '')

            if sampled_char != 'eos' and lens <= 12:
                p = output_tokens[sampled_token_index[i]]
                prob_new, path_new = list(prob), list(path)
                prob_new.append(p)
                path_new.append(sampled_char)
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index[i]] = 1.
                self.beam_search(target_seq, states_value, prob_new, path_new, lens + 1)
            else:
                p = output_tokens[sampled_token_index[i]]
                prob_new = list(prob)
                prob_new.append(p)
                p = functools.reduce(operator.mul, prob_new, 1)
                if p > self.max_probability:
                    decode_seq = path
                    self.max_probability = p

    def decoded_sentence_tuning(self, decoded_sentence):
        """
        Tune the decoded sentence to remove unnecessary tokens.
        """
        decode_str = []
        filter_string = ['bos', 'eos']
        last_string = ""

        for c in decoded_sentence:
            if c in filter_string:
                continue
            if last_string == c:
                continue
            if c:
                decode_str.append(c)
            last_string = c

        return decode_str

    def index_to_word(self):
        """
        Invert the word tokenizer.
        """
        return {value: key for key, value in self.tokenizer.word_index.items()}

    def get_test_data(self):
        """
        Load the features array from the test data.
        """
        file_list = os.listdir(os.path.join(self.test_path, 'video'))
        file_name = file_list[self.num]
        path = os.path.join(self.test_path, 'feat', file_name + '.npy')

        if os.path.exists(path):
            f = np.load(path)
        else:
            model = extract_features.model_cnn_load()
            f = extract_features.extract_features(file_name, model)

        self.num = (self.num + 1) % len(file_list)
        return f, file_name

    def test(self):
        """
        Generate inference test outputs.
        """
        X_test, filename = self.get_test_data()

        if self.search_type == 'greedy':
            sentence_predicted = self.greedy_search(X_test)
        else:
            decoded_sentence = self.decode_sequence2bs(X_test)
            decode_str = self.decoded_sentence_tuning(decoded_sentence)
            sentence_predicted = ' '.join(decode_str)

        self.max_probability = -1
        return sentence_predicted, filename

    def main(self, filename, caption):
        """
        Display the video with the caption.
        """
        cap1 = cv2.VideoCapture(os.path.join(self.test_path, 'video', filename))
        cap2 = cv2.VideoCapture(os.path.join(self.test_path, 'video', filename))
        caption = '[' + ' '.join(caption.split()[1:]) + ']'

        while cap1.isOpened():
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if ret1:
                imS = cv2.resize(frame1, (480, 300))
                cv2.putText(imS, caption, (100, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_4)
                cv2.imshow("VIDEO CAPTIONING", imS)

            if ret2:
                imS = cv2.resize(frame2, (480, 300))
                cv2.imshow("ORIGINAL", imS)

            if not ret1 and not ret2:
                break

            if cv2.waitKey(25) == 27:  # ESC key to exit
                break

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_to_text = CaptionVideo(config)
    while True:
        print('Generating Caption...\n')
        start_time = time.time()
        video_caption, filename = video_to_text.test()
        end_time = time.time()
        print('Caption:', video_caption)
        print('Time taken: {:.2f} seconds'.format(end_time - start_time))

        video_to_text.main(filename, video_caption)

        play_video = input('Should I play the video? (y/n): ')
        if play_video.lower() != 'y':
            break
