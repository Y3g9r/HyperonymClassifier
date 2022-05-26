import re
import pymorphy2

class hyperonym_handler:

    def __init__(self, model_link = None):
        self.model_link = model_link
        self.morph = pymorphy2.MorphAnalyzer()

    def sentence_parse(self, sentence):
        splited_sentence = sentence.split(' ')
        normalazed_words = []
        for word in splited_sentence:
            new_word = re.sub("[^А-Яа-я]","",word)
            normalazed_words.append(self.morph.parse(new_word)[0].normal_form)
        print(normalazed_words)