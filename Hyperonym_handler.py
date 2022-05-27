import re
import pymorphy2
from neo4j import GraphDatabase
from HyperonymNN import predict

class Neo4jClient:

    def __init__(self, uri, user, password):
        # set  the connection with neo4j
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.session = self.driver.session()

    def is_word_exist(self, the_word):
        with self.driver.session() as session:
            result = session.write_transaction(self.__is_word_exist, the_word)
        if result is not None:
            return True
        return False

    @staticmethod
    def __is_word_exist(tx, the_word):
        # single return true if we have at least 1 element
        the_word = the_word + ' ' + '\d*'
        return tx.run("MATCH (w:Word) WHERE w.name=~$the_word  RETURN w.name", the_word=the_word).single()

    def get_definitions(self, the_word):
        the_word = the_word + ' ' + '\d*'
        result = self.session.run("MATCH (w:Word) WHERE w.name=~$the_word RETURN w.definition as word_def", the_word=the_word)
        results = [record['word_def'] for record in result]
        return results

class hyperonym_handler:

    def __init__(self, model_link = None):
        self.model_link = model_link
        self.morph = pymorphy2.MorphAnalyzer()
        self.db_client = Neo4jClient('bolt://192.168.85.128:7687', 'neo4j', 'newPassword')


    def sentence_parse(self, sentence):
        sentence_mapping = self.__sentence_handler(sentence)
        normalazed_words = []
        analyzed_information = []

        for mapping in sentence_mapping:
            normalazed_words.append(self.morph.parse(sentence[mapping[0]:mapping[1]])[0].normal_form)

        for i in range(len(normalazed_words)):
           if self.db_client.is_word_exist(normalazed_words[i]):
               current_word_definition = self.db_client.get_definitions(normalazed_words[i])
               data_for_nn = []
               for definition in current_word_definition:
                    data_for_nn.append([[sentence_mapping[i]], [sentence], ["EMPTY"], definition, [3]])
               nn_predicted = predict(data_for_nn)
               analyzed_information.append([normalazed_words[i], current_word_definition[nn_predicted.index(max(nn_predicted))][0]])
        return analyzed_information

    def __sentence_handler(self, sentence):
        word_flag = True
        mapping = []
        first_pos = 0
        second_pos = 0
        sentence = sentence + ' '
        for symbol in range(len(sentence)):
            if sentence[symbol].isalpha() and not word_flag:
                word_flag = True
                first_pos = symbol
            else:
                if word_flag and not sentence[symbol].isalpha():
                    second_pos = symbol
                    mapping.append((first_pos, second_pos))
                    word_flag = False
        return mapping