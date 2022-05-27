import re
import pymorphy2
from neo4j import GraphDatabase

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
        print(results)

class hyperonym_handler:

    def __init__(self, model_link = None):
        self.model_link = model_link
        self.morph = pymorphy2.MorphAnalyzer()
        self.db_client = Neo4jClient('bolt://192.168.85.128:7687', 'neo4j', 'newPassword')


    def sentence_parse(self, sentence):
        sentence_mapping = self.__sentence_handler(sentence)
        splited_sentence = sentence.split(' ')
        normalazed_words = []
        print(sentence_mapping)

        for word in splited_sentence:
            new_word = re.sub("[^А-Яа-я]","",word)
            normalazed_words.append(self.morph.parse(new_word)[0].normal_form)

        #for word in normalazed_words:
        #    if self.db_client.is_word_exist(word):
        #        current_word_definition = self.db_client.get_definitions(word)
        #        for definition in current_word_definition:

    def __sentence_handler(self, sentence):
        temp_new_sentence = sentence.split(' ')
        new_sentence = []
        for splited_word in temp_new_sentence:
            temp_split = splited_word.split('!')
            for sub_temp_split in temp_split:
                new_sentence.append(sub_temp_split)


