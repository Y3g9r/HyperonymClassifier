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
        return tx.run("MATCH (w:Word {name: $the_word}) RETURN w.name", the_word=the_word).single()

class hyperonym_handler:

    def __init__(self, model_link = None):
        self.model_link = model_link
        self.morph = pymorphy2.MorphAnalyzer()
        self.db_client = Neo4jClient('bolt://192.168.85.128:7687', 'neo4j', 'newPassword')

    def sentence_parse(self, sentence):
        print(self.db_client.is_word_exist('мышь 0'))
        splited_sentence = sentence.split(' ')
        normalazed_words = []
        for word in splited_sentence:
            new_word = re.sub("[^А-Яа-я]","",word)
            normalazed_words.append(self.morph.parse(new_word)[0].normal_form)