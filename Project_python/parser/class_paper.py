import xml.etree.ElementTree as ET
from gensim.utils import simple_preprocess
from nltk.stem import WordNetLemmatizer
import warnings
import spacy
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet as wn

nlp = spacy.load('en_core_web_md')

class Paper:
    path=""
    tree=ET
    name=""
    authors=[]
    keywords=[]
    abstract=""
    introduction=""
    body=""
    results=""
    discussion=""
    conclusions=""
    ackowledgments=""
    references=""
    
    class IterLog:
        abstract=0
        keywords=0
        introduction=0
        discussion=0
        conclusions=0
    iter = IterLog

    def __init__(self, path=""):
        self.path = path
        if len(path) > 0:
            self.extract_information()
        
    #todo mejorar pasando el iter desde el que seguir?
        
    def extract_information(self, path=""):
        if len(path) == 0:
            path = self.path
        self.path = path
        self.tree = ET.parse(self.path)
        
        self.extract_file_description()
        self.extract_profile_description()
        self.extract_introduction()
        self.extract_discussion()
        self.extract_conclusions()
        self.extract_body()
        
    def extract_file_description(self):
        tag = ""
        prev_tag = ""
        flag = False
        for elem in self.tree.iter():
            prev_tag = tag
            tag = self.parse_tag(elem)

            if tag == "fileDesc":
                # print("file description init")
                flag = True
                
            elif tag == "profileDesc":
                # print('file description end')
                return
            
            if tag == "title" and flag == True:
                if prev_tag == "titleStmt":
                    self.name = elem.text
            
    def extract_profile_description(self):
        tag = ""
        prev_tag = ""
        flag = False
        self.keywords = []
        self.abstract = ""
        
        def get_abstract(iter):
            abstract = ""
            for elem in iter:
                tag = self.parse_tag(elem)
                if tag == "p":
                    abstract += " " + elem.text
            return abstract
                
        def get_keywords(keywords_elem):
            text = keywords_elem.text
            if len(text) > 20:
                keywords = list(text.split(' '))
                return keywords
            else:
                keywords = []
                for elem in keywords_elem.iter():
                    tag = self.parse_tag(elem)
                    if tag == "term":
                        keywords.append(elem.text)
                return keywords
                
        iteration = 1
        for elem in self.tree.iter():
            prev_tag = tag
            tag = self.parse_tag(elem)

            if tag == "profileDesc":
                # print("profile description init")
                flag = True
            
            if tag == "abstract" and flag:
                self.abstract = get_abstract(elem.iter())
                self.iter.abstract = iteration
            elif tag == "keywords" and flag:
                self.keywords = get_keywords(elem)
                self.iter.keywords = iteration
                
            iteration += 1
                
    def extract_introduction(self):
        tag = ""
        flag = False
        self.introduction = ""
        iteration = 1
        for elem in self.tree.iter():
            tag = self.parse_tag(elem)

            if tag == "body":
                flag = True
                
            if flag:
                if tag == "head":
                    if elem.text == None:
                        continue
                    head = self.standarize_text(elem.text, lemmatize=True)
                    if self.is_word_in_text("introduction", head, precision=False):
                        introduction = self.extract_text_from_section(prev_elem.iter())
                        self.introduction = introduction
                        self.iter.introduction = iteration
                        break
            prev_elem = elem
            iteration += 1
        
    def extract_discussion(self):
        tag = ""
        flag = False
        self.discussion = ""
        iteration = 1
        for elem in self.tree.iter():
            tag = self.parse_tag(elem)

            if tag == "body" or tag == "back":
                flag = True
                
            if flag:
                if tag == "head":
                    if elem.text == None:
                        continue
                    head = self.standarize_text(elem.text, lemmatize=True)
                    if self.is_word_in_text("discussion", head, precision=False) and not(self.is_word_in_text("table", head, precision=False) or self.is_word_in_text("figure", head, precision=False)):
                        discussion = self.extract_text_from_section(prev_elem.iter())
                        self.discussion = discussion
                        self.iter.discussion = iteration
                        break
                elif tag == "note":
                    if elem.text == None:
                        continue
                    text_lem = self.standarize_text(elem.text, lemmatize=True)
                    if self.is_word_in_text("discussion", text_lem, precision=False) and not(self.is_word_in_text("table", text_lem, precision=False) or self.is_word_in_text("figure", text_lem, precision=False)):
                        discussion = elem.text
                        self.discussion = discussion
                        self.iter.discussion = iteration
                        break
            prev_elem = elem
            iteration += 1
        
    def extract_conclusions(self):
        tag = ""
        flag = False
        self.conclusions = ""
        iteration = 1
        for elem in self.tree.iter():
            tag = self.parse_tag(elem)

            if tag == "body" or tag == "back":
                flag = True
                
            if flag:
                if tag == "head":
                    if elem.text == None:
                        continue
                    head = self.standarize_text(elem.text, lemmatize=True)
                    if self.is_word_in_text("conclusion", head) or (self.is_word_in_text("summary", head) and not(self.is_word_in_text("table", head, precision=False) or self.is_word_in_text("figure", head, precision=False))):
                        conclusions = self.extract_text_from_section(prev_elem.iter())
                        self.conclusions = conclusions
                        self.iter.conclusions = iteration
                        break
                elif tag == "note":
                    text_lem = self.standarize_text(elem.text, lemmatize=True)
                    if self.is_word_in_text("conclusion", text_lem, precision=False):
                        conclusions = elem.text
                        self.conclusions = conclusions
                        self.iter.conclusions = iteration
                        break
            prev_elem = elem
            iteration += 1
            
        if len(self.conclusions) == 0:
            if len(self.discussion) > 0:
                warnings.warn("No conclusions founds, assuming discussion for file {}".format(self.name))
                self.conclusions = self.discussion
                self.iter.conclusions = self.iter.discussion
                self.discussion = ""
                
    def extract_body(self):
        tag = ""
        self.body = ""
        iteration = 1
        for elem in self.tree.iter():
            if iteration > self.iter.introduction:
                if iteration > self.iter.conclusions:
                    break
                # the body text is assummed all the text between intro and conclusions
                tag = self.parse_tag(elem)
                if tag == "head" or tag == "p":
                    if elem.text == None:
                        continue
                    self.body += elem.text
            iteration += 1
        
    def extract_text_from_section(self, iter):
        text = ""
        for elem in iter:
            tag = self.parse_tag(elem)
            if tag == "p":
                text += elem.text
            elif tag == "ref":
                if not(elem.tail == None):
                    text += elem.tail
                if not(elem.text == None):
                    text += elem.text
        return text
    
    def print_description(self):
        print('#######################')
        print('- File name: {}'.format(self.name))
        print('- Keywords: {}'.format(",".join(self.keywords)))
        print('- Abstract: {}'.format(self.abstract))
        print('- Conclusions: {}'.format(self.conclusions))
        
    def export_as_dictionary(self):
        dict = {}
        dict["name"] = self.name
        dict["abstract"] = self.abstract
        dict["keywords"] = ",".join(self.keywords)
        dict["introduction"] = self.introduction
        dict["body"] = self.body
        dict["conclusions"] = self.conclusions
        return dict
    
    def parse_tag(self, elem):
        tag = elem.tag
        return tag[(tag.find('}') + 1):]
    
    def is_word_in_text(self, word, text, precision=True):
        flag = False
        
        if precision:
            for textWord in text.split(' '):
                token = nlp(word)
                token2 = nlp(textWord)
                if token.similarity(token2) > 0.75:
                    flag = True
        else:
            if word in text:
                flag = True
        return flag      
    
    def standarize_text(self, text, lemmatize=False, pos="n"):
        tokens = simple_preprocess(text)
        if lemmatize:
            wnl = WordNetLemmatizer()
            tokens = [wnl.lemmatize(token, pos=pos) for token in tokens]
        return ' '.join(tokens)