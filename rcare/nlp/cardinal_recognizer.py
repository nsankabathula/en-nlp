#import spacy;
#from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Span, Doc, Token
from spacy.matcher import Matcher
from rcare.utils import Utils
import string

class CardinalRecognizer(object):
    name = "CardinalRecognizer"
    label = 'CUSTOM_CARDINAL'
  
    def __init__(self, nlp, patterns, label):        
        self.matcher = Matcher(nlp.vocab)
        for pattern in patterns:
            self.matcher.add(label, None, pattern)  

        Doc.set_extension('cardinals', force = True, default=[])
        Token.set_extension('is_cardinal', force = True, default=False)
        Doc.set_extension('has_cardinal', getter=self.has_cardinal, force = True )
        Span.set_extension('has_cardinal', getter=self.has_cardinal, force = True)
        
            
    def __call__(self, doc):
        matches = self.matcher(doc)        
        for i in range(len(matches)):
            match_id, curr_start, curr_end = matches[i]                        
            entity  = Span(doc, curr_start, curr_end, label=match_id)
            for token in entity :
                token._.set('is_cardinal', True)
                
            #doc._.cardinals.append(span)
            doc.ents = list(doc.ents) + [entity ]
            
        return doc
    
    def has_cardinal(self, tokens):
        """Getter for Doc and Span attributes. Returns True if one of the tokens
        is a tech org. Since the getter is only called when we access the
        attribute, we can refer to the Token's 'is_tech_org' attribute here,
        which is already set in the processing step."""
        return any([t._.get('is_cardinal') for t in tokens])
    
    
    DEFAULT_CARDINAL_PATTERN = [[{'ENT_TYPE': 'CARDINAL'},{'ORTH':'.'}] ] + \
    list(map(lambda i : [{'ORTH': '('}, {'ORTH': Utils.int_to_roman(i)}, {'ORTH': ')'}], range(1, 20))) + \
    list(map(lambda i : [{'ORTH': '('}, {'ORTH': Utils.int_to_roman(i, True)}, {'ORTH': ')'}], range(1, 20))) + \
    list(map(lambda char : [{'ORTH': '('}, {'ORTH': char}, {'ORTH': ')'}], string.ascii_letters)) + \
    list(map(lambda digit : [{'ORTH': '('}, {'ORTH': digit}, {'ORTH': ')'}], string.digits))
    
    
