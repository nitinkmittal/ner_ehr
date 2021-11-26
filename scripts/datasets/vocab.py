from collections import Counter

class Vocab:
    """Creates a vocabulary for dataset"""

    def __init__(self):
        self.stats = Counter()
    
    def add_to_vocab(self, token: str, tag: str) -> None:
        """
        Adds the token and it's corresponding tags and counts in a dictionary of dictionary
        stats = {'token1': {'tag1': count, 'tag2': count}, 'token2':{'tag1':count},...}

        Args:
            token: string
            tag: string containing entity class of the token

        Returns:
            None
        """
        
        if token in self.stats:
            pass      
        else:
            self.stats[token] = Counter()
        
        self.stats[token][tag]+=1
