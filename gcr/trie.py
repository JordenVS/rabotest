class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class ProcessTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens):
        node = self.root
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = TrieNode()
            node = node.children[tok]
        node.is_end = True

    def allowed_next(self, prefix_tokens):
        node = self.root
        for tok in prefix_tokens:
            if tok not in node.children:
                return set()
            node = node.children[tok]
        return set(node.children.keys())