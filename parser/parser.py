import re
from action import Action

class Parser:
    
    SUPPORTED_REQUIREMENTS = [':strips', ':negative-preconditions']
    
    def __init__(self):
        super().__init__()
        self.do_init()
        
    def do_init(self):
        self.domain_name = None
        self.requirements = []
        self.objects = {}
        self.actions = []
        self.predicates = {}
    
    def parse_file(self, filename: str):
        with open(filename, 'r') as f:
            data = f.read()
            # Remove comments
            data = re.sub(r';.*', '', data, flags=re.MULTILINE).lower()
        
        # Tokenization, parentheses as distinct tokens
        pattern = r'[()]|[^\s()]+'
        
        # Use Python eval to build hierachical list
        data_new = ""
        
        for t in re.findall(pattern, data):
            if t == '(':
                data_new += '['
            elif t == ')':
                data_new += '],'
            else:
                data_new += f'"{t}",'
        try:
            return eval(data_new)[0]
        except:
            raise ValueError("Not a valid PDDL content")
    
    def parse_domain(self, domain_filename, requirements=SUPPORTED_REQUIREMENTS):
        tokens = self.parse_file(domain_filename)
        if type(tokens) is list and tokens.pop(0) == 'define':
            self.do_init()
            while tokens:
                token_group = tokens.pop(0)
                token_head = token_group.pop(0)
                if token_head == 'domain':
                    self.domain_name = token_group.pop(0)
                elif token_head == ':requirements':
                    for req in token_group:
                        if req not in requirements:
                            raise ValueError(f'Requirement {req} not supported')
                    self.requirements = token_group
                elif token_head == ':predicates':
                    pass    ### TODO
                elif token_head == ':precondition':
                    pass    ### TODO
                elif token_head == ':action':
                    pass    ### TODO
                else:
                    raise ValueError(f'{token_head} not supported')
            
        else:
            raise ValueError(f'File {domain_filename} does not match domain pattern')
        
            
if __name__ == '__main__':
    
    parser = Parser()
    
    print('------ Test 1: from file to token list ------')
    
    domain_file = './blockstacking/blockstacking.pddl'
    domain_parsed = parser.parse_file(domain_file)
    
    problem_file = './blockstacking/problem1.pddl'
    problem_parsed = parser.parse_file(problem_file)
    
    domain_target = ['define', ['domain', 'blockstacking'], [':requirements', ':strips', ':negative-preconditions'], [':predicates', ['clear', '?x'], ['on', '?x', '?y'], ['ontable', '?x']], [':action', 'move_to', ':parameters', ['?x', '?y'], ':precondition', ['and', ['clear', '?x'], ['clear', '?y'], ['ontable', '?x']], ':effect', ['and', ['not', ['clear', '?y']], ['on', '?x', '?y'], ['not', ['ontable', '?x']]]], [':action', 'move_totable', ':parameters', ['?x', '?y'], ':precondition', ['and', ['clear', '?x'], ['on', '?x', '?y']], ':effect', ['and', ['ontable', '?x'], ['not', ['on', '?x', '?y']], ['clear', '?y']]]]
    
    problem_target = ['define', ['problem', 'problem1'], [':domain', 'blockstacking'], [':objects', 'a', 'b'], [':init', ['clear', 'a'], ['clear', 'b'], ['ontable', 'a'], ['ontable', 'b']], [':goal', ['and', ['on', 'a', 'b'], ['ontable', 'b']]]]
        
    assert domain_parsed.__eq__(domain_target)
    assert problem_parsed.__eq__(problem_target)
    
    print('--------------- Test 1 passed ---------------')
    
    
    
    