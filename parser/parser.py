import re
from action import Action

class Parser:
    
    SUPPORTED_REQUIREMENTS = [':strips', ':negative-preconditions']
    
    def __init__(self):
        super().__init__()
    
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
        return eval(data_new)[0]
            
if __name__ == '__main__':
    
    domain_parser = Parser()
    problem_parser = Parser()
    
    print('------ Test 1: from file to token list ------')
    
    domain_file = './blockstacking/blockstacking.pddl'
    domain_parsed = domain_parser.parse_file(domain_file)
    
    problem_file = './blockstacking/problem1.pddl'
    problem_parsed = problem_parser.parse_file(problem_file)
    
    domain_target = ['define', ['domain', 'blockstacking'], [':requirements', ':strips', ':negative-preconditions'], [':predicates', ['clear', '?x'], ['on', '?x', '?y'], ['ontable', '?x']], [':action', 'move_to', ':parameters', ['?x', '?y'], ':precondition', ['and', ['clear', '?x'], ['clear', '?y'], ['ontable', '?x']], ':effect', ['and', ['not', ['clear', '?y']], ['on', '?x', '?y'], ['not', ['ontable', '?x']]]], [':action', 'move_totable', ':parameters', ['?x', '?y'], ':precondition', ['and', ['clear', '?x'], ['on', '?x', '?y']], ':effect', ['and', ['ontable', '?x'], ['not', ['on', '?x', '?y']], ['clear', '?y']]]]
    
    problem_target = ['define', ['problem', 'problem1'], [':domain', 'blockstacking'], [':objects', 'a', 'b'], [':init', ['clear', 'a'], ['clear', 'b'], ['ontable', 'a'], ['ontable', 'b']], [':goal', ['and', ['on', 'a', 'b'], ['ontable', 'b']]]]
        
    assert domain_parsed.__eq__(domain_target)
    assert problem_parsed.__eq__(problem_target)
    
    print('--------------- Test 1 passed ---------------')
    
    
    
    