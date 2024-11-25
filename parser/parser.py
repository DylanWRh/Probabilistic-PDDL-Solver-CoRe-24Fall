import re
from .action import Action

class Parser:
    
    SUPPORTED_REQUIREMENTS = [':strips', ':negative-preconditions']
    
    def __init__(self):
        super().__init__()
    
    def parse_file(self, filename: str):
        with open(filename, 'r') as f:
            data = f.read()
            # Remove comments
            data = re.sub(r';.*', '', data, flags=re.MULTILINE).lower()
        
        brackets = []