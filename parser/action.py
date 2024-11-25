from typing import List, Tuple

class Action:
    
    def __init__(
        self, 
        name: str, 
        params: List[str],  # Not support types for now
        pos_pre: List[Tuple[str]], 
        neg_pre: List[Tuple[str]], 
        add_eff: List[Tuple[str]], 
        del_eff: List[Tuple[str]]
    ):
        self.name = name
        self.params = params
        self.pos_pre = pos_pre
        self.neg_pre = neg_pre
        self.add_eff = add_eff
        self.del_eff = del_eff
    
    def __str__(self):
        return f'Action Name: {self.name}\n' + \
               f'   Parameters: {self.params}\n' + \
               f'   Positive Preconditions: {self.pos_pre}\n' + \
               f'   Negative Preconditions: {self.neg_pre}\n' + \
               f'   Add Effects: {self.add_eff}\n' + \
               f'   Delete Effects: {self.del_eff}\n'
    
    def __eq__(self, other: 'Action'):
        return self.__dict__ == other.__dict__

    def assign_val(self, objects: List[str]):
        
        # Types note supported for now
        
        if not self.params:
            return self
        
        if len(self.params) != len(objects):
            raise ValueError('Number of objects does not match number of parameters')
        
        # Assign values to parameters
        pos_pre = self.pos_pre.copy()
        neg_pre = self.neg_pre.copy()
        add_eff = self.add_eff.copy()
        del_eff = self.del_eff.copy()
        for (obj, param) in zip(objects, self.params):
            pos_pre = [tuple(obj if x == param else x for x in tup) for tup in pos_pre]
            neg_pre = [tuple(obj if x == param else x for x in tup) for tup in neg_pre]
            add_eff = [tuple(obj if x == param else x for x in tup) for tup in add_eff]
            del_eff = [tuple(obj if x == param else x for x in tup) for tup in del_eff]
        
        return Action(self.name, objects, pos_pre, neg_pre, add_eff, del_eff)


if __name__ == '__main__':
    print('----------Test Case 1----------')
    action = Action(
        name='move_onTable',
        params=['?x', '?y'],
        pos_pre=[('clear', '?x'), ('on', '?x', '?y')],
        neg_pre=[],
        add_eff=[('onTable', '?x'), ('clear', '?y')],
        del_eff=[('clear', '?x'), ('on', '?x', '?y')]
    )
    print(action)

    assigned = action.assign_val(['A', 'B'])
    print(assigned)
    
    
    
    print('----------Test Case 2----------')
    
    action = Action(
        name='move_to',
        params=['?x', '?y'],
        pos_pre=[('clear', '?x'), ('clear', '?y'), ('onTable', '?x')],
        neg_pre=[],
        add_eff=[('on', '?x', '?y')],
        del_eff=[('clear', '?x'), ('onTable', '?x')]
    )
    print(action)

    assigned = action.assign_val(['A', 'B'])
    print(assigned)
