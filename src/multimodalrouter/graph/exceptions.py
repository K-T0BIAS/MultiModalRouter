

class FrozenException(Exception):
    
    def __init__(self, caller: str):
        super().__init__(f"\033[91m[MultiModalRouter Frozen Exception]\033[0m The graph is frozen and cannot be changed by {caller}")


class NotSafeReadingStateWarning:
    
    def __init__(self, caller: str):
        self.warning :str = f"\n\033[93m[MultiModalRouter Warning]\033[0m {caller} the graph is not in a safe reading state.\n Consider freezing first"

    def __repr__(self):
        return self.warning
    
    def __str__(self):
        return self.warning