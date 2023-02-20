from melodie.element import Element


class Agent(Element):
    def __init__(self, agent_id: int):
        self.id = agent_id

    def setup(self):
        pass

    def post_setup(self):
        pass

    def __repr__(self) -> str:
        d = {k: v for k, v in self.__dict__.items() if
             not k.startswith("_")}
        return "<%s %s>" % (self.__class__.__name__, d)
