import transitions as ts


class Matter(object):

    def evaporate(self):
        print("I'm evaporating...")


lump = Matter()

# The states
states = ['solid', 'liquid', 'gas', 'plasma']

# And some transitions between states. We're lazy, so we'll leave out
# the inverse phase transitions (freezing, condensation, etc.).
transitions = [
    {'trigger': 'melt', 'source': 'solid', 'dest': 'liquid'},
    {'trigger': 'evaporate', 'source': 'liquid', 'dest': 'gas'},
    {'trigger': 'sublimate', 'source': 'solid', 'dest': 'gas'},
    {'trigger': 'ionize', 'source': 'gas', 'dest': 'plasma'}
]

# Initialize
machine = ts.Machine(
    lump, states=states, transitions=transitions, initial='liquid')

print(lump.state)

lump.evaporate()

print(lump.state)

lump.melt()
