from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer
import pymo.viz_tools as vt
import matplotlib.pyplot as plt


parser = BVHParser()

parsed_data = parser.parse('walk-01-normal-azumi.bvh')

mp = MocapParameterizer('position')

positions = mp.fit_transform([parsed_data])

# positions is a pandas dataframe and can be used for whatever purpose here

vt.draw_stickfigure3d(positions[0], frame=150)
plt.show()
