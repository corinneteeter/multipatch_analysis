"""It was too difficult to deal with development tables in the production database
So here I save them out so I can delete the original tables.
"""

import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from weighted_error_lib import *
import pdb


session=db.Session()
out = session.query(db.AvgFirstPulseFit, db.AvgFirstPulseFit2,
                    db.AvgFirstPulseFit3, db.AvgFirstPulseFit4).join(
                    db.Pair).filter(db.Pair.synapse == True
                    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit2.pair_id
                    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit3.pair_id
                    ).filter(db.AvgFirstPulseFit.pair_id == db.AvgFirstPulseFit4.pair_id)

#out = session.query(db.AvgFirstPulseFit4, db.Pair).join(db.Pair).filter(db.Pair.synapse == True)

df=pd.read_sql_query(out.statement, out.session.bind)
df.to_csv('test.csv')


