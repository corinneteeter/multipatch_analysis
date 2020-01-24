
from aisynphys.database import default_db as db

q=db.query(db.Experiment)
expts = q.all()
import pdb; pdb.set_trace()
print(expts)
