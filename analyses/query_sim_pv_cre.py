
from aisynphys.database import default_db as db

q=db.query(db.Cell)
q=q.filter(db.Cell.cre_type == 'pvalb,sim1')
cells = q.all()
#import pdb; pdb.set_trace()
print(cells)

