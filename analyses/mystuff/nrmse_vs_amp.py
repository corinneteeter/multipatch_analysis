
import multipatch_analysis.database as db
import multipatch_analysis.connection_strength as cs 
import pandas
import matplotlib.pyplot as plt 
import numpy as np

# db.Session()
# pair=db.experiment_from_timestamp(1547248417.685).pairs[(2, 7)]
# afpf = afpf.fit_first_pulses(pair)

# print(afpf)


session=db.Session()
afpfs=session.query(db.AvgFirstPulseFit).join(db.Pair).filter(db.Pair.synapse == True).all()
print len(afpfs)



nrmse=[]
amp=[]
for afpf in afpfs:
    nrmse.append(afpf.ic_nrmse)
    amp.append(afpf.ic_amp)

plt.plot(amp, nrmse, '.')
plt.xlabel('amp (V)')
plt.ylabel('nrmse')
plt.title('Average First Pulse Fit (all connected pairs)')
plt.show()    



