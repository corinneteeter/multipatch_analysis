from multipatch_analysis.database import database as db
import sys

def pair_query(session):
    """
    Build a query to get values in pair table
    """
    q = session.query(
#        db.Baseline.id.label('response_id'),
#        db.Baseline.data,
#        db.PatchClampRecording.clamp_mode,
        db.Pair.id,
        db.Pair.distance,
    )
#    q = q.join(db.Baseline.recording).join(db.PatchClampRecording) 
#
#    # Ignore anything that failed QC
#    q = q.filter(db.or_(db.Baseline.ex_qc_pass==True, db.Baseline.in_qc_pass==True))

    return q

@db.default_session
def _compute_first_pulse_features(inds, session=None):
    '''Compute the features of the first pulse of a connection, put them in a table and write them to the database.
    #TODO: check this comment is correct
    input: 
        inds:tuple
            first value: string usually specifying the type of data being used
            second value: integer specifying id (in database ?) in which to start
            third value:integer specifying id (in database ?) in which to stop
    '''     
    source, start_id, stop_id = inds
    #1. do queries to get what you need from database for calculations
    q = pair_query(session)
    
    #2. do the computation
    
#    for ii in int(start_id, stop_id):
    q1=q.filter(db.Pair.id>=start_id).filter(db.Pair.id<=start_id+5)
    recs=q1.all()

    new_recs=[]
    for rec in recs:
        print 'index id', rec.id
        new_rec={'%s_id'%source: rec.id}
        #use belowish when you move to real functions
        #for table_columns in first_pulse_features_tables.schemas['first_pulse_features'])[1:len(first_pulse_features_tables.schemas['first_pulse_features'])] #note this assumes 0 is the id
        new_rec['test_number']=rec.id+100000
        new_recs.append(new_rec)
    
    #I think the first should be entries specified by the schema.  Second part is a list of dictionaries
    session.bulk_insert_mappings(FirstPulseFeatures, new_recs)
    
    # commit it to the database
    session.commit()

@db.default_session
def rebuild_first_pulse_features(parallel=True, workers=6, session=None):
    '''Specifies whether to run single or multithreaded and then calls function
    '''
    for source in ['baseline', 'pulse_response']: #TODO: not sure what these should be yet
        print("Rebuilding %s strength table.." % source)
        
        # Divide workload into equal parts for each worker
        max_pulse_id = session.execute('select max(id) from %s' % source).fetchone()[0]
        chunk = 1 + (max_pulse_id // workers)
        parts = [(source, chunk*i, chunk*(i+1)) for i in range(4)]

        if parallel:
            pool = multiprocessing.Pool(processes=workers)
            pool.map(compute_strength, parts)
        else:
            for part in parts:
                compute_first_pulse_features(part)
                
def compute_first_pulse_features(inds, session=None):
    # Thin wrapper just to allow calling from pool.map
    return _compute_first_pulse_features(inds, session=session)

class TableGroup(object):
    def __init__(self):
        self.mappings = {}
        self.create_mappings()

    def __getitem__(self, item):
        return self.mappings[item]

    def create_mappings(self):
        for k,schema in self.schemas.items():
            self.mappings[k] = db.generate_mapping(k, schema)

    def drop_tables(self):
        for k in self.schemas:
            if k in db.engine.table_names():
                self[k].__table__.drop(bind=db.engine)

    def create_tables(self):
        for k in self.schemas:
            if k not in db.engine.table_names():
                self[k].__table__.create(bind=db.engine)


class FirstPulseFeaturesTableGroup(TableGroup):
    # these define the table columns and the types of data within the columns
    schemas={
        'first_pulse_features':[
                    ('id', 'int', '', {'index': True}),
                    ('test_number', 'int'),
        ],
    }
    
    def create_mappings(self):
        TableGroup.create_mappings(self)
        
        FirstPulseFeatures = self['first_pulse_features']
        
        #note these are just needed for relational mappings in the analysis
        db.Pair.first_pulse_features = db.relationship(FirstPulseFeatures, back_populates="pair", cascade="delete", single_parent=True)
        FirstPulseFeatures.pair = db.relationship(db.Pair, back_populates="first_pulse_features", single_parent=True)

first_pulse_features_tables=FirstPulseFeaturesTableGroup()
def init_tables():
    '''create #### and make then accessible to the rest of the code.
    '''
    global FirstPulseFeatures
    first_pulse_features_tables.create_tables()

    FirstPulseFeatures=first_pulse_features_tables['first_pulse_features']
    

   
if __name__ == '__main__':
   
#    first_pulse_features_tables.drop_tables()
#    init_tables()
    rebuild_first_pulse_features(parallel='--local' not in sys.argv)
    

