import pickle
def loader(picklename):
    with open(picklename,'rb') as f:
        a = pickle.load(f)
    return(a)
