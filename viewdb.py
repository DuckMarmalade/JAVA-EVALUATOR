import pickle
with open('java_analysis_memory.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)