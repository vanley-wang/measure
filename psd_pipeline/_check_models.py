import pickle, os
base = r'D:\Desktop\music\measure\psd_pipeline'
for name, fname in [('model_nnunet','nnunet_delta_deploy.pkl'), ('model_nnunet_new','nnunet_delta_deploy.pkl'), ('model','paper_delta_deploy.pkl')]:
    path = os.path.join(base, name, fname)
    m = pickle.load(open(path, 'rb'))
    print(f'=== {name}/{fname} ===')
    print(f'  Keys: {list(m.keys())}')
    print(f'  Features: {m.get("features","N/A")}')
    print(f'  Training r: {m.get("training_r", m.get("r","N/A"))}')
    print()