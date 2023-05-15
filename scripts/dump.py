import pdb
import os
import sys

# SEP_INTERVAL = 4

parse_metric = lambda x: x.strip().split()[-1]
parse_pretrain_eps = lambda x: list(map(parse_metric, [x[6], x[3], x[4]]))
parse_pretrain_std = lambda x: list(map(parse_metric, [x[7], x[4], x[5]]))
pad_string = lambda x, l: [i.ljust(j) for i, j in zip(x, l)]

dump = []
root = f'{sys.argv[1]}'
column_size = [12, 60, 10, 10, 10]
column_name = ['slurm_id', 'exp_id', 'epoch', 'loss', 'best rms']

def get_file_name(path):
    f_list = os.listdir(path)
    files = []
    for i in f_list:
        if i[-4:] == '.out':
            files.append(i)
    return files

print(f'\ndumping results under {root}:\n')
print(' '.join(pad_string(column_name, column_size)))

file_lst = get_file_name(root)
for i, d in enumerate(sorted(file_lst)):
    path = os.path.join(root, d)
    slurm_id = path[8:-4]
    with open(path, 'r') as f:
        validate_lines = 0
        exp_id = None
        least_rms = 1.0
        for l in f.read().split('\n'):
            # deal with exp_id
            if l[:8] == '[exp_id]':
                exp_id = l[10:]
            # deal with Epoch Results
            elif l[:5] == 'silog':
                validate_lines += 1
                rms = l.split(' ')[10]
                if float(rms) < least_rms:
                    # pdb.set_trace()
                    least_rms = float(rms)


    if exp_id is None or validate_lines == 0:
        continue
    else:
        epoch = 'Test'
        loss = 'Test'
        
        results = [slurm_id, exp_id, epoch, loss, str(least_rms)]
    # print results
    print(' '.join(pad_string(results, column_size)))