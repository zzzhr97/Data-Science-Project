import re
import os

def extract_file_result_1(filepath, pattern, **kwargs):
    result = []
    with open(filepath + '.txt', 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                if os.path.basename(filepath) == 'euclidean' and kwargs['only_distance'] == False:
                    match_keys = ['K', 'W', 'acc', 'train_time', 'test_time']
                    map_func = [int, str, float, float, float]
                else:
                    match_keys = ['K', 'acc', 'train_time', 'test_time']
                    map_func = [int, float, float, float]
                cur_result = {}
                for idx, key in enumerate(match_keys):
                    cur_result[key] = map_func[idx](match.group(idx+2))
                result.append(cur_result)
            else:
                continue
    return result

def extract_file_result_2(filepath):
    result = []
    for filename in os.listdir(filepath):
        if filename.endswith('.txt'):
            cur_result = {}

            pattern = r'lr(0\.\d+|1e-05|5e-05)_dim(\d+)_epoch(\d+)'
            match = re.search(pattern, filename)
            if match:
                match_keys = ['lr', 'dim', 'epoch']
                map_func = [float, int, int]

                try:
                    for idx, key in enumerate(match_keys):
                        cur_result[key] = map_func[idx](match.group(idx+1))
                except IndexError:
                    raise IndexError(f'Group not matched in {filename}.')

                with open(os.path.join(filepath, filename)) as f:
                    f.readline()
                    string = f.readline()
                    pattern = r'acc: ([\d.]+), train_time: ([\d.]+), test_time: ([\d.]+)'
                    match = re.search(pattern, string)
                    if match:
                        match_keys = ['acc', 'train_time', 'test_time']
                        map_func = [float, float, float]
                        for idx, key in enumerate(match_keys):
                            cur_result[key] = map_func[idx](match.group(idx+1))
                    else:
                        if string != '':
                            raise RuntimeError(f"Not match in file {os.path.join(filepath, filename)}.\nThe string is {string}.")
                
                result.append(cur_result)
    
    return result

def get_result(filename, **kwargs):
    """
    Args:
    - filename: should be in ['chebyshev', 'cosine', 'manhattan', 'seuclidean', 
            'euclidean', 'pml_circle', 'pml_contrastive', 'pml_triplet']
    """
    if filename in ['chebyshev', 'cosine', 'manhattan', 'seuclidean']:
        filepath = 'results/' + filename
        pattern = r'Test - .., MM - (\w+), K - (\d+), W - distance ------ acc - ([\d.]+), train_time - ([\d.]+) s, test_time - ([\d.]+) s'
        return extract_file_result_1(filepath, pattern)
    elif filename in ['euclidean']:
        filepath = 'results/' + filename
        if kwargs['only_distance'] == True:
            pattern = r'Test - .., MM - (\w+), K - (\d+), W - distance ------ acc - ([\d.]+), train_time - ([\d.]+) s, test_time - ([\d.]+) s'
        else:
            pattern = r'Test - .., MM - (\w+), K - (\d+), W - (\w+) ------ acc - ([\d.]+), train_time - ([\d.]+) s, test_time - ([\d.]+) s'
        return extract_file_result_1(filepath, pattern, only_distance=kwargs['only_distance'] )
    elif filename in ['pml_circle', 'pml_contrastive', 'pml_triplet']:
        filepath = 'results/' + filename + '/'
        result = extract_file_result_2(filepath)
        result = sorted(result, key=lambda x:(x['dim'], x['lr'], x['epoch']))
        return result
    else:
        raise ValueError("Filename not found.")
    
def t1():
    filename = 'euclidean'
    result = get_result(filename, only_distance=False)

    for i, s in enumerate(result):
        if i % 2 == 0:
            print(result[i]['K'], result[i]['acc'], result[i]['train_time'], result[i]['test_time'], sep=' & ', end=' & ')
            print(result[i+1]['acc'], result[i+1]['train_time'], result[i+1]['test_time'], sep=' & ', end='\\\n')

def t2():
    # euclidean, manhattan, cosine, chebyshev
    filename1 = 'chebyshev'
    filename2 = 'manhattan'
    result1 = get_result(filename1, only_distance=True)
    result2 = get_result(filename2, only_distance=True)

    for i, s in enumerate(result1):
        print(s['K'], s['acc'], s['train_time'], s['test_time'], sep=' & ', end=' & ')
        print(result2[i]['acc'], result2[i]['train_time'], result2[i]['test_time'], sep=' & ', end=' \\\\\n')

if __name__ == '__main__':
    t2()