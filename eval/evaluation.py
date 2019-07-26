from google.cloud import storage
import os, errno, re
from collections import defaultdict
import numpy as np


def download_blob(tmp_dir):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('sventestbucket')
    test_filename = 'test_output.txt'
    test_users_filename = 'test_users.txt'
    test_filepath = os.path.join(tmp_dir,test_filename)
    test_users_filepath = os.path.join(tmp_dir,test_users_filename)
    blob = bucket.blob(os.path.join('test_results/',test_filename))
    blob2 = bucket.blob(os.path.join(test_users_filename))
    if not os.path.exists(test_filepath):
        blob.download_to_filename(test_filepath)
    if not os.path.exists(test_users_filepath):
        blob2.download_to_filename(test_users_filepath)

def _test_output_filename(tmp_dir):
    return os.path.join(tmp_dir,'test_output.txt')

def _test_users_filename(tmp_dir):
    return os.path.join(tmp_dir,'test_users.txt')


def average_precision(rankedpeople, goldconditions, condPOS, condNEG):
    '''
    Compute the average precision for a ranked list of people,
    where their true conditions are given in goldconditions.
    :param rankedpeople: list of strings
    :param goldconditions: dict of {string: string} mapping people to conditions
    :param condPOS: string; condition to use as positive class
    :param condNEG: string; condition to use as negative class
    '''
    ap = 0.0

    def calc_precision(rankedpeople, goldconditions, k, condPOS):
        positives = 0
        for i in range(k):
            condition = goldconditions[rankedpeople[i]]
            if condition == condPOS:
                positives += 1
        precision = positives / k
        return precision

    ## Solution start
    n = len(rankedpeople)
    relevantPeople = 0
    for k in range(n):
        condition = goldconditions[rankedpeople[k]]
        if condition == condPOS:
            precision = calc_precision(rankedpeople, goldconditions, k + 1, condPOS)
            ap += precision
            relevantPeople += 1
        else:
            ap += 0

    ap = ap / relevantPeople
    ## Solution end

    return ap

if __name__ == '__main__':
    pattern=r'(control|depression)'
    try:
        os.mkdir('/tmp/t2t')
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    tmp_dir='/tmp/t2t'
    download_blob(tmp_dir)
    test_output_filename = _test_output_filename(tmp_dir)
    test_users_filename = _test_users_filename(tmp_dir)

    user_post_classes = defaultdict(list)
    user_class = {}
    user_score = {}
    output = open(test_output_filename, 'r').readlines()
    users = open(test_users_filename, 'r').readlines()
    current_user = ''
    usercndn = False
    for i,line in enumerate(users):
        m = re.search(pattern, line)
        cndn = m.group(1)
        m = re.sub(pattern, '', line)
        userid = m
        post_class=output[i]
        user_class[userid.split()[0]] = cndn
        user_post_classes[userid.split()[0]].append(post_class.split()[0])
    for usrid in user_class:
        count = 1
        windowscores = []
        d_posts = 0
        c_posts = 0
        for postclass in user_post_classes[usrid]:
            if postclass == 'Depression':
                d_posts += 1
            else:
                c_posts += 1
            if count == 15:
                if c_posts == 0:
                    windowscore = d_posts
                else:
                    windowscore = d_posts/c_posts
                windowscores.append(windowscore)
                d_posts = 0
                c_posts = 0
                count = 0
            count += 1

        user_score[usrid] = np.median(windowscores)
    rankings = []
    with open('gold.txt', 'w') as fout:
        for usrid, label in user_class.items():
            fout.write(usrid +' '+ label + '\n')
    with open('rankings.txt', 'w') as fout:
        sorted_x = sorted(user_score.items(), key=lambda kv: kv[1], reverse=True)
        for tuple in sorted_x:
            fout.write(str(tuple[1])+'\n')
            rankings.append(tuple[0])

    ap = average_precision(rankings, user_class, 'depression', 'control')
    print(ap)


