# Wrapper of preprocess.sh
import sys, os, subprocess
from subprocess import PIPE

PREFIX = os.path.dirname(__file__)
PREPRO = os.path.join(PREFIX, 'preprocess.sh')
SPM = {
    'en': os.path.join(PREFIX, './en_sp16k.model'),
    'ru': os.path.join(PREFIX, './ru_sp16k.model')
}

def prepro(lines, l):
    input_text = '\n'.join([line.strip() for line in lines])
    out = subprocess.run([PREPRO, l], input=input_text, stdout=PIPE, universal_newlines=True).stdout
    if out[-1] == '\n':
        out = out[:-1]
    return out.split('\n')


def spm_dec(lines, l):
    input_text = '\n'.join([line.strip() for line in lines])
    out = subprocess.run(['spm_decode', '--model', SPM[l]], input=input_text, stdout=PIPE, universal_newlines=True).stdout
    if out[-1] == '\n':
        out = out[:-1]
    return out.split('\n')


if __name__ == '__main__':
    out = prepro(sys.stdin.readlines(), sys.argv[1])
    for line in out:
        print(line)
