import re
import os
import glob
import subprocess
import pdb
from subprocess import Popen, PIPE, STDOUT

valid_list = []
invalid_list = []
bug_list = []

bug_file = open("bugs/bugs.txt",'w+')
invalid_file = open("invalid/invalids.txt",'w+')
valid_file = open("valid/valids.txt", 'w+')

def check_with_gcc(program):
    for version in ["-5", "-6", "-7", "-8"]:
        for opt_level in ["", "-00 ", "-01 ", "-02 ", "-03 "]:
            command = "gcc" + version + " -w " + opt_level + program
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)

            p_out = str(p.stdout.read())

            if "error" in p_out:
                # result = "error"
                if "internal compiler" in p_out:
                    if p_out not in bug_list:
                        bug_list.append(program)
                        bug_file.write(program)
                        bug_file.write('\n')
                else:
                    if p_out not in invalid_list:
                        invalid_list.append(program)
                        invalid_file.write(program)
                        invalid_file.write('\n')
            else:
                if p_out not in valid_list:
                    valid_list.append(program)
                    valid_file.write(program)
                    valid_file.write('\n')

def check_with_clang(program):
    for version in ["-3.8", "-4.0"]:
        for opt_level in ["", "-00 ", "-01 ", "-02 ", "-03 "]:
            command = "clang" + version + " -w " + opt_level + program
            p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True)

            p_out = str(p.stdout.read())

            if "error" in p_out:
                # result = "error"
                if "internal compiler" in p_out:
                    if p_out not in bug_list:
                        bug_list.append(program)
                        bug_file.write(program)
                        bug_file.write('\n')
                else:
                    if p_out not in invalid_list:
                        invalid_list.append(program)
                        invalid_file.write(program)
                        invalid_file.write('\n')
            else:
                if p_out not in valid_list:
                    valid_list.append(program)
                    valid_file.write(program)
                    valid_file.write('\n')

def check_code(write_to_file, code_file):

    with open(write_to_file,'w') as log_file:
        with open(code_file, 'r') as c_file:
            for line_code in c_file:
                check_with_gcc(line_code)
        write_to_file.write("Number of Valid programs: " + str(len(valid_list)))
        write_to_file.write("Number of Invalid programs: " + str(len(invalid_list)))
        write_to_file.write("Number of Bugs: " + str(len(bug_list)))


if __name__ == '__main__':
    check_code('save/experiment-log','save/output_batch_80')



