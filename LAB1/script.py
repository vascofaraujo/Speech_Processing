
# script to append specific string (i.e 1 or 0) to end of lines

file_name = "new_captions.rt"
string_to_add = ' 1'

with open(file_name, 'r') as f:
    file_lines = []
    for line in f.readlines():
        result = line
        if line.startswith('00:'):
            result = ''.join([line.strip(), string_to_add, '\n'])
        file_lines.append(result)

with open(file_name, 'w') as f:
    f.writelines(file_lines)