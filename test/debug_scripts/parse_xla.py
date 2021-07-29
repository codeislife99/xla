import csv
import re


def parse_time(text):
    if 'us' not in text:
        return 0
    array = re.findall(r'\d*\.\d+|\d+', text)
    parsed_time = 0
    counter = 0
    # Calculate in us

    assert len(array) <= 4
    if len(array) == 4:
        array[1] =float(array[1]) + float(array[0]) * 60
    array = array[1:]
    for value in reversed(array):
        parsed_time += pow(1000, counter) * float(value)
        counter += 1
    # Go to ms
    parsed_time /= 1000
    return parsed_time

def get_db():
    filename = 'xla_debug.txt'
    db = dict()
    with open(filename) as fh:
        lines = fh.readlines()
        headers = list()
        line_number = 0
        while line_number != len(lines):
            line = lines[line_number]
            line = line.rstrip()
            if "Metric" in line:
                name = line.rstrip().split(" ")[-1]
                samples = float(lines[line_number + 1].rstrip().split(" ")[-1])
                acc_time = parse_time(lines[line_number + 2].rstrip().split(" ")[-1])
                line_number += 6
                db[name] = [samples, acc_time]
            else:
                line_number += 1
    return db


def add_perc(db):
    total_time = 0
    acc_perc = 0
    for entry in db:
        total_time += entry[1][1]
    for entry in db:
        this_time = entry[1][1]
        perc = this_time/total_time * 100
        acc_perc += perc
        entry[1].append(perc)
        entry[1].append(acc_perc)
    print("Total time =", total_time)


def sort_db(db):
    sorted_db = sorted(db.items(), key=lambda x: x[1][1], reverse=True)
    xla_db = list(filter(lambda x: "Xrt" not in x[0], sorted_db))
    xrt_db = list(filter(lambda x: "Xrt" in x[0], sorted_db))
    add_perc(xla_db)
    add_perc(xrt_db)
    return xla_db, xrt_db


def pretty_print_sorted_db(sorted_db, name):
    doc = str()

    columns = ["Name", "Num_calls", "Total time (ms)", "Percentage", "Acc Percentage",]
    doc += '\t'.join(columns)
    doc += '\n'
    for record in sorted_db:
        entries = list()
        entries.append(record[0])
        entries += record[1]
        doc += '\t'.join(map(lambda x : str(x), entries))
        doc += '\n'
    with open(name, 'w') as fw:
        fw.write(doc)



if __name__ == '__main__':
    db = get_db()
    xla_db, xrt_db = sort_db(db)
    pretty_print_sorted_db(xla_db, "xla_sorted_db.csv")
    pretty_print_sorted_db(xrt_db, "xrt_sorted_db.csv")

