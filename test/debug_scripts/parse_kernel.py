import csv


def get_db():
    filename = 'dlprof_kernel.csv'
    db = dict()
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        total_time = 0
        headers = list()
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            elif line_count == 1:
                headers = row
                line_count += 1
            else:
                db[row[0]] = [row[3], float(row[4]), float(row[5])]
                total_time = total_time + float(row[4])
                line_count += 1
    return db, total_time


def sort_db(db, total_time):
    sorted_db = sorted(db.items(), key=lambda x: x[1][1], reverse=True)
    acc_perc = 0
    for record in sorted_db:
        perc = record[1][1]/total_time*100
        acc_perc += perc
        # perc = round(perc, 2)
        # acc_perc = round(acc_perc, 2)
        record[1].append(perc)
        record[1].append(acc_perc)
    return sorted_db


def pretty_print_sorted_db(sorted_db):
    doc = str()

    columns = ["Name", "Uses TC", "Total time (ns)", "Avg time (ns)", "Percentage", "Acc Percentage",]
    doc += '\t'.join(columns)
    doc += '\n'
    for record in sorted_db:
        entries = list()
        entries.append(record[0])
        entries += record[1]
        doc += '\t'.join(map(lambda x : str(x), entries))
        doc += '\n'
    with open("sorted_db.csv", 'w') as fw:
        fw.write(doc)


def calculate_mem_vs_compute(sorted_db):
    compute_perc = 0
    for record in sorted_db:
        name = record[0]
        perc = record[1][3]
        if 'volta' in name or 'gemm' in name:
            compute_perc += perc
    print("Compute perc = ", compute_perc, " and Memory perc = ", 100 - compute_perc)

if __name__ == '__main__':
    # Get the database
    db, total_time = get_db()

    # Sort the kernels
    sorted_db = sort_db(db, total_time)
    pretty_print_sorted_db(sorted_db)

    # Total time
    print("Total time = ", total_time)

    # Compute vs Memory bound
    calculate_mem_vs_compute(sorted_db)
