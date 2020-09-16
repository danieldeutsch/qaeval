import csv
import json
import sys


def main():
    with open(sys.argv[2], 'w') as out:
        with open(sys.argv[1], 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue

                question = row[2]
                answer = row[3]
                context = row[4]

                try:
                    answer_start = context.lower().index(answer.lower())
                    answer_end = answer_start + len(answer)

                    out.write(json.dumps({
                        'context': context,
                        'answer': answer,
                        'answer_start': answer_start,
                        'answer_end': answer_end,
                        'question': question
                    }) + '\n')

                except ValueError:
                    pass


if __name__ == '__main__':
    main()