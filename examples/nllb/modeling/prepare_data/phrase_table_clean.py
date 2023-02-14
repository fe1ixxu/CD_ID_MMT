import argparse
import re
import random
class PhraseTable():
    def __init__(self, max_num_tgt=5):
        self.max_num_tgt = max_num_tgt

    def clean_table(self, old_table_file):
        fr = open(old_table_file)
        table = {}

        ## Read Phrase Table line by line
        line = fr.readline()
        while line:
            src, tgt, probs = line.split("|||")[:3]
            # Clean src and tgt:
            src = self.clean_key(src)
            tgt = self.clean_key(tgt)
            # Get the prob of the third one, which is P(tgt|src)
            prob = float(probs.strip().split()[2])
            self.feed_tgt(table, src, tgt, prob)
            line = fr.readline()
        fr.close()
        return table

    def clean_key(self, key):
        # Never need to clean mose tokenizer, which is cleaned during the preprocessing phase
        # Remove "▁&quot;"
        # key = key.replace("▁&quot;", '▁"').strip()
        # key = key.replace("&quot;", '"').strip()
        # # Remove "&apos;"
        # key = key.replace("▁&apos;", "▁'").strip()
        # key = key.replace("&apos;", "'").strip()
        # Remove extra spaces:
        key = key.strip()
        key = re.sub('\s+', ' ', key)
        return key

    def feed_tgt(self, table, src, tgt, prob):
        if src not in table:
            # if not in the table, start a new set
            table[src] = set([(tgt, prob)])
        else:
            # if num < max_num, append it
            table[src].add((tgt, prob))
            # if num >= max_num, discard the one with lowest prob # if num >= max_num, discard the one with lowest prob 
            if len(table[src]) > self.max_num_tgt:
                lowest_tgt = min(table[src], key=lambda x: x[1])
                # if the lowest prob is the same as the current one, randomly discard one, i.e. drop the lowest one
                hypos = [t for t in table[src] if t[1] == lowest_tgt[1]]
                table[src].remove(random.choice(hypos))

    def remove_probs(self, table):
        new_table = {}
        for src in table.keys():
            new_table[src] = set([tgt[0] for tgt in table[src]])
        return new_table

    def remove_short_pairs(self, table, filter_short=1):
        all_keys = list(table.keys())
        for src in all_keys:
            if len(src.split(" ")) <= filter_short:
                table.pop(src)

    def save_table(self, table, table_file):
        with open(table_file, "w") as f:
            for src in sorted(table.keys()):
                tgt = [t[0] for t in table[src]]
                tgt = "\t".join(tgt)
                f.writelines([src, "\t", tgt, "\n"])

    def load_table(self, table_file, with_prob=True):
        table = {}
        with open(table_file) as f:
            line =  f.readline()
            while line:
                src_tgt = line.split("\t")
                src, tgt = src_tgt[0], src_tgt[1:]
                
                if with_prob:
                    prob = 1/len(tgt)
                    table[src] = set([(t.strip(), prob) for t in tgt])
                else:
                    table[src] = set([t.strip() for t in tgt])
                line = f.readline()
        return table




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input phrase table file')
    parser.add_argument('--output', type=str, required=True, help='output phrase table file')
    parser.add_argument('--max_num', type=int, default=5, help='max tgt num phrases for each source phrase')
    args = parser.parse_args()
    PT = PhraseTable(args.max_num)
    table = PT.clean_table(args.input)
    PT.save_table(table, args.output)
