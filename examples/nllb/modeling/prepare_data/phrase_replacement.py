import argparse
import re
import random
from phrase_table_clean import PhraseTable
import os
from collections import defaultdict
import matplotlib.pyplot as plt

class Monitor():
    def __init__(self, table):
        self.num_all_sen = 0
        self.num_all_sen_changed = 0

        self.num_long_sen = 0
        self.num_long_sen_changed = 0

        self.num_short_sen = 0
        self.num_short_sen_changed = 0

        self.long_threshold = 20

        self.phrase_monitor = {k: [len(k.split(" ")), 0] for k in table.keys()}  #phrase: number of time used
        self.num_rephrased_per_sentence = defaultdict(int)
        self.avg_len_phrase_replaced = [0, 0]
        self.avg_sen_len_rephrased = [0, 0]

        self.num_length_phrase_stats = defaultdict(int)
        for k in table.keys():
            self.num_length_phrase_stats[len(k.split(" "))] += 1 

        

    def record(self, context, recorder):
        ## Record num of sentences
        self.num_all_sen += 1
        if len(context) > self.long_threshold:
            self.num_long_sen += 1
        else:
            self.num_short_sen += 1
        
        if len(recorder) > 0:
            ## Record num of sentences changed
            self.num_all_sen_changed += 1
            if len(context) > self.long_threshold:
                self.num_long_sen_changed += 1
            else:
                self.num_short_sen_changed += 1

            self.avg_sen_len_rephrased[0] = \
                (self.avg_sen_len_rephrased[0]*self.avg_sen_len_rephrased[1]+len(context)) / (self.avg_sen_len_rephrased[1] +1)
            self.avg_sen_len_rephrased[1] += 1

            ## Record num of phrase used
            subtotal = 0
            for _, _, phrase, _ in recorder:
                self.phrase_monitor[phrase][1] += 1
                subtotal += len(phrase.split())

            ## How long are these rephrased?
            self.avg_len_phrase_replaced[0] = \
                (self.avg_len_phrase_replaced[0]* self.avg_len_phrase_replaced[1] + subtotal)\
                /(self.avg_len_phrase_replaced[1]+1)
            self.avg_len_phrase_replaced[1] += 1

            ## How many times is each sentence been rephrased?
            self.num_rephrased_per_sentence[len(recorder)] += 1
            
    def get_statistic_results(self, out_file="tmp.log"):
        with open(out_file, "a") as f:
            print(f"Number of all sentence: {self.num_all_sen}", file=f)
            print(f"Number of long sentence: {self.num_long_sen}", file=f)
            print(f"Number of short sentence: {self.num_short_sen}", file=f)
            print(f"Ratio of all sentences been rephrased: {self.num_all_sen_changed / self.num_all_sen}", file=f)
            print(f"Ratio of long sentences been rephrased: {self.num_long_sen_changed / self.num_long_sen}", file=f)
            print(f"Ratio of short sentences been rephrased: {self.num_short_sen_changed / self.num_short_sen}", file=f)
            
            print(f"Statistics of how many times each sentence is rephrased: {self.num_rephrased_per_sentence}", file=f)
            print(f"Avg length of phrases are used: {self.avg_len_phrase_replaced}", file=f)
            print(f"Avg length of sentences are rephrased: {self.avg_sen_len_rephrased}", file=f)

            print(f"phrase length vs. Number of phrases {self.num_length_phrase_stats}", file=f)
            num_length_used = defaultdict(int)
            for k, v in self.phrase_monitor.items():
                num_length_used[len(k.split(" "))] += v[1]
            print(f"phrase_length vs. time of phrases used {num_length_used}", file=f)

            phrases = sorted(self.phrase_monitor, key=lambda x: self.phrase_monitor[x][1])
            plt.plot(range(len(phrases)), [self.phrase_monitor[p][1] for p in phrases])
            plt.show()
            plt.savefig(out_file+".png")
            print(f"Most 50 popular phrases used: {[(p, self.phrase_monitor[p][1]) for p in phrases[-50:]]}", file=f)
            print(f"Least 100 popular phrases used: {[(p, self.phrase_monitor[p][1]) for p in phrases[:50]]}", file=f)
            print(f"Number of phrases that never been used for replacement: {sum([1 for p in phrases if self.phrase_monitor[p][1]==0])}; Total number of phrases {len(phrases)}", file=f)


def filter_recorder(ratio, recorder, context, replace_prob):
    new_recorder = []
    cur_len = 0
    recorder.sort(key=lambda x: len(x[2].split(" ")), reverse=True)
    for r in recorder:
        if cur_len/len(context) <= ratio:
            if random.random() < replace_prob:
                r[3] = " ".join(["<mask>"]*len(r[3].split(" "))) 
            new_recorder.append(r)
        else:
            break
        src = r[2]
        cur_len += len(src.split(" "))
    return new_recorder


def phrase_replacement_step(context, table, max_phrase_len, max_replace_ratio, monitor, replace_prob):
    context = context.split(" ")
    
    # record replaced info: [start_ind, end_ind, src, tgt]
    recorder = []
    def check_not_overlap_phrase(ind, window):
        for r in recorder:
            if r[0] <= ind < r[1] or r[0] < ind + window < r[1]:
                return False
        return True
    ## Slide window from 1 to max_phrase_len, reversed order to prioritize longest phrase
    for window in range(max_phrase_len, 0, -1):
        # skip if window > context length
        if window < len(context): 
            for ind in range(len(context) + 1 - window):
                src = " ".join(context[ind:ind+window])
                if src in table:
                    if check_not_overlap_phrase(ind, window):
                        recorder.append([ind, ind+window, src, random.sample(table[src],1)[0]])
                elif src.lower() in table:
                    if check_not_overlap_phrase(ind, window):
                        recorder.append([ind, ind+window, src.lower(), random.sample(table[src.lower()],1)[0]])
                
    
    recorder = filter_recorder(max_replace_ratio, recorder, context, replace_prob)
    monitor.record(context, recorder)
    recorder.sort(key=lambda x: x[0])

    prev_id = 0
    new_context = []
    for start_id, end_ind, src, tgt in recorder:
        new_context += context[prev_id:start_id] + [tgt]
        prev_id = end_ind
    new_context += context[prev_id:]

    return " ".join(new_context)


def get_max_phrase_len(table):
    return max([len(src.split(" ")) for src in table.keys()])

def phrase_replacement(input_file, output_file, table, monitor, max_replace_ratio=0.2, replace_prob=0.30):
    rewrite_original_input_file = False
    if output_file== input_file:
        output_file = input_file + ".tmp"
        rewrite_original_input_file = True

    max_phrase_len = get_max_phrase_len(table)
    with open(input_file) as fr:
        with open(output_file, "w") as fw:
            context = fr.readline()
            while context:
                if monitor.num_all_sen % 10000 == 0:
                    print(f"{monitor.num_all_sen} now!")
                context = context.strip()
                ## do phrase_replacement:
                context = phrase_replacement_step(context, table, max_phrase_len, max_replace_ratio, monitor, replace_prob)
                fw.writelines([context, "\n"])
                context = fr.readline()
    
    if rewrite_original_input_file:
        # remove input file and rename output file
        os.remove(input_file)
        os.rename(output_file, input_file)

def masking_replacement_step(context, ratio, special_symbol):
    context = context.split(" ")
    for i in range(len(context)):
        if random.random() < ratio:
            context[i] = special_symbol
    return " ".join(context)


def masking_replacement(input_file, output_file, ratio=0.2, special_symbol="<mask>"):
    rewrite_original_input_file = False
    if output_file == input_file:
        output_file = input_file + ".tmp"
        rewrite_original_input_file = True

    count = 0
    with open(input_file) as fr:
        with open(output_file, "w") as fw:
            context = fr.readline()
            while context:
                if count % 10000 == 0:
                    print(f"{count} now!")
                context = context.strip()
                ## do phrase_replacement:
                context = masking_replacement_step(context, ratio, special_symbol)
                fw.writelines([context, "\n"])
                context = fr.readline()
                count += 1
    
    if rewrite_original_input_file:
        # remove input file and rename output file
        os.remove(input_file)
        os.rename(output_file, input_file)



def main(args):
    pt = PhraseTable()
    table = pt.load_table(args.table_file, with_prob=False)
    monitor = Monitor(table)
    random.seed(42)
    phrase_replacement(args.input, args.output, table, monitor, args.max_replace_ratio, args.replace_prob)
    monitor.get_statistic_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input phrase table file')
    parser.add_argument('--output', type=str, required=True, help='output phrase table file')
    parser.add_argument('--table_file', type=str, required=True, help='table file')
    parser.add_argument('--replace_prob', type=float, default=0.3, help='prob to be masked (<mask>)')
    parser.add_argument('--max_replace_ratio', type=float, default=0.2, help='max ratio of rephrased sentences length to the original length')
    args = parser.parse_args()
    main(args)