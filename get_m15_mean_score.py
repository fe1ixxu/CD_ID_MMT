import argparse

# HIGH = "ind,msa,dan,isl,slv,tgl,deu".split(",")
# LOW = "nso,run,ben,nob,cat,glg".split(",")
# VERY_LOW= "ssw,asm,awa,fao,fur,yid,lim".split(",")
# ALL = 'nso,run,ssw,ben,asm,awa,ind,msa,dan,isl,nob,fao,slv,tgl,cat,glg,fur,deu,yid,lim'.split(",")
HIGH = "ind,msa,isl,slv,tgl".split(",")
LOW = "nso,run,nob,cat,glg".split(",")
VERY_LOW= "ssw,fao,fur,ltz,lim".split(",")
ALL = 'nso,run,ssw,ind,msa,isl,nob,fao,slv,tgl,cat,glg,fur,ltz,lim'.split(",")


def get_m20_mean(args):
    all_bleu_scores = []
    all_chrf_scores = []

    high_bleu_scores = []
    high_chrf_scores = []

    low_bleu_scores = []
    low_chrf_scores = []

    very_low_bleu_scores = []
    very_low_chrf_scores = []

    
    for lang in ALL:
        if args.engtgt:
            src, tgt = lang, "eng"
            bleu_file = f"/predict.{tgt}-{src}.{tgt}.bleu"
            chrf_file = f"/predict.{tgt}-{src}.{tgt}.chrf"
        else:
            src, tgt = "eng", lang
            bleu_file = f"/predict.{src}-{tgt}.{tgt}.bleu"
            chrf_file = f"/predict.{src}-{tgt}.{tgt}.chrf"
        ## BLEU
        try:
            with open(args.input + bleu_file) as f:
                l = f.readlines()
            bleu = float(l[0].split(" ")[2])
        except:
            continue
        ## CHRF
        try:
            with open(args.input + chrf_file) as f:
                l = f.readlines()
            chrf = float(l[2].split(":")[-1].split(",")[0].strip())
        except:
            continue

        all_bleu_scores.append(bleu)
        all_chrf_scores.append(chrf)
        if lang in LOW:
            low_bleu_scores.append(bleu)
            low_chrf_scores.append(chrf)
        if lang in VERY_LOW:
            very_low_bleu_scores.append(bleu)
            very_low_chrf_scores.append(chrf)
        elif lang in HIGH:
            high_bleu_scores.append(bleu)
            high_chrf_scores.append(chrf)

    def get_mean(scores):
        return round(sum(scores)/len(scores) ,2)
    print(f"Avg of all BLEU and chrf are {get_mean(all_bleu_scores)}, {get_mean(all_chrf_scores)}")
    print(f"Avg of high resource BLEU and chrf are {get_mean(high_bleu_scores)}, {get_mean(high_chrf_scores)}")
    print(f"Avg of low resource BLEU and chrf are {get_mean(low_bleu_scores)}, {get_mean(low_chrf_scores)}")
    print(f"Avg of very low resource BLEU and chrf are {get_mean(very_low_bleu_scores)}, {get_mean(very_low_chrf_scores)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input folder')
    parser.add_argument('--engtgt', type=int, default=0, help='eng is tgt')
    args = parser.parse_args()
    get_m20_mean(args)

