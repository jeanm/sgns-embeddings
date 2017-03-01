from collections import namedtuple

Phrases = namedtuple("Phrases", ["an", "vs", "vo", "dn", "nn", "vps", "vpo", "nvs", "nvo"])

# returns several lists of (head, dep) corresponding to phrases
def find_phrases(tags, grs, tokens):
    length = len(tokens)
    phrases = Phrases(an=[], vs=[], vo=[], dn=[], nn=[], vps=[], vpo=[], nvs=[], nvo=[])
    negated_verbs = {}
    verbs_with_particles = {}
    for gr in grs:
        head, dep = gr[1]-1, gr[2]-1
        # decrement indices since 0 represents ROOT and everything else is 1-indexed
        if gr[1] == 0 or gr[2] == 0:
            continue
        if gr[0] == "det":
            if tags[head].startswith("NN") and tags[dep] == "DT":
                phrases.dn.append((dep, head))
        if gr[0] == "amod":
            if tags[head].startswith("NN") and tags[dep].startswith("JJ"):
                phrases.an.append((dep, head))
        if gr[0] == "nsubj":
            # NB – hack: CoreNLP messes up with "to let go of".
            # We choose to treat "let go" as a single unit in this case.
            if ((tokens[head] == "let" or tokens[head] == "lets" or tokens[head] == "letting") and
                    head+2 < length and tokens[head+1] == "go" and tokens[head+2] == "of"):
                phrases.vps.append((dep, head, head+1, head+2))
            if tags[head].startswith("VB"):
                if tags[dep].startswith("NN") or tags[dep].startswith("PRP"):
                    phrases.vs.append((dep, head))
        if gr[0] == "dobj":
            if tags[head].startswith("VB"):
                if tags[dep].startswith("NN") or tags[dep].startswith("PRP"):
                    phrases.vo.append((head, dep))
        if gr[0] == "nmod:of":
            # NB – hack: CoreNLP messes up with "to let go of".
            # We choose to treat "let go" as a single unit in this case.
            if (tokens[head] == "go" and head+1 < length and tokens[head+1] == "of" and
                    head-1 >= 0 and (tokens[head-1] == "let" or tokens[head-1] == "lets" or tokens[head-1] == "letting")):
                phrases.vpo.append((head-1, head, head+1, dep))
        if gr[0] == "neg":
            if tags[head].startswith("VB"):
                if tags[dep].startswith("RB"):
                    negated_verbs[head] = dep  # needs a second pass
        if gr[0] == "compound:prt":
            if tags[head].startswith("VB"):
                if tags[dep].startswith("RP"):
                    verbs_with_particles[head] = dep  # needs a second pass
        if gr[0] == "compound":
            if tags[head].startswith("NN"):
                if tags[dep].startswith("NN"):
                    # sort them as they appear in the corpus
                    if head > dep:
                        phrases.nn.append((dep, head))
                    else:
                        phrases.nn.append((head, dep))

    # second pass to process negated verbs and verbs with particles
    for gr in grs:
        head, dep = gr[1]-1, gr[2]-1
        # decrement indices since 0 represents ROOT and everything else is 1-indexed
        if gr[1] == 0 or gr[2] == 0:
            continue
        if gr[0] == "nsubj" and (tags[dep].startswith("NN") or tags[dep].startswith("PRP")):
            if head in verbs_with_particles:
                phrases.vps.append((dep, head, verbs_with_particles[head]))
            if head in negated_verbs:
                phrases.nvs.append((dep, negated_verbs[head], head))
        if gr[0] == "dobj" and (tags[dep].startswith("NN") or tags[dep].startswith("PRP")):
            if head in verbs_with_particles:
                phrases.vpo.append((head, verbs_with_particles[head], dep))
            if head in negated_verbs:
                phrases.nvo.append((negated_verbs[head], head, dep))

    # NB – hack: CoreNLP never identifies "soda can" as a compound.
    # The following approach isn't ideal, but a quick `grep` of wikipedia shows
    # that most occurrences of "soda can" have indeed the expected meaning.
    for pos, token in enumerate(tokens):
        if token == "soda" and pos+1 < length and (tokens[pos+1] == "can" or tokens[pos+1] == "cans"):
            phrases.nn.append((pos, pos+1))

    return phrases

replacements = {
  "-lrb-": "(",
  "-rrb-": ")",
  "-lsb-": "[",
  "-rsb-": "]",
  "-lcb-": "{",
  "-rcb-": "}",
  "``": "\"",
  "“": "\"",
  "''": "\"",
  "”": "\"",
  "`": "'",
  "‘": "'",
  "’": "'",
  "---": "--",
  "–": "--",
  "—": "--",
  "\\/": "/",
  "\*": "*",
  "\n": "",
  "\t": "",
  "\r": ""
}

def cleanup_token(token):
    token = token.lower()
    if token in replacements:
        return replacements[token]
    else:
        return token
