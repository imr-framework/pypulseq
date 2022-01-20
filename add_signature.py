# This function appends the md5_hash signature to the .seq file to make it compatible with pulseq Version 1.4
# Calling the function "add_signature.add_md5_sig(file_name)" after the "seq.write(file_name)" function will append the required signature.

# Copyright (c) 2022, Niklas Wehkamp <niklas.wehkamp@uniklinik-freiburg.de>
# License CC BY-SA

import hashlib

def add_md5_sig(file_name):
    #missing! check if input file_name is valid
    #missing! check if input file already has a signature
    md5_hash = hashlib.md5()
    a_file = open(file_name, "rb")
    content = a_file.read()
    md5_hash.update(content)
    digest = md5_hash.hexdigest()

    explanation = "# This is the hash of the Pulseq file, calculated before the [SIGNATURE] section was added\n# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be sripped away for recalculating/verification)\nType md5\nHash "

    with open(file_name, "a") as a_file:
        a_file.write("\n")
        a_file.write('[SIGNATURE]\n')
        a_file.write(explanation)
        a_file.write(digest)
