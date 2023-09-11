import re
from typing import Tuple

def readasc(filename : str) -> Tuple[dict, dict]:
    """
    Reads Siemens ASC ascii-formatted textfile and returns a dictionary
    structure.
    E.g. a[0].b[2][3].c = "string"
    parses into:
      asc['a'][0]['b'][2][3]['c'] = "string"

    Parameters
    ----------
    filename : str
        Filename of the ASC file.

    Returns
    -------
    asc : dict
        Dictionary of ASC part of file.
    extra : dict
        Dictionary of other fields after "ASCCONV END"
    """
    
    asc, extra = {}, {}
    
    # Read asc file and convert it into a dictionary structure
    with open(filename, 'r') as fp:
        end_of_asc = False
        
        for next_line in fp:
            next_line = next_line.strip()
            
            if next_line == '### ASCCONV END ###': # find end of mrProt in the asc file
                end_of_asc = True
    
            if next_line == '' or next_line[0] == '#':
                continue

            # regex wizardry: Matches lines like 'a[0].b[2][3].c = "string" # comment'
            # Note this assumes correct formatting, e.g. does not check whether
            # brackets match.
            match = re.match('^\s*([a-zA-Z0-9\[\]\._]+)\s*\=\s*((\"[^\"]*\"|\\\'[^\\\']\\\')|(\d+)|([0-9\.e\-]+))\s*((#|\/\/)(.*))?$', next_line)
    
            if match:
                field_name = match[1]

                # Keep track of where to put the value: base[assign_to] = value
                if end_of_asc:
                    base = extra
                else:
                    base = asc

                assign_to = None
                
                # Iterate over every segment of the field name
                parts = field_name.split('.')
                for p in parts:
                    # Update base so final assignement is like: base[assign_to][p] = value
                    if assign_to != None and assign_to not in base:
                        base[assign_to] = {}
                    if assign_to != None:
                        base = base[assign_to]
                    
                    # Iterate over brackets
                    start = p.find('[')
                    if start != -1:
                        name = p[:start]
                        assign_to = name
                        
                        while start != -1:
                            stop = p.find(']', start)
                            index = int(p[start+1:stop])
                            
                            # Update base so final assignement is like: base[assign_to][p][index] = value
                            if assign_to not in base:
                                base[assign_to] = {}
                            base = base[assign_to]
                            assign_to = index
                            
                            start = p.find('[', stop)
                    else:
                        assign_to = p

                # Depending on which regex section matched we can infer the value type
                if match[3]:
                    base[assign_to] = match[3][1:-1]
                elif match[4]:
                    base[assign_to] = int(match[4])
                elif match[5]:
                    base[assign_to] = float(match[5])
                else:
                    raise RuntimeError('This should not be reached')
            elif next_line.find('=') != -1:
                raise RuntimeError(f'Bug: ASC line with an assignment was not parsed correctly: {next_line}')

    return asc, extra
