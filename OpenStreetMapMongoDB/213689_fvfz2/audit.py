
# coding: utf-8

# audits streets and phone nos also provides cleaning method: filename - audit.py

from collections import defaultdict
import re
import pprint
import phonenumbers
import xml.etree.cElementTree as ET

street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

SAMPLE_FILE = "./honolulu_hawaii_sample.osm"

expected_street_types = ["Avenue", "Boulevard", "Commons", "Court", "Drive", "Lane", "Parkway",  
                         "Place", "Road", "Square", "Street", "Trail"]

mapping = {'Ave'  : 'Avenue',  
                       'Blvd' : 'Boulevard',
                       'Dr'   : 'Drive',
                       'Ln'   : 'Lane',
                       'Pkwy' : 'Parkway',
                       'Rd'   : 'Road',
                       'St'   : 'Street'}

def audit_street_type(street_types, street_name, regex, expected_street_types):  
    """
    Search and match street names that are not in expected street type
    """
    m = regex.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected_street_types:
            street_types[street_type].add(street_name)              

def is_street_name(elem):  
    return (elem.attrib['k'] == "addr:street")

def is_state_name(elem):  
    return (elem.attrib['k'] == "addr:state")

def audit(osmfile, regex, limit=-1):  
    """
    Parse element tags for street names and return their street type
    """
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)

    # iteratively parse the mapping xml
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        # iterate 'tag' tags within 'node' and 'way' tags
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'], regex, expected_street_types)            
    return street_types

def update_street_name(name, mapping):
    """
    Convert street name by mapping so they follow uniform format.
    """
    m = street_type_re.search(name)
    street_type = m.group()
    if street_type not in expected_street_types: 
        if street_type in mapping.keys():  #not needed
            new_street_type = mapping[street_type]
            name = name.replace(street_type, new_street_type)
            return name
        
def update_phone(phone):
    """
    Convert all valid phone number into "(xxx) xxx-xxxx" format.
    """
    try:
        x = phonenumbers.parse(phone, "US")
        phone = phonenumbers.format_number(x, phonenumbers.PhoneNumberFormat.NATIONAL)
    except:
        pass
    return phone

# check the update_street_name and update_phone functions perform the way we expect
def test():
    street_types = audit(SAMPLE_FILE, street_type_re)

    for street_types, ways in street_types.iteritems():  
        if street_types in mapping:
            for name in ways:
                better_name = update_street_name(name, mapping)
                print name, "=>", better_name

    a = update_phone("899-090-9300")   
    print a

if __name__ == '__main__':
    test()

