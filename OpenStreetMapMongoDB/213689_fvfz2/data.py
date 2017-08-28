
# coding: utf-8

# builds the JSON file from the OSM data; parses, cleans, and shapes data: filename - data.py


import xml.etree.ElementTree as ET
import pprint
import re
import codecs
import json
import audit

lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
addresschars = re.compile(r'addr:(\w+)')
address_1 = re.compile(r'^addr:([a-z]|_)*$')
address_2 = re.compile(r'^addr:([a-z]|_)*:([a-z]|_)*$')
CREATED = [ "version", "changeset", "timestamp", "user", "uid"]

SAMPLE_FILE = "./honolulu_hawaii_sample.osm"

mapping = {'Ave'  : 'Avenue',  
                       'Blvd' : 'Boulevard',
                       'Dr'   : 'Drive',
                       'Ln'   : 'Lane',
                       'Pkwy' : 'Parkway',
                       'Rd'   : 'Road',
                       'St'   : 'Street'}

def shape_element(element):
    """
    Convert data from openstreetmap in format that can be accessed and used by MongodDB via JSON file
    """
    node = {}
    if element.tag == "node" or element.tag == "way" :
        node["type"] = element.tag
        for k, v in element.attrib.items(): # Attribute
            if k in CREATED:
                if "created" not in node:
                    node["created"] = {}
                node["created"][k] = v
            elif k == "lat":
                if "pos" not in node:
                    node["pos"] = [float(v), None]
                node["pos"][0] = float(v)
            elif k == "lon":
                if "pos" not in node:
                    node["pos"] = [None, float(v)]
                node["pos"][1] = float(v)
            else:
                node[k] = v
        for sub_elem in element.iter():     # Sub-tree
            if sub_elem.tag == "tag":
                key = sub_elem.attrib["k"]
                if re.search(problemchars, key):    #Skip problematic chars
                    continue
                if re.search(address_1, key):   # Start with addr, and only has one ":"
                    if "address" not in node:
                        node["address"] = {}
                    if key[5:] == "street":     # Update street
                        node["address"]["street"] = audit.update_street_name(sub_elem.attrib["v"], mapping)
                    else:
                        node["address"][key[5:]] = sub_elem.attrib["v"]
                elif re.search(address_2, key):
                    continue
                elif key == "phone":    # Convert telephone format
                    node["phone"] = audit.update_phone(sub_elem.attrib["v"])
                else:
                    node[key] = sub_elem.attrib["v"]
            elif sub_elem.tag == "nd":
                if "node_refs" not in node:
                    node["node_refs"] = []
                node["node_refs"].append(sub_elem.attrib["ref"])

        # print node
        return node
    else:
        return None

def process_map(file_in, pretty = False):
    """
    Process the osm file to json file to be prepared for input file to monggo
    """
    file_out = "{0}.json".format(file_in)
    data = []
    with codecs.open(file_out, "w") as fo:
        for _, element in ET.iterparse(file_in):
            el = shape_element(element)
            if el:
                data.append(el)
                if pretty:
                    fo.write(json.dumps(el, indent=2)+"\n")
                else:
                    fo.write(json.dumps(el) + "\n")
    return data

def test():

    data = process_map(SAMPLE_FILE)
    pprint.pprint(data[0:5])


if __name__ == "__main__":
#    test()
    data = process_map("/Users/sunilanagal/Documents/OpenStreetMap-Project/honolulu_hawaii_sample.osm")
    pprint.pprint(data[0:6])

# connect to Mongo Client
from pymongo import MongoClient
client  = MongoClient('mongodb://localhost:27017')
db = client.examples

# insert records into database db collection honolulu
[db.honolulu.insert_one(e) for e in data]

# Number of documents:
db.honolulu.count()

# Check 10 records
print "check 10 records from Honolulu collection"
pipeline = [
    {'$limit' : 10}
]
cursor = db.honolulu.aggregate(pipeline)
pprint.pprint(list(cursor))

# Number of nodes:

db.honolulu.find({'type':'node'}).count()

# Number of ways:
db.honolulu.find({'type':'way'}).count()

# Top 5 contributing users:
print "Top 5 contributing users in Honolulu collection"

pipeline = [
            {'$match': {'created.user':{'$exists':1}}},
            {'$group': {'_id':'$created.user',
                        'count':{'$sum':1}}},
            {'$sort': {'count':-1}},
            {'$limit' : 5}
]
cursor = db.honolulu.aggregate(pipeline)
pprint.pprint(list(cursor))

# Restaurants in Honolulu, Hawaii:
print "Restaurants in Honolulu, Hawaii"

pipeline = [
            {'$match': {'amenity':'restaurant',
                        'name':{'$exists':1}}},
            {'$project':{'_id':'$name',
                         'cuisine':'$cuisine',
                         'contact':'$phone'}},
            {'$limit': 10}
]
cursor = db.honolulu.aggregate(pipeline)
pprint.pprint(list(cursor))

# user with most contributions
print "user with most contributions in Honolulu, Hawaii openstreetmap"
pipeline = [{'$group': 
                 {'_id':'$created_user',
                        'count':{
                            '$sum':1
                            }
                 }
            },{
                '$sort':{
                    'count':-1
                }
            },{
                '$limit': 1
            }]

cursor = db.honolulu.aggregate(pipeline)
pprint.pprint(list(cursor))

# No. of Nodes without addresses
print "No. of Nodes without addresses in Honolulu, Hawaii openstreetmap"
pipeline = [{
             '$match': {
               'type': 'node',
                 'address': {
                   '$exists': 0
                }
             }
         }, {
            '$group': {
                 '_id': 'Nodes without addresses',
                 'count': {
                     '$sum': 1
                 }
            }
         }]
cursor = db.honolulu.aggregate(pipeline)
pprint.pprint(list(cursor))

# Most common building types/entries:
print "Most common building types/entries in Honolulu, Hawaii openstreetmap"
pipeline = [{
            '$match': {
                 'building': {
                     '$exists': 1
                 }
             }
        }, {
            '$group': {
                 '_id': '$building',
                'count': {
                     '$sum': 1
                 }
             }
         }, {
             '$sort': {
                 'count': -1
             }
         }, {
        '$limit': 10
        }]
cursor = db.honolulu.aggregate(pipeline)
pprint.pprint(list(cursor))