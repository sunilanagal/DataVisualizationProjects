
# coding: utf-8

# contains MongoDB queries:filename - queries.py

# connect to Mongo Client
from pymongo import MongoClient
client  = MongoClient('mongodb://localhost:27017')
db = client.examples

db.honolulu.remove()
db.honolulu1.remove()

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

