

# script for conducting exploratory data analysis on tag data: filename- tags.py
"""
Explores different `tag` contents.
"""
from collections import defaultdict
from pprint import pprint
import json
import xml.etree.cElementTree as ET

#from mapparser import count_tags

SAMPLE_FILE = "./honolulu_hawaii_sample.osm"
TAG_KEYS = [('highway', 1788),
 ('name', 1384),
 ('tiger:county', 653),
 ('tiger:cfcc', 636),
 ('tiger:name_base', 580),
 ('tiger:name_type', 552),
 ('tiger:zip_left', 512),
 ('tiger:zip_right', 492),
 ('building', 472),
 ('tiger:reviewed', 459),
 ('source', 296),
 ('service', 242),
 ('oneway', 216),
 ('amenity', 143),
 ('lanes', 128),
 ('natural', 124),
 ('tiger:tlid', 103),
 ('tiger:source', 103),
 ('tiger:upload_uuid', 101),
 ('ref', 100),
 ('tiger:separated', 95),
 ('addr:street', 83),
 ('leisure', 80),
 ('building:levels', 78),
 ('addr:housenumber', 73),
 ('addr:postcode', 69),
 ('landuse', 64),
 ('addr:city', 61),
 ('height', 55),
 ('aeroway', 51),
 ('bridge', 50),
 ('golf', 49),
 ('layer', 47),
 ('access', 46),
 ('addr:state', 46),
 ('power', 45),
 ('name_1', 44),
 ('man_made', 43),
 ('building:material', 40),
 ('roof:material', 39),
 ('roof:colour', 38),
 ('type', 37),
 ('roof:shape', 37),
 ('building:part', 35),
 ('ele', 33),
 ('surface', 32),
 ('tiger:name_base_1', 31),
 ('shop', 30),
 ('tiger:zip_left_1', 29),
 ('sport', 28),
 ('gnis:feature_id', 27),
 ('place', 24),
 ('website', 23),
 ('tiger:name_type_1', 23),
 ('gnis:county_name', 23),
 ('building:color', 23),
 ('waterway', 22),
 ('roof:height', 22),
 ('width', 20),
 ('roof:orientation', 20),
 ('parking', 19),
 ('bicycle', 18),
 ('building:colour', 17),
 ('barrier', 17),
 ('tourism', 16),
 ('tracktype', 15),
 ('gnis:reviewed', 15),
 ('gnis:import_uuid', 15),
 ('is_in', 15),
 ('roof:angle', 14),
 ('maxspeed', 14),
 ('wikipedia', 13),
 ('wheelchair', 12),
 ('entrance', 12),
 ('foot', 12),
 ('created_by', 11),
 ('mown', 11),
 ('tiger:zip_right_1', 11),
 ('min_height', 10),
 ('roof:levels', 10),
 ('note', 10),
 ('area', 10),
 ('gnis:ST_alpha', 9),
 ('cuisine', 9),
 ('phone', 9),
 ('gnis:County_num', 9),
 ('gnis:Class', 9),
 ('import_uuid', 9),
 ('level', 9),
 ('fixme', 9),
 ('gnis:County', 9),
 ('gnis:id', 9),
 ('gnis:ST_num', 9),
 ('restriction', 9),
 ('name:haw', 8),
 ('gnis:created', 8),
 ('cycleway', 8),
 ('operator', 8),
 ('tiger:name_direction_prefix', 8),
 ('boundary', 8),
 ('name:en', 8),
 ('gnis:feature_type', 8),
 ('building:min_level', 7),
 ('fee', 6),
 ('toilets:wheelchair', 6),
 ('is_in:iso_3166_2', 5),
 ('is_in:country_code', 5),
 ('is_in:country', 5),
 ('opening_hours', 5),
 ('leaf_type', 5),
 ('junction', 5),
 ('genus', 4),
 ('species:en', 4),
 ('is_in:state', 4),
 ('start_date', 4),
 ('tiger:name_base_2', 4),
 ('horse', 4),
 ('noexit', 4),
 ('is_in:state_code', 4),
 ('tiger:zip_left_2', 4),
 ('tiger:PLACEFP', 3),
 ('emergency', 3),
 ('tiger:FUNCSTAT', 3),
 ('tiger:LSAD', 3),
 ('species', 3),
 ('direction', 3),
 ('tiger:PCICBSA', 3),
 ('mtb', 3),
 ('tiger:NAME', 3),
 ('population', 3),
 ('tiger:PCINECTA', 3),
 ('boat', 3),
 ('tiger:PLACENS', 3),
 ('tiger:name_type_2', 3),
 ('atm', 3),
 ('route', 3),
 ('tiger:NAMELSAD', 3),
 ('name:ja', 3),
 ('tiger:CLASSFP', 3),
 ('religion', 3),
 ('tiger:CPI', 3),
 ('capacity', 3),
 ('generator:source', 3),
 ('tiger:STATEFP', 3),
 ('tower:type', 3),
 ('sidewalk', 3),
 ('network', 3),
 ('fax', 3),
 ('attraction', 3),
 ('tiger:PLCIDFP', 3),
 ('tiger:MTFCC', 3),
 ('takeaway', 2),
 ('addr:country', 2),
 ('traffic_signals', 2),
 ('url', 2),
 ('antenna:type', 2),
 ('building_1', 2),
 ('parking:lane:right', 2),
 ('mooring', 2),
 ('denotation', 2),
 ('governance_type', 2),
 ('smoking', 2),
 ('bus', 2),
 ('alt_name', 2),
 ('motorcar', 2),
 ('construction:railway', 2),
 ('parking:lane:both', 2),
 ('parking:lane:left', 2),
 ('horizontal_bar', 2),
 ('roof:alignment', 2),
 ('footway', 2),
 ('protect_class', 2),
 ('water', 2),
 ('fence_type', 2),
 ('railway', 2),
 ('roof:color', 2),
 ('name:ru', 2),
 ('name_2', 2),
 ('source:population', 2),
 ('name:lt', 2),
 ('site_ownership', 2),
 ('lit', 2),
 ('name:de', 2),
 ('tiger:zip_left_3', 2),
 ('motorcycle', 2),
 ('addr:place', 1),
 ('dock', 1),
 ('state_capital', 1),
 ('structure', 1),
 ('alt_name:fr', 1),
 ('store_ref', 1),
 ('symbol', 1),
 ('collection_times', 1),
 ('alt_name:zh_pinyin', 1),
 ('alt_name:es', 1),
 ('alt_name:mn', 1),
 ('name:vi', 1),
 ('generator:method', 1),
 ('alt_name:el', 1),
 ('wetland', 1),
 ('shower', 1),
 ('tiger:zip_right_4', 1),
 ('park_ride', 1),
 ('oneway:bicycle', 1),
 ('name:zh', 1),
 ('traffic_calming', 1),
 ('generator:type', 1),
 ('supervised', 1),
 ('craft', 1),
 ('aerodrome:type', 1),
 ('building_2', 1),
 ('building_3', 1),
 ('building_4', 1),
 ('name:hi', 1),
 ('id', 1),
 ('alt_name:hi', 1),
 ('change:backward', 1),
 ('unisex', 1),
 ('tiger:name_direction_prefix_1', 1),
 ('Ohiaku', 1),
 ('name:pl', 1),
 ('name:ko', 1),
 ('name:botanical', 1),
 ('addr:housename', 1),
 ('name:pt', 1),
 ('turn:lanes:forward', 1),
 ('internet_access', 1),
 ('cycleway:right', 1),
 ('monitoring_station', 1),
 ('note:en', 1),
 ('overtaking', 1),
 ('incline', 1),
 ('organic', 1),
 ('alt_name:ko', 1),
 ('denomination', 1),
 ('lanes:backward', 1),
 ('email', 1),
 ('artwork_type', 1),
 ('capital', 1),
 ('generator:output:electricity', 1),
 ('name:zh_pinyin', 1),
 ('change:lanes:forward', 1),
 ('is_in:ocean', 1),
 ('trail_visibility', 1),
 ('content', 1),
 ('alt_name:zh', 1),
 ('name:es', 1),
 ('public_transport', 1),
 ('tiger:zip_right_2', 1),
 ('lanes:forward', 1),
 ('destination:ref:lanes:forward', 1),
 ('motor_vehicle', 1),
 ('population:date', 1),
 ('note:ko', 1),
 ('icao', 1),
 ('iata', 1),
 ('Island', 1),
 ('voltage', 1),
 ('alt_name:tl', 1),
 ('covered', 1),
 ('sac_scale', 1),
 ('location', 1),
 ('is_state', 1),
 ('tiger:zip_left_4', 1),
 ('drive_through', 1),
 ('tiger:zip_right_3', 1),
 ('name:uk', 1)]
def examine_tags(osmfile, tag_range, item_limit):
    assert len(tag_range) == 2
    # use pre-loaded tag_keys list of tuples, if exists
    if TAG_KEYS:
        tag_keys = TAG_KEYS
    # else call mapparser count_tags method to pull sorted list of top tags
    else:
        _, tag_keys = count_tags(osmfile)
    # list comprehension for producing a list of tag_keys in string format
    tag_keys = [tag_key[0] for tag_key in tag_keys][tag_range[0]:tag_range[1]]
    print "Examining tag keys: {}".format(tag_keys)

    # open osm file
    osm_file = open(osmfile, "r")

    # initialize data with default set data structure
    data = defaultdict(set)

    # iterate through elements
    for _, elem in ET.iterparse(osm_file, events=("start",)):
        # check if the element is a node or way
        if elem.tag == "node" or elem.tag == "way":
            # iterate through children matching `tag`
            for tag in elem.iter("tag"):
                # skip if does not contain key-value pair
                if 'k' not in tag.attrib or 'v' not in tag.attrib:
                    continue
                key = tag.get('k')
                val = tag.get('v')
                # add to set if in tag keys of interest and is below the item limit
                if key in tag_keys and len(data[key]) < item_limit:
                    data[key].add(val)
    return data

def main(tag_range=(0, 10), item_limit=10):
    # call examine_tags fucntion
    tag_data = dict(examine_tags(SAMPLE_FILE, tag_range, item_limit))

    # convert sets to JSON-read/writeable format (list)
    for key in tag_data:
        tag_data[key] = list(tag_data[key])

    # write to file
    #json.dump(tag_data, open(OSMFILE + '-tag-data.json', 'w'))

    # pretty print
    pprint(tag_data)

    # return data
    return tag_data

if __name__ == '__main__':
    main(tag_range=(0, -1), item_limit=20)


