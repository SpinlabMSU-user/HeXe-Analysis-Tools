import json
import struct
import numpy as np
from NumericalTools import *

def read_dig(file_path, unit="volt"):
    '''Reads data from data file. Code made by Flo'''
    data_file = open(file_path, "rb") #opens binary file
    header_length = struct.unpack("<L", data_file.read(4))[0]
    data_file.seek(4) #moves to header
    #header is just of length of header found earlier
    hdr = json.loads(data_file.read(header_length)) 
    #to interpret the numbers
    byte_depth = hdr["byte_depth"] #finds number of bytes per sample point
    bit_shift = hdr["bit_shift"] #finds if there any bit shifting
    #extract header information to scale the numbers
    number_of_channels = len(hdr["channel_list"])
    if "current_gains" in hdr: #finds gains
        channel_gains_list = hdr["current_gains"]["1"]
        #finds conversions for gains
        gain_conversion_table = hdr["gain_conversion"] 
        gains_list = {}
        for i in channel_gains_list: 
            #lookup the gain table
            gain = gain_conversion_table[str(channel_gains_list[str(i)])]
            #convert to float for scaling
            gains_list[str(i)] = float(gain.lstrip("x"))    
    else: gains_list=[1 for i in range(number_of_channels)]
    
    if "full_scale" in hdr: full_scale = hdr["full_scale"]
    else: full_scale = 20
    
    if unit == "tesla": 
        conversion_factor = hdr["conversion_factor"]
    else: 
        conversion_factor = 1
 
    sample_rate = hdr["freq_hz"]
    if "channel_names" in hdr: channel_names = hdr["channel_names"]
    else: channel_names = {}
    measurement_name = hdr["measurement_name"]
    
    #convert the data
    dt = None
    if byte_depth == 2: dt = np.int16 #size of each sample
    elif byte_depth == 4: dt = np.int32
    else: raise Exception("unknown bit_depth")
 
    # Reads from position 4 + header_length
    data_file.seek(4+header_length) #moves to after header
    all_arrays = np.fromfile(data_file, dtype=dt) #reads binary data
    # Do a right shift if necessary
    if bit_shift != 0:
        all_arrays = np.right_shift(all_arrays, bit_shift)
    
    # Scaling of the raw data
    scaled_data_dict={}
    for i in range(number_of_channels):
        #skips over to only look at 1 array at a time
        single_ch = all_arrays[i::number_of_channels] 
        single_ch_scaled = single_ch*full_scale*conversion_factor/\
                           (gains_list[str(i)]*2**(24))
        #create a dictionary of the channels
        scaled_data_dict.update(dict([(str(i), single_ch_scaled)]))
    time_axis = np.array([i/sample_rate for i in range(len(single_ch))])
    scaled_data_dict.update(dict(time=time_axis))
    return (scaled_data_dict, sample_rate, channel_names,measurement_name,hdr)


# #Make data into workable data

def make_channels_list(hdr,data):
    '''Makes list of channels from input data and another list with their names
    works by using a list of channel names and if there exists each different
    desired channel in names then uses same channel to put into channels list
    to be used later'''
    cn = hdr[u'channel_names']
    channel_names_list = []
    channels = []
    channel_names = []
    for key, value in cn.iteritems():
        temp = [key,value]
        channel_names_list.append(temp)
    for ch in channel_names_list: #Assumes there exists each squid channel
        if ch[1] == u'SQUID_X1':
            x1 = data[ch[0]]
            channels.append(x1)
            channel_names.append('x1')
        elif ch[1] == u'SQUID_X2':
            x2 = data[ch[0]]
            channels.append(x2)
            channel_names.append('x2')
        elif ch[1] == u'SQUID_Y1':
            y1 = data[ch[0]]
            channels.append(y1)
            channel_names.append('y1')
        elif ch[1] == u'SQUID_Y2':
            y2 = data[ch[0]]
            channels.append(y2)
            channel_names.append('y2')
        elif ch[1] == u'SQUID_Z1':
            z1 = data[ch[0]]
            channels.append(z1)
            channel_names.append('z1')
        elif ch[1] == u'SQUID_Z2':
            z2 = data[ch[0]]
            channels.append(z2)
            channel_names.append('z2')   
        #sometimes data doesn't have all the channels that channel names does   
        if len(data) > 700:      
            if ch[1].lower() == u'lockin_he':
                lockin_he = data[ch[0]]
                channels.append(lockin_he)
                channel_names.append('lockin he')
            if ch[1].lower() == u'lockin_xe':
                lockin_xe = data[ch[0]]
                channels.append(lockin_xe)
                channel_names.append('lockin xe')
    return (channels, channel_names)

def find_locations(channel_names):
    '''
    finds locations of different channels in lists
    of channels so orientation is known for later use.
    -returns-
    tuple of channel locations: (x1 location, x2 location, y1 location, z1 
                                 location, z2 location)
    and the lockin channels if they are available.
    '''
    
    x1loc = channel_names.index('x1')
    x2loc = channel_names.index('x2')
    y1loc = channel_names.index('y1')
    y2loc = channel_names.index('y2')
    z1loc = channel_names.index('z1')
    z2loc = channel_names.index('z2')
    channel_locs = (x1loc,x2loc,y1loc,y2loc,z1loc,z2loc)
    try:
        lockin_he_loc = channel_names.index('lockin he')
        channel_locs = channel_locs + (lockin_he_loc,)
    except:
        pass
    try:
        lockin_xe_loc = channel_names.index('lockin xe')
        channel_locs = channel_locs + (lockin_xe_loc,)
    except:
        pass
    return channel_locs

def get_any_channels_from_dig(f, list_of_channels, chopped_time = None):
    '''
    From a .dig file, list of desired channels are returned in order
    inputed If time is needed to be chopped then it must either be an int,
    float or list of form [start time, end time].
    
    list of channels may only include:
    x1,x2, y1,y2, z1,z2, lockin_he, lockin_xe'''
    desired_channels = []    
    list_of_channels = [a.lower() for a in list_of_channels]
    
    data, sample_freq, channel_names, measurement_names, hdr = read_dig(f)
    time = data['time']
    channels, channel_names = make_channels_list(hdr,data)
    
    try: #checks if both lockins are present
        x1loc, x2loc, y1loc, y2loc, z1loc, z2loc, lockin_he_loc,\
        lockin_xe_loc = find_locations(channel_names)      
    except: #if xe or he or both isnt in channels
        if 'lockin he' in channel_names:
            x1loc, x2loc, y1loc, y2loc, z1loc, z2loc, lockin_he_loc \
                = find_locations(channel_names)
        elif 'lockin xe' in channel_names:
            x1loc, x2loc, y1loc, y2loc, z1loc, z2loc, lockin_xe_loc \
                = find_locations(channel_names)
        else: #neither lockin present
            x1loc, x2loc, y1loc, y2loc, z1loc, z2loc \
                = find_locations(channel_names)
                
    for channel in list_of_channels:
        if channel == 'x1':
            desired_channels.append(channels[x1loc])
        elif channel == 'x2':
            desired_channels.append(channels[x2loc])
        elif channel == 'y1':
            desired_channels.append(channels[y1loc])
        elif channel == 'y2':
            desired_channels.append(channels[y2loc])
        elif channel == 'z1':
            desired_channels.append(channels[z1loc])
        elif channel == 'z2':
            desired_channels.append(channels[z2loc])
        elif channel == 'lockin_he' or channel == 'lockin he':
            desired_channels.append(channels[lockin_he_loc])
        elif channel == 'lockin_xe' or channel == 'lockin xe':
            desired_channels.append(channels[lockin_xe_loc])
        else:
            print '{} is not a valid channel'.format(channel)
            
    if chopped_time:
        if type(chopped_time) == int or type(chopped_time) == float:
            start_time = chopped_time
            time_loc_start = min(range(len(time)), key = lambda i: \
                                                    abs(time[i]-start_time))
            
            time = time[time_loc_start:]
            desired_channels = [dc[time_loc_start:] for dc in \
                                desired_channels]
        if type(chopped_time) == list and len(chopped_time) == 2:
            start_time, end_time = chopped_time
            time_loc_start = min(range(len(time)), key=lambda i: \
                                                    abs(time[i]-start_time))
            time_loc_end = min(range(len(time)), key=lambda i: \
                                                     abs(time[i]-end_time))
        
            time = time[time_loc_start:time_loc_end]
            desired_channels = [dc[time_loc_start:time_loc_end] for dc in\
                                   desired_channels]
    
    return time, desired_channels, sample_freq

def make_gradiometer(channel_1, channel_2, channel='z'):
    '''
    Creates gradiometer, difference in magnetic field between channels, 
    of channels. Must be in order of channel 1, channel 2 so orientations
    are as correctly assumed.
    -Returns-
    gradiometer (np.array)
    '''
    if channel == 'z':
        return np.subtract(channel_1,channel_2)
    else:
        return np.add(channel_1, channel_2)

def make_magnetometer(channel_1,channel_2, channel = 'z'):
    '''
    Creates magnetometer, average magnetic field, of channels. Must be in
    order of channel 1, channel 2 so orientations are as correctly assumed.
    -Returns-
    magnetometer (np.array)
    '''
    if channel == 'z':
        return np.add(channel_1, channel_2) / 2
    else:
        return np.subtract(channel_1, channel_2) / 2