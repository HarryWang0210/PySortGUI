import tables
import os
import pandas as pd
import numpy as np
filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
# filename = "data/MD123_2022-09-07_10-38-00.h5"

with tables.open_file(filename, mode="r") as file:
    # EventsHeader = file.get_node("/Spikes/spike000/UnitsHeader")
    # print(pd.DataFrame(EventsHeader.read()))

    # EventsHeader = file.get_node("/Spikes/spike000/Waveforms")
    # print(pd.DataFrame(EventsHeader.read()))

    # EventsHeader = file.get_node("/Spikes/spike000/TimeStamps")
    # print(EventsHeader.read()[0: 100])

    # # EventsHeader = file.get_node("/Spikes/spike000")
    # # print([group for group in EventsHeader._v])

    # EventsHeader = file.get_node("/Spikes/spike000/Unit_01/Indxs")
    # print(pd.DataFrame(EventsHeader.read()))

    chan_ID = 0
    if "/Spikes" in file.root:
        Spikes = file.get_node("/Spikes")
        spike_chan_list = Spikes._f_list_nodes()

        try:
            spike_chan = Spikes._f_get_child("spike" + str(chan_ID).zfill(3))
        except tables.NodeError:
            print("The /Spikes node does not contain the spike" +
                  str(chan_ID).zfill(3) + " node.")

        # get timestamps
        try:
            timestamps = spike_chan._f_get_child("TimeStamps").read()
        except tables.NodeError:
            print("The " + str(chan_ID).zfill(3) +
                  " node does not contain the TimeStamps node.")

        # get units_info
        try:
            units_info = pd.DataFrame(
                spike_chan._f_get_child("UnitsHeader").read())
        except tables.NodeError:
            print("The " + str(chan_ID).zfill(3) +
                  " node does not contain the UnitsHeader node.")

        # get waveforms
        try:
            waveforms = spike_chan._f_get_child("Waveforms").read()
        except tables.NodeError:
            print("The " + str(chan_ID).zfill(3) +
                  " node does not contain the Waveforms node.")

        units_id = np.zeros(len(timestamps))
        not_zero_units = units_info[units_info["NumRecords"] > 0]
        for unit in not_zero_units.index:
            unit_h5_name = "/".join([not_zero_units.loc[unit, "H5Location"].decode(
            ), not_zero_units.loc[unit, "H5Name"].decode()])
            ind = file.get_node(unit_h5_name).read()
            units_id[ind] = not_zero_units.loc[unit, "ID"]

        # return units_info, units_id, timestamps, waveforms
        # units_info[units_info["NumRecords"] > 0]
        # try:
        #     waveforms = spike_chan._f_get_child("Waveforms").read()
        # except tables.NodeError:
        #     print("The " + str(chan_ID).zfill(3) +
        #           " node does not contain the Waveforms node.")

        # for i in spike_chan._f_list_nodes():
        # print(i)
        # if isinstance(Spikes, tables.Group):
        #     data = Spikes._f_get_child("spike" + str(0).zfill(3))
        #     print(data)

    # # get all
    # print(file)
