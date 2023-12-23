import tables
import os
import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)


def loadPyephys(filename):
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        # FileHeader = file.get_node("/FileHeader")
        if "/RawsHeader" in file.root:
            raws_header = file.get_node("/RawsHeader").read()
            df_raws_header = pd.DataFrame(raws_header)

            # convert to string
            df_raws_header = df_raws_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)

            df_raws_header = df_raws_header.sort_values('ID')
        else:
            logger.critical('Can not load RawsHeader.')

        if "/SpikesHeader" in file.root:
            spikes_header = file.get_node("/SpikesHeader").read()
            df_spikes_header = pd.DataFrame(spikes_header)

            # convert to string
            df_spikes_header = df_spikes_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)

            # add Label field if not have
            if "Label" not in df_spikes_header.columns:
                df_spikes_header['Label'] = "default"

            # ========== Test Label ==========
            new_row = df_spikes_header.iloc[0, :].copy()
            new_row['Label'] = 'test label'
            df_spikes_header = pd.concat(
                [df_spikes_header, new_row.to_frame().T], axis=0, ignore_index=True)
            # ================================
            df_spikes_header = df_spikes_header.sort_values('ID')

        else:
            df_spikes_header = None
            logger.info('Can not load SpikesHeader.')

        if "/EventsHeader" in file.root:
            spikes_header = file.get_node("/EventsHeader").read()
            df_events_header = pd.DataFrame(spikes_header)

            # convert to string
            df_events_header = df_events_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)

            df_events_header = df_events_header.sort_values('ID')

        else:
            df_events_header = None
            logger.info('Can not load EventsHeader.')

    return {'RawsHeader': df_raws_header,
            'SpikesHeader': df_spikes_header,
            'EventsHeader': df_events_header}


def loadRaws(filename: str, path: str, name: str):
    # chan_ID = int(chan_ID)
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if "/Raws" in file.root:
            raws = file.get_node(path)
            with tables.open_file(raws(mode='r')._v_file.filename, mode="r") as rawsfile:
                data = rawsfile.get_node('/'.join([path, name]))
                return data[:]
        else:
            raise


def loadSpikes(filename, path):
    # chan_ID = int(chan_ID)
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if path not in file.root:
            return

        spike_chan = file.get_node(path)

        df_units_header = pd.DataFrame(
            spike_chan._f_get_child("UnitsHeader").read())

        # convert to string
        df_units_header = df_units_header.applymap(lambda x: x.decode(
            'utf-8') if isinstance(x, bytes) else x)

        if 'UnitType' not in df_units_header.columns:
            df_units_header['UnitType'] = 'Unit'

            unsorted_pattern = r'(?i)unsorted'
            matches = df_units_header['Name'].str.contains(
                unsorted_pattern, regex=True)
            df_units_header.loc[matches, 'UnitType'] = 'Unsorted'

            invalid_pattern = r'(?i)invalid'
            matches = df_units_header['Name'].str.contains(
                invalid_pattern, regex=True)
            df_units_header.loc[matches, 'UnitType'] = 'Invalid'

        timestamps = spike_chan._f_get_child("TimeStamps").read()
        waveforms = spike_chan._f_get_child("Waveforms").read()

        # get units id
        unitID = np.zeros(len(timestamps))
        not_zero_units = df_units_header[df_units_header["NumRecords"] > 0]
        for unit in not_zero_units.index:
            unit_h5_name = "/".join([not_zero_units.loc[unit, "H5Location"],
                                    not_zero_units.loc[unit, "H5Name"]])
            ind = file.get_node(unit_h5_name).read()
            unitID[ind] = not_zero_units.loc[unit, "ID"]

        return {"unitHeader": df_units_header,
                "unitID": unitID,
                "timestamps": timestamps,
                "waveforms": waveforms}


if __name__ == '__main__':
    filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
    headers = loadPyephys(filename)
    spike_header = pd.DataFrame(headers['SpikesHeader']).to_dict('records')[0]
    spike = loadSpikes(filename, spike_header['H5Location'])
    print(spike is dict)
