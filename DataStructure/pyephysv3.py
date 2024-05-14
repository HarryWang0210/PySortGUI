from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import tables

if TYPE_CHECKING:
    from .datav3 import ContinuousData, DiscreteData, SpikeSorterData

from .header_class import EventsHeader, FileHeader, RawsHeader, SpikesHeader

logger = logging.getLogger(__name__)


def loadPyephysHeader(filename):
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    file_header = np.array([])
    raws_header = np.array([])
    spikes_header = np.array([])
    events_header = np.array([])

    file_headers: list[dict] = []
    raws_headers: list[tuple[str, dict]] = []
    events_headers: list[tuple[str, dict]] = []
    spikes_headers: list[tuple[str, dict]] = []

    with tables.open_file(filename, mode="r") as file:
        if "/FileHeader" in file.root:
            file_header = file.get_node("/FileHeader").read()
        else:
            logger.info('Can not load FileHeader.')

        if "/RawsHeader" in file.root:
            raws_header = file.get_node("/RawsHeader").read()
        else:
            logger.critical('Can not load RawsHeader.')

        if "/SpikesHeader" in file.root:
            spikes_header = file.get_node("/SpikesHeader").read()
        else:
            df_spikes_header = None
            logger.info('Can not load SpikesHeader.')

        if "/EventsHeader" in file.root:
            events_header = file.get_node("/EventsHeader").read()
        else:
            df_events_header = None
            logger.info('Can not load EventsHeader.')

    file_header = recarrayToDictList(file_header)
    file_headers = [FileHeader.model_validate(header).model_dump()
                    for header in file_header]

    raws_header = recarrayToDictList(raws_header)
    raws_headers = [(filename, RawsHeader.model_validate(header).model_dump())
                    for header in raws_header]
    # print([RawsHeader.model_validate(dict(zip(raws_header.dtype.names, record)))
    #       for record in raws_header])
    # print(raws_headers)
    # df_raws_header = pd.DataFrame(raws_header)

    # # convert to string
    # df_raws_header = df_raws_header.applymap(lambda x: x.decode(
    #     'utf-8') if isinstance(x, bytes) else x)

    # df_raws_header = df_raws_header.sort_values('ID')

    # df_spikes_header = pd.DataFrame(spikes_header)
    spikes_header = recarrayToDictList(spikes_header)
    spikes_headers = [(filename, SpikesHeader.model_validate(header).model_dump())
                      for header in spikes_header]
    # convert to string
    # df_spikes_header = df_spikes_header.applymap(lambda x: x.decode(
    #     'utf-8') if isinstance(x, bytes) else x)

    # # add Label field if not have
    # if "Label" not in df_spikes_header.columns:
    #     df_spikes_header['Label'] = "default"

    # df_spikes_header = df_spikes_header.sort_values('ID')

    events_header = recarrayToDictList(events_header)
    events_headers = [(filename, EventsHeader.model_validate(header).model_dump())
                      for header in events_header]

    data = {}
    if file_headers:
        data['FileHeader'] = file_headers
    if raws_headers:
        data['RawsHeader'] = raws_headers
    if events_headers:
        data['EventsHeader'] = events_headers
    if spikes_headers:
        data['SpikesHeader'] = spikes_headers

    # df_events_header = pd.DataFrame(spikes_header)

    # # convert to string
    # df_events_header = df_events_header.applymap(lambda x: x.decode(
    #     'utf-8') if isinstance(x, bytes) else x)

    # df_events_header = df_events_header.sort_values('ID')

    # return {'RawsHeader': df_raws_header,
    #         'SpikesHeader': df_spikes_header,
    #         'EventsHeader': df_events_header}
    return data


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


def saveSpikes(filename, header: dict, unit_header: pd.DataFrame | None = None,
               unit_IDs: np.ndarray = [], timestamps: np.ndarray = [], waveforms: np.ndarray = []):
    basename, extname = os.path.splitext(filename)
    # if extname == "h5raw":
    #     filename = ".".join(basename, "h5")
    path = header['H5Location']
    path_split = path.split('/')
    with tables.open_file(filename, mode="a") as file:
        if path in file.root:
            # spike_chan = file.get_node(path)
            file.remove_node(path)
            file.flush()
            # spike_chan._f_remove(force=True)
        file.create_group('/'+path_split[-2], path_split[-1])
        file.create_array(path, "TimeStamps", timestamps)
        file.create_array(path, "Waveforms", waveforms)
        file.flush()

        object_cols = unit_header.dtypes[unit_header.dtypes == 'object'].index
        string_len = unit_header[object_cols].applymap(len)
        max_length = string_len.max()
        max_length = max_length.apply(lambda x: f'S{x}' if x > 0 else 'S1')
        unit_table = unit_header.to_records(
            index=False, column_dtypes=max_length.to_dict())
        file.create_table(path, "UnitsHeader", unit_table)
        file.flush()

        not_zero_unit = unit_header.loc[unit_header['NumRecords'] > 0, 'ID']
        for unit_ID in not_zero_unit:
            ID_group = file.create_group(path, f'Unit_{unit_ID:02}')
            file.create_array(ID_group, 'Indxs',
                              (unit_IDs == unit_ID).nonzero()[0])
            file.flush()


def saveSpikesHeader(filename, header: pd.DataFrame | None = None):
    path = '/SpikesHeader'
    with tables.open_file(filename, mode="a") as file:
        if path in file.root:
            # spike_chan = file.get_node(path)
            # spike_chan._f_remove()
            file.remove_node(path)
            file.flush()

        object_cols = header.dtypes[header.dtypes == 'object'].index
        # # logger.debug(header.dtypes)
        # # logger.debug(header)

        string_len = header[object_cols].applymap(len)
        max_length = string_len.max()
        # # logger.debug(max_length)
        max_length = max_length.apply(lambda x: f'S{x}' if x > 0 else 'S1')
        spike_table = header.to_records(index=False,
                                        column_dtypes=max_length.to_dict())
        # logger.debug(spike_table)
        # logger.debug(spike_table.dtype)
        file.create_table('/', 'SpikesHeader', spike_table)
        file.flush()


def saveHeaders(filename: str, path: str, headers: list[RawsHeader | FileHeader | SpikesHeader | EventsHeader]):
    """_summary_

    Args:
        filename (str): _description_
        path (str): /RawsHeader
        header (RawsHeader): _description_
    """
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
        if path in file.root:
            file.remove_node(path, recursive=True)
            file.flush()

        df_header = pd.DataFrame(
            [header.model_dump(extra='append') for header in headers])
        # file.create_external_link(
        #     '/', 'Raws', '{}:/Raws'.format(filename+'raw'))
        # file.flush()

        # RawsHeader
        file.create_table(os.path.split(path)[0], os.path.split(path)[1],
                          dataframeToRecarry(df_header),
                          createparents=True)
        file.flush()


def saveRawsData(filename: str, path: str, header: RawsHeader, data: np.ndarray, overwrite=False):
    """_summary_

    Args:
        filename (str): _description_
        path (str): /Raws/raw000
        header (RawsHeader): _description_
        data (np.ndarray): _description_
        overwrite (bool, optional): _description_. Defaults to False.
    """
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    where, name = os.path.split(path)

    # with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
    with tables.open_file(filename+'raw', mode='a', title=title, filters=filt) as file_raw:
        if path in file_raw.root:
            if not overwrite:
                return

            file_raw.remove_node(path, recursive=True)
            file_raw.flush()

        # elif where in file_raw.root:
        #     file_raw.create_group('/', 'Raws', createparents=True)

        data_array = file_raw.create_carray(where, name,
                                            obj=data, title=header.Name,
                                            createparents=True)
        for key, value in header.model_dump(extra='append').items():
            data_array.set_attr(f'header_{key}', value)
        file_raw.flush()


def saveDiscreteData(filename: str, path: str, header: SpikesHeader, unit_header: pd.DataFrame,
                     unit_IDs: np.ndarray, timestamps: np.ndarray, waveforms: np.ndarray | None = None,
                     overwrite=False):
    """_summary_

    Args:
        filename (str): _description_
        path (str): /Spikes/spike000
        header (SpikesHeader): _description_
        unit_header (pd.DataFrame): _description_
        unit_IDs (np.ndarray): _description_
        timestamps (np.ndarray): _description_
        waveforms (np.ndarray | None, optional): _description_. Defaults to None.
        overwrite (bool, optional): _description_. Defaults to False.
    """
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    where, name = os.path.split(path)
    if waveforms is None and where == '/Spikes':
        logger.error(f'Missing waveforms data when saving {name}.')

    with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
        if path in file.root:
            if not overwrite:
                return

            file.remove_node(path, recursive=True)
            file.flush()

        file.create_carray(path, "TimeStamps",
                           obj=timestamps,  createparents=True)
        file.flush()

        unit_table_name = 'EventsHeader'
        unit_suffix = 'Event'
        if not waveforms is None:
            # is spikes
            unit_table_name = 'UnitsHeader'
            unit_suffix = 'Unit_'
            file.create_carray(path, "Waveforms",
                               obj=waveforms, createparents=True)

        file.create_table(path, unit_table_name,
                          obj=dataframeToRecarry(unit_header), createparents=True)

        file.flush()

        not_zero_unit = unit_header.loc[unit_header['NumRecords'] > 0, 'ID']
        for unit_ID in not_zero_unit:
            file.create_carray(f'{path}/{unit_suffix}{unit_ID:02}', 'Indxs',
                               obj=(unit_IDs == unit_ID).nonzero()[0],
                               createparents=True)
            file.flush()


def deleteSpikes(filename, path):
    with tables.open_file(filename, mode="a") as file:
        if path in file.root:
            file.remove_node(path, recursive=True)
            file.flush()


def exportToPyephys(new_filename: str, data_object: SpikeSorterData):
    # filt = tables.Filters(complib='zlib', complevel=1)
    overwrite = True

    # Raws
    raws_headers: list[RawsHeader] = []
    for ID in data_object.channel_IDs:
        raws_object = data_object.getRaw(ID, load_data=True)
        raws_header = RawsHeader.model_validate(raws_object.header,
                                                extra='allow')
        saveRawsData(new_filename, f'/Raws/raw{raws_object.channel_ID:03}',
                     raws_header, raws_object.data, overwrite=overwrite)
        raws_headers.append(raws_header)
    saveHeaders(new_filename, '/RawsHeader', raws_headers)

    with tables.open_file(new_filename, mode='a') as file:
        if not '/Raws' in file.root:
            file.create_external_link(
                '/', 'Raws', '{}:/Raws'.format(os.path.basename(new_filename+'raw')))
            file.flush()

    # Spikes
    spikes_headers: list[SpikesHeader] = []
    for ID in data_object.channel_IDs:
        raws_object = data_object.getRaw(ID)
        for label in raws_object.spikes:
            spikes_object = data_object.getSpike(ID, label, load_data=True)

            h5_location = f'spike{ID:03}'
            if label != 'default':
                h5_location = f'spike{ID:03}{label}'

            if spikes_object == 'Removed':
                deleteSpikes(new_filename, f'/Spikes/{h5_location}')
                continue

            spikes_header = SpikesHeader.model_validate(spikes_object.header,
                                                        extra='allow')

            saveDiscreteData(new_filename, f'/Spikes/{h5_location}', spikes_header,
                             unit_header=spikes_object.unit_header,
                             unit_IDs=spikes_object.unit_IDs,
                             timestamps=spikes_object.timestamps,
                             waveforms=spikes_object.waveforms,
                             overwrite=overwrite)
            spikes_headers.append(spikes_header)
    saveHeaders(new_filename, '/SpikesHeader', spikes_headers)


def dataframeToRecarry(df: pd.DataFrame):
    object_cols = df.dtypes[df.dtypes == 'object'].index
    string_len = df[object_cols].applymap(len)
    max_length = string_len.max()
    max_length = max_length.apply(lambda x: f'S{x}' if x > 0 else 'S1')
    return df.to_records(index=False,
                         column_dtypes=max_length.to_dict())


def recarrayToDictList(recarray: np.ndarray):
    if recarray.dtype.names is None:
        return []
        # print(recarray.tolist())
    return [dict(zip(recarray.dtype.names, record)) for record in recarray]


if __name__ == '__main__':
    data = loadPyephysHeader(
        r'C:\Users\harry\Desktop\Lab\Project_spikesorter\PySortGUI\data\RU01_2022-08-01_11-20-12\RU01_2022-08-01_11-20-12.h5')
    # print(data)

    # filename = "data/MX6-22_2020-06-17_17-07-48_no_ref.h5"
    # headers = loadPyephys(filename)
    # spike_header = pd.DataFrame(headers['SpikesHeader']).to_dict('records')[0]
    # spike = loadSpikes(filename, spike_header['H5Location'])
    # print(spike is dict)
