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

    update_spikes_header = False
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
            if not 'Label' in spikes_header.dtype.names:
                update_spikes_header = True
        else:
            logger.info('Can not load SpikesHeader.')

        if "/EventsHeader" in file.root:
            events_header = file.get_node("/EventsHeader").read()
        else:
            logger.info('Can not load EventsHeader.')

    file_header = recarrayToDictList(file_header)
    file_headers = [FileHeader.model_validate(header).model_dump()
                    for header in file_header]

    raws_header = recarrayToDictList(raws_header)
    raws_headers = [(filename, RawsHeader.model_validate(header).model_dump())
                    for header in raws_header]

    spikes_header = recarrayToDictList(spikes_header)
    spikes_headers = [(filename, SpikesHeader.model_validate(header).model_dump())
                      for header in spikes_header]
    if update_spikes_header:
        records = []
        for spike in spikes_headers:
            records.append(spike[1])
        new_spikes_header = pd.DataFrame.from_records(records)
        with tables.open_file(filename, mode="a") as file:
            file.remove_node("/SpikesHeader")
            file.flush()

            file.create_table('/', 'SpikesHeader',
                              obj=dataframeToRecarry(new_spikes_header),
                              createparents=True)

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


def loadEvents(filename, path):
    with tables.open_file(filename, mode="r") as file:
        if path not in file.root:
            return

        event_node = file.get_node(path)

        df_units_header = pd.DataFrame(
            event_node._f_get_child("EventsHeader").read())

        # convert to string
        df_units_header = df_units_header.applymap(lambda x: x.decode(
            'utf-8') if isinstance(x, bytes) else x)

        # if 'UnitType' not in df_units_header.columns:
        #     df_units_header['UnitType'] = 'Unit'

        #     unsorted_pattern = r'(?i)unsorted'
        #     matches = df_units_header['Name'].str.contains(
        #         unsorted_pattern, regex=True)
        #     df_units_header.loc[matches, 'UnitType'] = 'Unsorted'

        #     invalid_pattern = r'(?i)invalid'
        #     matches = df_units_header['Name'].str.contains(
        #         invalid_pattern, regex=True)
        #     df_units_header.loc[matches, 'UnitType'] = 'Invalid'

        timestamps = event_node._f_get_child("TimeStamps").read()
        # waveforms = event_node._f_get_child("Waveforms").read()

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
                }


def _saveHeader(filename, path, ID,
                header: FileHeader | RawsHeader | EventsHeader | SpikesHeader, label=''):
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    where, name = os.path.split(path)
    _deleteHeader(filename=filename, path=path, ID=ID, label=label)
    with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
        if path in file.root:
            table = file.get_node(where=where, name=name)
            if not 'ID' in table.colnames:
                logger.error('Header missing ID field.')
                return
            if label != '':
                if not 'Label' in table.colnames:
                    logger.error('Header missing Label field.')
                    return

            header_dict = header.model_dump(extra='append')
            header_df = pd.DataFrame(
                {k: [v] for k, v in header_dict.items()}, index=[0])
            header_rec = dataframeToRecarry(header_df)

            type_consist = True
            for k, v in table.coldtypes.items():
                if not np.can_cast(header_rec[k].dtype, v):
                    type_consist = False
                    break

            if type_consist:
                table.append([tuple(header.model_dump(extra='append')[col]
                                    for col in table.colnames)])
            else:
                new_headers = recarrayToDictList(table.read())
                records = [header.__class__.model_validate(h, extra='allow').model_dump(extra='append')
                           for h in new_headers]
                records.append(header.model_dump(extra='append'))
                new_header_df = pd.DataFrame.from_records(records)

                file.remove_node(path)
                file.flush()

                file.create_table(where=where, name=name,
                                  obj=dataframeToRecarry(new_header_df),
                                  createparents=True)

        else:
            header_dict = header.model_dump(extra='append')
            header_df = pd.DataFrame(
                {k: [v] for k, v in header_dict.items()}, index=[0])
            file.create_table(where=where, name=name,
                              obj=dataframeToRecarry(header_df),
                              createparents=True)
            file.flush()


def _deleteHeader(filename, path, ID, label=''):
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    where, name = os.path.split(path)
    with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
        if path in file.root:
            table = file.get_node(where=where, name=name)

            if not 'ID' in table.colnames:
                logger.error('Header missing ID field.')
                return

            condiction = f'ID == {ID}'
            if isinstance(ID, str):
                condiction = f'ID == b"{ID}"'

            if label != '':
                if not 'Label' in table.colnames:
                    logger.error('Header missing Label field.')
                    return

                condiction = f'({condiction}) & (Label == b"{label}")'

            rows = table.get_where_list(condiction).tolist()

            if len(rows) > 0:
                [table.remove_row(row) for row in rows]
                table.flush()


def _saveRawsData(filename, path,
                  header: RawsHeader, data: np.ndarray, overwrite=False):
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    where, name = os.path.split(path)

    _deleteData(filename=filename, path=path)

    with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
        data_array = file.create_carray(where, name,
                                        obj=data, title=header.Name,
                                        createparents=True)
        for key, value in header.model_dump(extra='append').items():
            data_array.set_attr(f'header_{key}', value)
        file.flush()


def _saveDiscreteData(filename: str, path: str, header: SpikesHeader | EventsHeader,
                      unit_header: pd.DataFrame, unit_IDs: np.ndarray,
                      timestamps: np.ndarray,  waveforms: np.ndarray = None):
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    where, name = os.path.split(path)
    if waveforms is None and where == '/Spikes':
        logger.error(f'Missing waveforms data when saving {name}.')
        return

    _deleteData(filename=filename, path=path)

    with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
        file.create_carray(where=path, name="TimeStamps",
                           obj=timestamps,  createparents=True)
        file.flush()

        unit_table_name = 'EventsHeader'
        unit_suffix = 'Event'
        if not waveforms is None:
            # is spikes
            unit_table_name = 'UnitsHeader'
            unit_suffix = 'Unit_'
            file.create_carray(where=path, name="Waveforms",
                               obj=waveforms, createparents=True)

        # def _computeH5Location(series):
        #     ID = series['ID']
        #     if isinstance(ID, int):
        #         return f'{path}/{unit_suffix}{ID:02}'
        #     elif isinstance(ID, str):
        #         return f'{path}/{unit_suffix}{ID}'

        unit_header['H5FileName'] = filename
        unit_header['H5Location'] = ''
        unit_header['H5Name'] = ''

        for ID in unit_header['ID']:
            record = unit_header.loc[unit_header['ID'] == ID, :]

            if isinstance(ID, int):
                H5Location = f'{path}/{unit_suffix}{ID:02}'
            elif isinstance(ID, str):
                H5Location = f'{path}/{unit_suffix}{ID}'

            H5Name = 'Indxs'

            unit_header.loc[unit_header['ID'] == ID, 'H5Location'] = H5Location
            unit_header.loc[unit_header['ID'] == ID, 'H5Name'] = H5Name
            # print(record)
            # print(record['NumRecords'])

            if record['NumRecords'].to_list()[0] > 0:
                # print(type(record['H5Location']), type(record['H5Name']))
                file.create_carray(where=H5Location, name=H5Name,
                                   obj=(unit_IDs == ID).nonzero()[0],
                                   createparents=True)
                file.flush()

        unit_header_records = dataframeToRecarry(unit_header)
        file.create_table(where=path, name=unit_table_name,
                          obj=unit_header_records, createparents=True)
        file.flush()


def _deleteData(filename, path):
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(filename))[0]
    where, name = os.path.split(path)
    with tables.open_file(filename, mode='a', title=title, filters=filt) as file:
        if path in file.root:
            file.remove_node(path, recursive=True)
            file.flush()


def saveRaws(filename: str, ID: int, header: RawsHeader, data: np.ndarray, create_link=False):
    """Save raws header and data.

    Args:
        filename (str): _description_
        ID (int): _description_
        header (RawsHeader): _description_
        data (np.ndarray): _description_
        create_link (bool, optional): _description_. Defaults to False.
    """
    raws_data_path = f'/Raws/raw{ID:03}'
    raws_root = os.path.split(raws_data_path)[0]
    header.H5FileName = filename
    header.H5Location = raws_root
    header.H5Name = os.path.split(raws_data_path)[1]
    _saveHeader(filename=filename, path='/RawsHeader', ID=ID, header=header)
    _saveRawsData(filename=filename+'raw', path=raws_data_path,
                  header=header, data=data)

    if create_link:
        with tables.open_file(filename, mode='a') as file:
            if not raws_root in file.root:
                file.create_external_link(
                    where=os.path.split(raws_root)[0],
                    name=os.path.split(raws_root)[1],
                    target=f'{os.path.basename(filename+"raw")}:{raws_root}',
                    createparents=True)
                file.flush()


def deleteRaws(filename: str, ID: int):
    """Delete raws header and data.


    Args:
        filename (str): _description_
        ID (int): _description_
    """
    raws_data_path = f'/Raws/raw{ID:03}'
    _deleteHeader(filename=filename, path='/RawsHeader', ID=ID)
    _deleteData(filename=filename+'raw', path=raws_data_path)


def saveSpikes(filename: str, ID: int, label: str,
               header: SpikesHeader, unit_header: pd.DataFrame, unit_IDs: np.ndarray,
               timestamps: np.ndarray,  waveforms: np.ndarray):
    """Save spikes header and data.

    Args:
        filename (str): _description_
        ID (int): _description_
        label (str): _description_
        header (SpikesHeader): _description_
        unit_header (pd.DataFrame): _description_
        unit_IDs (np.ndarray): _description_
        timestamps (np.ndarray): _description_
        waveforms (np.ndarray): _description_
    """
    spikes_data_path = f'/Spikes/spike{ID:03}'
    if label != 'default':
        spikes_data_path = f'/Spikes/spike{ID:03}{label}'
    header.H5FileName = filename
    header.H5Location = spikes_data_path
    header.H5Name = 'TimeStamps'
    _saveHeader(filename=filename, path='/SpikesHeader',
                ID=ID, label=label, header=header)
    _saveDiscreteData(filename=filename, path=spikes_data_path,
                      header=header, unit_header=unit_header, unit_IDs=unit_IDs,
                      timestamps=timestamps, waveforms=waveforms)


def deleteSpikes(filename: str, ID: int, label: str):
    """Delete spikes header and data.

    Args:
        filename (str): _description_
        ID (int): _description_
        label (str): _description_
    """
    spikes_data_path = f'/Spikes/spike{ID:03}'
    if label != 'default':
        spikes_data_path = f'/Spikes/spike{ID:03}{label}'

    _deleteHeader(filename=filename, path='/SpikesHeader', ID=ID, label=label)
    _deleteData(filename=filename, path=spikes_data_path)


def saveEvents(filename: str, ID: int,
               header: EventsHeader, unit_header: pd.DataFrame, unit_IDs: np.ndarray,
               timestamps: np.ndarray):
    """Save events header and data.

    Args:
        filename (str): _description_
        ID (int): _description_
        header (EventsHeader): _description_
        unit_header (pd.DataFrame): _description_
        unit_IDs (np.ndarray): _description_
        timestamps (np.ndarray): _description_
    """
    events_data_path = f'/Events/event{ID:03}'
    header.H5FileName = filename
    header.H5Location = events_data_path
    header.H5Name = 'TimeStamps'
    _saveHeader(filename=filename, path='/EventsHeader',
                ID=ID, header=header)
    _saveDiscreteData(filename=filename, path=events_data_path,
                      header=header, unit_header=unit_header, unit_IDs=unit_IDs,
                      timestamps=timestamps, waveforms=None)


def deleteEvents(filename: str, ID: int):
    """Delete events header and data.

    Args:
        filename (str): _description_
        ID (int): _description_
    """
    events_data_path = f'/Events/event{ID:03}'

    _deleteHeader(filename=filename, path='/EventsHeader', ID=ID)
    _deleteData(filename=filename, path=events_data_path)


def exportToPyephys(new_filename: str, data_object: SpikeSorterData):
    filt = tables.Filters(complib='zlib', complevel=1)
    title = os.path.splitext(os.path.basename(new_filename))[0]
    overwrite = True
    if overwrite:
        with tables.open_file(new_filename, mode='w', title=title, filters=filt) as file:
            pass
        with tables.open_file(new_filename+'raw', mode='w', title=title, filters=filt) as file:
            pass

    # total_size = 0
    # with tables.open_file(new_filename, mode='r') as h5file:
    #     for node in h5file.walk_nodes("/"):
    #         total_size += node.size_in_memory
    # logger.debug(total_size)

    # logger.debug(os.path.getsize(new_filename))

    # File
    for file_header in data_object._file_headers:
        _saveHeader(filename=new_filename, path='/FileHeader', ID=file_header['ID'],
                    header=FileHeader.model_validate(file_header, extra='allow'))

    # Raws
    for ID in data_object.channel_IDs:
        raws_object = data_object.getRaw(ID, load_data=True)
        raws_header = RawsHeader.model_validate(raws_object.header,
                                                extra='allow')
        saveRaws(filename=new_filename, ID=ID,
                 header=raws_header, data=raws_object.data,
                 create_link=True)

    # Spikes
    for ID in data_object.channel_IDs:
        raws_object = data_object.getRaw(ID)
        for label in raws_object.spikes:
            spikes_object = data_object.getSpike(ID, label, load_data=True)

            if spikes_object == 'Removed':
                deleteSpikes(filename=new_filename, ID=ID, label=label)
                del raws_object._spikes[label]
                continue

            spikes_header = SpikesHeader.model_validate(spikes_object.header,
                                                        extra='allow')

            saveSpikes(filename=new_filename, ID=ID, label=label,
                       header=spikes_header,
                       unit_header=spikes_object.unit_header,
                       unit_IDs=spikes_object.unit_IDs,
                       timestamps=spikes_object.timestamps,
                       waveforms=spikes_object.waveforms)

    # Events
    for ID in data_object.event_IDs:
        events_object = data_object.getEvent(ID)
        events_header = EventsHeader.model_validate(events_object.header,
                                                    extra='allow')
        saveEvents(filename=new_filename, ID=ID,
                   header=events_header,
                   unit_header=events_object.unit_header,
                   unit_IDs=events_object.unit_IDs,
                   timestamps=events_object.timestamps)


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
