import tables
import os
import pandas as pd
import numpy as np


def loadPyephys(filename):
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        # FileHeader = file.get_node("/FileHeader")
        if "/RawsHeader" in file.root:
            raws_header = file.get_node("/RawsHeader").read()
            df_raws_header = pd.DataFrame(raws_header)[
                ['ID', 'Name', 'NumRecords', 'SamplingFreq', 'SigUnits']]

            # convert to string
            df_raws_header = df_raws_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)
        else:
            raise

        spikes_columns = ['ID', 'Label', 'NumUnits',
                          'ReferenceID', 'LowCutOff', 'HighCutOff', 'Threshold']
        if "/SpikesHeader" in file.root:
            spikes_header = file.get_node("/SpikesHeader").read()
            df_spikes_header = pd.DataFrame(spikes_header)

            # add Label field if not have
            if "Label" not in df_spikes_header.columns:
                df_spikes_header = df_spikes_header[[
                    col for col in spikes_columns if col != 'Label']]
                df_spikes_header['Label'] = "default"

            # multi-label test ----------------------------------------------------------------
            test_append = [{
                'ID': 0,
                'Label': "0test",
                'NumUnits': .0,
                'ReferenceID': np.NaN,
                'LowCutOff': .0,
                'HighCutOff': .0,
                'Threshold': .0,
            }]
            df_spikes_header = pd.concat(
                [df_spikes_header, pd.DataFrame.from_records(test_append)])
            #  ----------------------------------------------------------------
            df_spikes_header = df_spikes_header[spikes_columns]

            # convert to string
            df_spikes_header = df_spikes_header.applymap(lambda x: x.decode(
                'utf-8') if isinstance(x, bytes) else x)

            data = pd.merge(df_raws_header, df_spikes_header,
                            how="left", on="ID")

        else:
            data = df_raws_header.copy()
            data[[col for col in spikes_columns if col != 'ID']] = [
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

        data['Label'].fillna("default", inplace=True)

        data['ReferenceID'] = data['ReferenceID'].astype(pd.Int64Dtype())
    return data


def loadRaws(filename, chan_ID):
    chan_ID = int(chan_ID)
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if "/Raws" in file.root:
            raws = file.get_node("/Raws")
            with tables.open_file(raws(mode='r')._v_file.filename, mode="r") as rawsfile:
                data = rawsfile.get_node("/Raws/raw" + str(chan_ID).zfill(3))
                return data[:]
        else:
            raise


def loadSpikes(filename, chan_ID, label):
    chan_ID = int(chan_ID)
    basename, extname = os.path.splitext(filename)
    if extname == "h5raw":
        filename = ".".join(basename, "h5")

    with tables.open_file(filename, mode="r") as file:
        if "/Spikes" in file.root:
            spikes = file.get_node("/Spikes")

            try:
                spike_chan = spikes._f_get_child(
                    "spike" + str(chan_ID).zfill(3))
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

                # if 'Invalid' not in df_units_header['UnitType'].values:
                #     new_unit_ID = df_units_header['ID'].max() + 1
                #     new_row = {
                #         'H5Location': f'/Spikes/spike{str(chan_ID).zfill(3)}/Unit_{str(new_unit_ID).zfill(2)}',
                #         'H5Name': 'Indxs',
                #         'ID': new_unit_ID,
                #         'Name':  f'CH{chan_ID + 1}_Unit_{str(new_unit_ID).zfill(2)}_Invalid',
                #         'NumRecords': 0,
                #         'ParentID': chan_ID,
                #         'ParentType': 'Spikes',
                #         'Type': 'Unit',
                #         'UnitType': 'Invalid'}
                #     df_units_header = df_units_header.append(
                #         new_row, ignore_index=True)

                timestamps = spike_chan._f_get_child("TimeStamps").read()
                waveforms = spike_chan._f_get_child("Waveforms").read()

            except tables.NodeError:
                print("The /Spikes node does not contain the spike" +
                      str(chan_ID).zfill(3) + " node.")

                return {"unitInfo": None,
                        "unitID": None,
                        "timestamps": None,
                        "waveforms": None}
            # get units id
            unitID = np.zeros(len(timestamps))
            not_zero_units = df_units_header[df_units_header["NumRecords"] > 0]
            for unit in not_zero_units.index:
                unit_h5_name = "/".join([not_zero_units.loc[unit, "H5Location"],
                                        not_zero_units.loc[unit, "H5Name"]])
                ind = file.get_node(unit_h5_name).read()
                unitID[ind] = not_zero_units.loc[unit, "ID"]

            return {"unitInfo": df_units_header,
                    "unitID": unitID,
                    "timestamps": timestamps,
                    "waveforms": waveforms}
